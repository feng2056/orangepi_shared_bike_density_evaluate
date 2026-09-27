import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import psutil
import matplotlib
matplotlib.use("Agg")  # 无头服务器模式，不调用X11，只输出图片
import matplotlib.pyplot as plt
import numpy as np
from ais_bench.infer.interface import InferSession

# ========= 关键修复：别名导入，防止main名字冲突 =========
from speed_up import main as pipeline_main
from speed_up import adaptive_kde_heatmap

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


class ResourceMonitor:
    def __init__(self, sample_interval=0.2):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.records = []
        self._running = False

    def _get_npu_util(self):
        """香橙派AIpro 昇腾310B：读取系统proc节点获取NPU利用率，不需要npu‑smi"""
        npu_util_file = "/proc/ascend/device0/util"
        if not os.path.exists(npu_util_file):
            return None
        try:
            with open(npu_util_file, "r", encoding="utf‑8") as f:
                val = int(f.read().strip())
            return float(val)
        except Exception:
            return None


    def snapshot(self, timestamp):
        cpu_percent = self.process.cpu_percent(interval=None)
        mem_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        npu_util = self._get_npu_util()
        self.records.append({
            "ts": timestamp,
            "cpu": cpu_percent,
            "rss_mb": mem_rss_mb,
            "npu_util": npu_util
        })

    def get_stats(self):
        if not self.records:
            return {}
        rss_list = [r["rss_mb"] for r in self.records]
        cpu_list = [r["cpu"] for r in self.records]
        npu_list = [r["npu_util"] for r in self.records if r["npu_util"] is not None]
        return {
            "rss_start_mb": rss_list[0],
            "rss_end_mb": rss_list[-1],
            "rss_max_mb": max(rss_list),
            "rss_mean_mb": float(np.mean(rss_list)),
            "cpu_mean": float(np.mean(cpu_list)),
            "cpu_max": max(cpu_list),
            "npu_mean": float(np.mean(npu_list)) if npu_list else None,
            "npu_max": max(npu_list) if npu_list else None
        }

    def plot(self, save_path: Path):
        if len(self.records) < 2:
            return
        ts = [r["ts"] - self.records[0]["ts"] for r in self.records]
        rss = [r["rss_mb"] for r in self.records]
        cpu = [r["cpu"] for r in self.records]
        npu = [r["npu_util"] if r["npu_util"] is not None else np.nan for r in self.records]

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("RSS Memory (MB)", color="#d62728")
        ax1.plot(ts, rss, color="#d62728", label="RSS Memory MB")
        ax1.tick_params(axis='y', labelcolor="#d62728")

        ax2 = ax1.twinx()
        ax2.set_ylabel("CPU% / NPU%", color="#1f77b4")
        ax2.plot(ts, cpu, color="#1f77b4", label="CPU %", alpha=0.7)
        ax2.plot(ts, npu, color="#2ca02c", label="NPU Util %", alpha=0.7)
        ax2.tick_params(axis='y', labelcolor="#1f77b4")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()


def run_one_frame(session, img):
    """执行一帧完整流水线：检测 + KDE热力图"""
    t0 = time.perf_counter()
    # =========修复：使用别名 pipeline_main=========
    _, dets, pts = pipeline_main(session, img)
    t1 = time.perf_counter()
    _ = adaptive_kde_heatmap(pts, img.shape[0], img.shape[1])
    t2 = time.perf_counter()
    total_ms = (t2 - t0) * 1000
    infer_ms = (t1 - t0) * 1000
    return total_ms, infer_ms, len(dets)


def bench_image(session, img_dir, runs):
    img_dir = Path(img_dir)
    img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"{img_dir} 没有找到图片")
    monitor = ResourceMonitor(sample_interval=0.2)
    frame_results = []

    # numba jit预热，跑一张
    print("[Warmup] numba JIT compile ...")
    warm_img = cv2.imread(str(img_paths[0]))
    run_one_frame(session, warm_img)

    for idx in range(runs):
        p = img_paths[idx % len(img_paths)]
        img = cv2.imread(str(p))
        ts = time.perf_counter()
        total_ms, infer_ms, obj_cnt = run_one_frame(session, img)
        monitor.snapshot(ts)
        frame_results.append({
            "run_idx": idx,
            "file": str(p.name),
            "total_ms": total_ms,
            "infer_ms": infer_ms,
            "object_count": obj_cnt
        })
        print(f"run{idx:3d} | total={total_ms:.2f}ms infer={infer_ms:.2f}ms obj={obj_cnt}")
    return frame_results, monitor


def bench_video(session, video_path, loop=False, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("打开视频失败")
    monitor = ResourceMonitor(0.2)
    frame_results = []
    frame_idx = 0
    print("[Warmup video jit ...]")
    ret, frame = cap.read()
    if ret:
        run_one_frame(session, frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        ts = time.perf_counter()
        try:
            total_ms, infer_ms, obj_cnt = run_one_frame(session, frame)
        except Exception as e:
            print(f"Frame {frame_idx} Exception: {str(e)}")
            frame_results.append({"frame_idx": frame_idx, "error": str(e)})
            continue
        monitor.snapshot(ts)
        frame_results.append({
            "frame_idx": frame_idx,
            "total_ms": total_ms,
            "infer_ms": infer_ms,
            "object_count": obj_cnt
        })
        frame_idx +=1
        if max_frames is not None and frame_idx >= max_frames:
            break
    cap.release()
    return frame_results, monitor


def bench_stress(session, video_path, duration_sec):
    print(f"===== 压力测试，持续 {duration_sec} 秒 =====")
    cap = cv2.VideoCapture(str(video_path))
    monitor = ResourceMonitor(0.2)
    frame_results = []
    start_time = time.perf_counter()
    frame_idx = 0
    # warmup
    ret, frame = cap.read()
    if ret:
        run_one_frame(session, frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    leak_threshold_mb = 50.0
    while time.perf_counter() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue
        ts = time.perf_counter()
        try:
            total_ms, infer_ms, obj_cnt = run_one_frame(session, frame)
        except Exception as e:
            print(f"[CRASH] frame {frame_idx} error: {e}")
            frame_results.append({"frame_idx":frame_idx,"error":str(e)})
            continue
        monitor.snapshot(ts)
        frame_results.append({
            "frame_idx":frame_idx,
            "total_ms":total_ms,
            "infer_ms":infer_ms,
            "object_count":obj_cnt
        })
        frame_idx += 1
    cap.release()
    stat = monitor.get_stats()
    mem_growth = stat["rss_end_mb"] - stat["rss_start_mb"]
    print(f"压力测试结束，总帧数 {frame_idx}")
    print(f"内存增长: {mem_growth:.2f} MB, leak_threshold {leak_threshold_mb} MB")
    if mem_growth > leak_threshold_mb:
        print(f"⚠️警告：检测疑似内存泄漏，内存增长 {mem_growth:.2f}MB")
    return frame_results, monitor


def calc_summary(frame_list):
    valid = [x for x in frame_list if "error" not in x]
    if len(valid) ==0:
        return {"valid_frames":0}
    total_arr = [d["total_ms"] for d in valid]
    infer_arr = [d["infer_ms"] for d in valid]
    s = {
        "valid_frames": len(valid),
        "error_frames": len(frame_list)-len(valid),
        "total_ms_mean": float(np.mean(total_arr)),
        "total_ms_min": float(np.min(total_arr)),
        "total_ms_max": float(np.max(total_arr)),
        "total_ms_p95": float(np.percentile(total_arr,95)),
        "infer_ms_mean": float(np.mean(infer_arr)),
        "fps_avg": 1000.0 / float(np.mean(total_arr))
    }
    return s


def save_output(mode, frame_list, monitor:ResourceMonitor):
    summary = calc_summary(frame_list)
    res_stats = monitor.get_stats()
    out_json = {
        "mode":mode,
        "perf":summary,
        "resource": res_stats
    }
    json_path = REPORT_DIR / f"{mode}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    csv_path = REPORT_DIR / f"{mode}_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_list[0].keys()) if frame_list else [])
        writer.writeheader()
        writer.writerows(frame_list)

    # latency plot
    if summary.get("valid_frames",0) > 1:
        valid = [x for x in frame_list if "error" not in x]
        idx = [i for i,_ in enumerate(valid)]
        total_ms = [d["total_ms"] for d in valid]
        plt.figure(figsize=(12,4))
        plt.plot(idx, total_ms, label="total latency(ms)", color="#ff7f0e")
        plt.xlabel("Frame Index")
        plt.ylabel("Latency ms")
        plt.title(f"{mode} latency curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(REPORT_DIR / f"{mode}_latency_curve.png", dpi=120)
        plt.close()

    monitor.plot(REPORT_DIR / f"{mode}_resource_curve.png")
    print(f"报告输出目录 {REPORT_DIR.resolve()}")
    print(json.dumps(out_json, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    p_img = subparsers.add_parser("image")
    p_img.add_argument("--img-dir", required=True)
    p_img.add_argument("--runs", type=int, default=20)

    p_vid = subparsers.add_parser("video")
    p_vid.add_argument("--video", required=True)
    p_vid.add_argument("--loop", action="store_true")
    p_vid.add_argument("--max-frames", type=int, default=None)

    p_stress = subparsers.add_parser("stress")
    p_stress.add_argument("--video", required=True)
    p_stress.add_argument("--duration", type=int, default=1800, help="seconds")

    args = parser.parse_args()
    print("加载模型 best.om ...")
    session = InferSession(device_id=0, model_path="best.om")
    try:
        if args.mode == "image":
            frames, monitor = bench_image(session, args.img_dir, args.runs)
        elif args.mode == "video":
            frames, monitor = bench_video(session, args.video, loop=args.loop, max_frames=args.max_frames)
        elif args.mode == "stress":
            frames, monitor = bench_stress(session, args.video, args.duration)
        else:
            raise NotImplementedError
        save_output(args.mode, frames, monitor)
    except Exception as e:
        import traceback
        traceback.print_exc()
    # =========修复：删除 session.release()！ais_bench没有这个接口=========
    # 对象销毁自动释放acl资源


if __name__ == "__main__":
    main()

