# -*- coding: utf-8 -*-
"""
pytest 自动化用例：
  1) test_image_samples  —— 批量跑 samples/images 下每个样本，捕获异常，记录耗时/峰值内存
  2) test_video_stream   —— 若 samples/video.mp4 存在，跑前 N 帧视频流，验证不崩溃、FPS 达标
运行：
  pytest test_pipeline.py -v
  报告自动写到 reports/pytest_report.{json,md}
"""
import os
import sys
import time
import traceback

import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_up import main, adaptive_kde_heatmap  # noqa: E402
from resource_monitor import ResourceMonitor  # noqa: E402


@pytest.mark.parametrize("img_path", pytest.lazy_fixture("image_samples"))
def test_image_samples(session, results, img_path):
    name = os.path.basename(img_path)
    rec = {"case": name, "status": "FAIL", "total_ms": None,
           "det_ms": None, "kde_ms": None, "peak_rss_mb": None,
           "n_det": None, "error": ""}
    img = cv2.imread(img_path)
    assert img is not None, f"图片读取失败: {img_path}"

    mon = ResourceMonitor(interval=0.2)
    mon.start()
    try:
        t0 = time.perf_counter()
        _, dets, pts = main(session, img)
        t_det = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        adaptive_kde_heatmap(pts, img.shape[0], img.shape[1])
        t_kde = (time.perf_counter() - t1) * 1000.0

        rec.update(status="PASS", det_ms=round(t_det, 2), kde_ms=round(t_kde, 2),
                   total_ms=round(t_det + t_kde, 2), n_det=len(dets))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        pytest.fail(f"样本 {name} 推理异常: {e}")
    finally:
        mon.stop()
        s = mon.summary()
        rec["peak_rss_mb"] = round(s.get("rss_peak_mb", 0), 1)
        results.append(rec)


def test_video_stream(session, results, video_path="samples/video.mp4", max_frames=300,
                      min_fps=5.0):
    """视频流冒烟：跑前 max_frames 帧，要求无异常且 FPS 不低于 min_fps。"""
    rec = {"case": f"video:{os.path.basename(video_path)}", "status": "FAIL",
           "total_ms": None, "det_ms": None, "kde_ms": None,
           "peak_rss_mb": None, "n_det": None, "error": ""}
    if not os.path.exists(video_path):
        pytest.skip(f"未找到视频样本 {video_path}")

    cap = cv2.VideoCapture(video_path)
    mon = ResourceMonitor(interval=0.2)
    mon.start()
    lat = []
    total_det = 0
    try:
        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            _, dets, pts = main(session, frame)
            t_det = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            adaptive_kde_heatmap(pts, frame.shape[0], frame.shape[1])
            t_kde = (time.perf_counter() - t1) * 1000.0
            lat.append(t_det + t_kde)
            total_det += len(dets)
        assert lat, "未成功处理任何视频帧"
        avg_ms = sum(lat) / len(lat)
        fps = 1000.0 / avg_ms
        rec.update(status="PASS", det_ms=round(avg_ms, 2),
                   total_ms=round(avg_ms, 2), n_det=total_det)
        assert fps >= min_fps, f"FPS={fps:.1f} 低于阈值 {min_fps}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        mon.stop()
        cap.release()
        s = mon.summary()
        rec["peak_rss_mb"] = round(s.get("rss_peak_mb", 0), 1)
        results.append(rec)
#（注：内容由AI生成）
