# -*- coding: utf-8 -*-
"""
资源采样器：后台线程周期性采集本进程 CPU / 内存，并尝试读取昇腾 NPU 利用率与显存。
- CPU% / RSS 内存：psutil（始终可用）
- NPU 利用率 / NPU 显存：解析 `npu-smi info`（香橙派 AIpro / 昇腾 310B 自带），
  若命令不存在或解析失败，对应字段记为 None，不影响其它指标采集。

采样结果存在 self.records，每条记录：
    {ts, cpu_percent, rss_mb, npu_util, npu_mem_mb}
"""
import re
import shutil
import subprocess
import threading
import time

import psutil


class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.5, pid=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process(pid) if pid else psutil.Process()
        self.records = []
        self._stop_event = threading.Event()
        self._npu_smi_available = shutil.which("npu-smi") is not None
        # 预热 cpu_percent：第一次调用必须返回 0，之后才有意义
        self.proc.cpu_percent(interval=None)

    def _read_npu_smi(self):
        """返回 (npu_util_percent, npu_mem_used_mb)；失败返回 (None, None)。"""
        if not self._npu_smi_available:
            return None, None
        try:
            out = subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, timeout=2
            ).stdout
        except Exception:
            return None, None

        # 找到含 NPU 利用率百分比的行；兼容不同固件版本的列布局
        util, mem = None, None
        for line in out.splitlines():
            if "%" not in line:
                continue
            pcts = re.findall(r"(\d+)\s*%", line)
            mems = re.findall(r"(\d+)\s*MB", line, flags=re.IGNORECASE)
            if pcts:
                util = float(pcts[0])
                if mems:
                    mem = float(mems[0])
                break
        return util, mem

    def run(self):
        t0 = time.time()
        while not self._stop_event.is_set():
            try:
                rss_mb = self.proc.memory_info().rss / 1024.0 / 1024.0
                cpu = self.proc.cpu_percent(interval=None)
                npu_util, npu_mem = self._read_npu_smi()
                self.records.append({
                    "ts": time.time() - t0,
                    "cpu_percent": cpu,
                    "rss_mb": rss_mb,
                    "npu_util": npu_util,
                    "npu_mem_mb": npu_mem,
                })
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    # ---------- 汇总统计 ----------
    def summary(self):
        if not self.records:
            return {}
        rss = [r["rss_mb"] for r in self.records]
        cpu = [r["cpu_percent"] for r in self.records]
        npu = [r["npu_util"] for r in self.records if r["npu_util"] is not None]
        return {
            "rss_avg_mb": sum(rss) / len(rss),
            "rss_peak_mb": max(rss),
            "rss_start_mb": rss[0],
            "rss_end_mb": rss[-1],
            "rss_growth_mb": rss[-1] - rss[0],
            "cpu_avg_percent": sum(cpu) / len(cpu),
            "cpu_peak_percent": max(cpu),
            "npu_util_avg": (sum(npu) / len(npu)) if npu else None,
            "npu_util_peak": max(npu) if npu else None,
            "samples": len(self.records),
        }
#（注：内容由AI生成）
