# -*- coding: utf-8 -*-
"""
pytest 全局夹具：
- 加载一次 OM 模型会话（session 级，所有用例复用，避免反复加载 NPU）
- 预热 numba JIT
- 收集批量图片样本目录（默认 samples/images，可用环境变量 TEST_SAMPLES 覆盖）
- 提供一个共享结果容器 results，测试结束后由 pytest_sessionfinish 写报告
"""
import json
import os
import sys
import time

import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_up import InferSession, adaptive_kde_heatmap, OM_MODEL  # noqa: E402
from resource_monitor import ResourceMonitor  # noqa: E402

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images():
    d = os.environ.get("TEST_SAMPLES", "samples/images")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(IMG_EXTS)]


@pytest.fixture(scope="session")
def session():
    print(f"\n加载模型 {OM_MODEL} ...")
    s = InferSession(device_id=0, model_path=OM_MODEL)
    # 预热：触发 numba JIT 编译
    imgs = _list_images()
    if imgs:
        img = cv2.imread(imgs[0])
        for _ in range(3):
            from speed_up import main as _main
            _, _, pts = _main(s, img.copy())
            adaptive_kde_heatmap(pts, img.shape[0], img.shape[1])
    yield s
    s.release()


@pytest.fixture(scope="session")
def image_samples():
    files = _list_images()
    if not files:
        pytest.skip("samples/images 下没有图片样本（可设置环境变量 TEST_SAMPLES 指定目录）")
    return files


@pytest.fixture(scope="session")
def results(pytestconfig):
    """跨用例共享的结果列表，每个用例写入一条记录；测试结束后写报告。"""
    lst = []
    pytestconfig._results = lst
    return lst


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    """测试全部结束后，把 results 写成 JSON + Markdown 报告。"""
    yield  # 让其它 hook 先跑完
    res = getattr(session.config, "_results", None)
    if res is None:
        return
    json_path = os.path.join(REPORT_DIR, "pytest_report.json")
    md_path = os.path.join(REPORT_DIR, "pytest_report.md")
    with open(json_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    passed = sum(1 for r in res if r["status"] == "PASS")
    failed = sum(1 for r in res if r["status"] == "FAIL")
    lat = [r["total_ms"] for r in res if r.get("total_ms")]
    with open(md_path, "w") as f:
        f.write("# 自动化测试报告\n\n")
        f.write(f"- 用例总数：{len(res)}\n- 通过：{passed}\n- 失败：{failed}\n")
        if lat:
            f.write(f"- 平均耗时：{sum(lat)/len(lat):.1f} ms，"
                    f"最大：{max(lat):.1f} ms，最小：{min(lat):.1f} ms\n")
        f.write("\n| 样本 | 状态 | 耗时(ms) | det(ms) | kde(ms) | 峰值RSS(MB) | 检测数 | 备注 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in res:
            f.write(f"| {r['case']} | {r['status']} | {r.get('total_ms','-')} "
                    f"| {r.get('det_ms','-')} | {r.get('kde_ms','-')} "
                    f"| {r.get('peak_rss_mb','-')} | {r.get('n_det','-')} "
                    f"| {r.get('error','')[:80]} |\n")
    print(f"\n[REPORT] JSON: {json_path}\n[REPORT] Markdown: {md_path}")
#（注：内容由AI生成）
