# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.spatial import KDTree
from ais_bench.infer.interface import InferSession
from numba import jit
import threading
import time
# ===================== 全局配置 =====================
# 类别定义
CLASSES = {
    0: 'shared_bicycle'
}
# 置信度阈值
CONFIDENCE = 0.4
# NMS 的 IoU 阈值
IOU = 0.45
# KDE热力图参数（原始参数完全保留）
K_NEAREST = 10
SIGMA_MIN = 8
SIGMA_MAX = 35
# 文件路径
OM_MODEL = "best.om"
TEST_IMG = "test.jpg"
SAVE_HEATMAP = "heatmap_result.jpg"
SAVE_DET_IMG = "det_output.jpg"
# 随机框颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# ====================================================

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = "{} {:.2f}".format(CLASSES[class_id], confidence)
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_width, label_height = label_size
    label_x = x
    label_y = y - 10 if y - 10 > label_height else y + 10
    cv2.rectangle(img, (label_x, label_y - label_height),
                  (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main(session, original_image):
    height, width, _ = original_image.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(640, 640), swapRB=True)
    outputs = session.infer(feeds=blob, mode="static")
    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= CONFIDENCE:
            box = [
                (outputs[0][i][0] - outputs[0][i][2] / 2) * scale,
                (outputs[0][i][1] - outputs[0][i][3] / 2) * scale,
                outputs[0][i][2] * scale,
                outputs[0][i][3] * scale,
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE, IOU, 0.5)
    detections = []
    center_points = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0
        center_points.append([cx, cy])
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(x),
            round(y),
            round(x + w),
            round(y + h)
        )
    return original_image, detections, np.array(center_points)

# -------------------------- JIT加速 + 3σ裁剪核心函数 --------------------------
@jit(nopython=True)
def gaussian_kernel_region(heatmap, cx, cy, sigma, img_h, img_w):
    # 3σ截断范围，高斯外面区域权重几乎为0，可以直接跳过计算
    radius = int(3 * sigma)
    x_start = max(0, int(cx - radius))
    x_end = min(img_w, int(cx + radius) + 1)
    y_start = max(0, int(cy - radius))
    y_end = min(img_h, int(cy + radius) + 1)

    # 局部网格，不再生成整张图meshgrid
    for yy in range(y_start, y_end):
        for xx in range(x_start, x_end):
            dx = xx - cx
            dy = yy - cy
            dist_sq = dx * dx + dy * dy
            weight = np.exp(- dist_sq / (2 * sigma * sigma))
            heatmap[yy, xx] += weight
    return heatmap

def adaptive_kde_heatmap(points, h, w):
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float32)
    tree = KDTree(points)
    heatmap = np.zeros((h, w), dtype=np.float32)

    for (x, y) in points:
        dists, _ = tree.query([x, y], k=K_NEAREST)
        mean_dist = np.mean(dists[1:])
        sigma = np.clip(mean_dist, SIGMA_MIN, SIGMA_MAX)
        # 使用JIT局部高斯计算 +3σ裁剪
        heatmap = gaussian_kernel_region(heatmap, x, y, sigma, h, w)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# 线程包装函数，用于并行
def kde_task(points, h, w, result_container):
    hm = adaptive_kde_heatmap(points, h, w)
    result_container.append(hm)

if __name__ == "__main__":
    # 加载om推理会话（ais_bench自动处理NPU硬件）
    session = InferSession(device_id=0, model_path=OM_MODEL)
    # 读取图片
    image = cv2.imread(TEST_IMG)
    if image is None:
        raise FileNotFoundError(f"图片{TEST_IMG}不存在，请检查路径")
    orig_h, orig_w = image.shape[:2]

    # NPU检测推理（主线程）
    det_img, dets, points = main(session, image)
    print(f"✅ 检测到共享单车总数：{len(points)}")

    # 启动子线程执行KDE计算（并行）
    res_container = []
    t = threading.Thread(target=kde_task, args=(points, orig_h, orig_w, res_container))
    t.start()
    t.join()
    heat_map = res_container[0]

    heat_color = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    total_density = np.sum(heat_map)
    print(f"✅ KDE全局密度积分值：{total_density:.2f}")

    # 保存结果
    cv2.imwrite(SAVE_DET_IMG, det_img)
    cv2.imwrite(SAVE_HEATMAP, heat_color)
    print(f"✅ 检测图保存：{SAVE_DET_IMG}")
    print(f"✅ 热力图保存：{SAVE_HEATMAP}")

    t1 = time.time()
    heat_map = adaptive_kde_heatmap(points, orig_h, orig_w)
    print(f"KDE计算耗时 = {time.time()-t1:.3f} s")

    # 释放推理资源
    session.release()
