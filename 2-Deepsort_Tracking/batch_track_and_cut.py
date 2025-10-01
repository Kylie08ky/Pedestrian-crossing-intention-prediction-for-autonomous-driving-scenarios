# -*- coding: utf-8 -*-
"""
批量处理 datasets/cross 与 datasets/notcross：
- YOLOv8+DeepSORT 跟踪并输出带框视频
- 绿幕抠图：背景纯绿，仅保留检测框区域
输出保持 cross / notcross 分类

使用前提：
- 本项目内应存在 tracker/vars 与 AIDetector_pytorch.Detector
- 已正确配置检测/跟踪权重与依赖

运行方式：
    python batch_track_and_cut.py
"""

import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm

# 你项目里已有的依赖
from tracker import vars
from AIDetector_pytorch import Detector


# ---------- 基础配置 ----------
INPUT_ROOT = os.path.join(os.getcwd(), "datasets")
CLASSES = ["cross", "notcross"]
OUTPUT_ROOT = os.path.join(os.getcwd(), "results")

# 识别为视频的扩展名（可自行增减）
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# 是否在写 tracker 视频时，对显示图 resize 到指定高度（便于统一尺寸）
RESIZE_TRACKER_HEIGHT = 640  # 若不想缩放，可设为 None


# ---------- 工具函数 ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_video_file(name: str) -> bool:
    return os.path.splitext(name)[1] in VIDEO_EXTS


def green_screen_cut(image, rectangles):
    """
    将非检测框区域置为纯绿，仅保留框内原始像素
    rectangles: [(x1,y1,x2,y2), ...]  坐标允许 float，将被取整
    """
    green_bg = np.zeros_like(image)
    green_bg[:, :, 1] = 255  # 纯绿色背景
    if rectangles is None:
        return green_bg
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(image.shape[1], x2)), int(min(image.shape[0], y2))
        if x2 > x1 and y2 > y1:
            green_bg[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return green_bg


def process_video(input_path: str, out_tracker_path: str, out_cut_path: str):
    """
    对单个视频执行：
      - Detector().feedCap(im) -> result(带框帧), boxes(用于绿幕抠图)
      - 写出 tracker 视频与 cut 视频
    """
    # 每个视频单独初始化（清空状态）
    vars.init()
    det = Detector()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # 兜底

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 绿幕视频尺寸 = 原视频尺寸
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_cut = cv2.VideoWriter(out_cut_path, fourcc, fps, (w, h))

    writer_tracker = None  # tracker 视频尺寸等首帧确定

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # 检测+跟踪
        result_dict, boxes = det.feedCap(frame)  # boxes: [(x1,y1,x2,y2), ...]
        cut_frame = green_screen_cut(frame, boxes)
        writer_cut.write(cut_frame)

        # 渲染带框帧（Detector 返回的可视化结果）
        vis = result_dict.get("frame", frame)

        # 可选：统一高度
        if RESIZE_TRACKER_HEIGHT is not None:
            vis = imutils.resize(vis, height=RESIZE_TRACKER_HEIGHT)

        # 按首帧尺寸初始化 writer_tracker
        if writer_tracker is None:
            th, tw = vis.shape[0], vis.shape[1]
            writer_tracker = cv2.VideoWriter(out_tracker_path, fourcc, fps, (tw, th))

        writer_tracker.write(vis)

    cap.release()
    writer_cut.release()
    if writer_tracker is not None:
        writer_tracker.release()


# ---------- 主流程 ----------
def main():
    # 创建输出目录
    out_tracker_root = os.path.join(OUTPUT_ROOT, "tracker")
    out_cut_root = os.path.join(OUTPUT_ROOT, "cut")
    for c in CLASSES:
        ensure_dir(os.path.join(out_tracker_root, c))
        ensure_dir(os.path.join(out_cut_root, c))

    for c in CLASSES:
        in_dir = os.path.join(INPUT_ROOT, c)
        if not os.path.isdir(in_dir):
            print(f"️ 输入目录不存在，跳过：{in_dir}")
            continue

        videos = [f for f in os.listdir(in_dir) if is_video_file(f)]
        videos.sort()

        print(f"\n===== 处理类别：{c}，共 {len(videos)} 个视频 =====")
        for name in tqdm(videos):
            src = os.path.join(in_dir, name)
            stem, ext = os.path.splitext(name)

            out_tracker = os.path.join(out_tracker_root, c, f"{stem}_tracker.mp4")
            out_cut = os.path.join(out_cut_root, c, f"{stem}_cut.mp4")

            # 已存在则跳过（避免重复算）
            if os.path.exists(out_tracker) and os.path.exists(out_cut):
                continue

            try:
                process_video(src, out_tracker, out_cut)
            except Exception as e:
                print(f" 处理失败：{src}\n   原因：{e}")

    print("\n 全部完成,输出见 results/tracker 与 results/cut（按类别划分）。")


if __name__ == "__main__":
    main()
