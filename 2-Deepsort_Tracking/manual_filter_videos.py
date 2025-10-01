# -*- coding: utf-8 -*-
"""
人工筛选
- 扫描 results/cut/cross 和 results/cut/notcross 下的视频
- 每个视频循环播放，直到按下 y / n / q 确认
- y = 保留（复制到 cleaned_data/cross 或 notcross）
- n = 丢弃（不保存）
- q = 跳过（不保存）
"""

import os
import cv2
import shutil

INPUT_ROOT = os.path.join("results", "cut")   #  改成 cut
OUTPUT_ROOT = "cleaned_data"
CLASSES = ["cross", "notcross"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def filter_videos():
    for cls in CLASSES:
        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        ensure_dir(output_dir)

        videos = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
        videos.sort()

        print(f"\n===== 开始筛选 {cls} 类，共 {len(videos)} 个视频 =====")

        for vid in videos:
            src = os.path.join(input_dir, vid)
            cap = cv2.VideoCapture(src)

            if not cap.isOpened():
                print(f"️ 无法打开视频: {src}")
                continue

            print(f"\n正在播放: {vid}")

            # 读取视频帧缓存，避免循环时频繁读盘
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            idx = 0
            while True:
                frame = frames[idx]
                cv2.imshow(f"{cls} - {vid}", frame)
                key = cv2.waitKey(30) & 0xFF

                # 按键判断
                if key == ord("y"):  # 保留
                    dst = os.path.join(output_dir, vid)
                    shutil.copy(src, dst)
                    print(f" 保留: {vid}")
                    break
                elif key == ord("n"):  # 丢弃
                    print(f" 丢弃: {vid}")
                    break
                elif key == ord("q"):  # 跳过
                    print(f"️ 跳过: {vid}")
                    break

                # 循环播放
                idx = (idx + 1) % len(frames)

            cv2.destroyAllWindows()

    print("\n 筛选完成，结果保存在 cleaned_data/ 下")


if __name__ == "__main__":
    filter_videos()
