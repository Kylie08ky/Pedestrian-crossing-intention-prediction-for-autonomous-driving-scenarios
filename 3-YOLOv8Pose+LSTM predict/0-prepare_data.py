import os
import json
import numpy as np
import torch

# 参数
json_dir_cross = "results_yolov8pose/cross"
json_dir_notcross = "results_yolov8pose/notcross"
max_frames = 50
keypoint_num = 17  # YOLOv8-Pose 默认17个关键点 (COCO格式)

def normalize_keypoints(frame_points):
    """对单帧关键点做平移 + 尺度归一化"""
    frame_points = np.array(frame_points).reshape(-1, 2)  # [num_points, 2]

    # 平移：以左右臀部中点作为参考（COCO: 左臀=11, 右臀=12）
    if frame_points.shape[0] >= 13:
        center = (frame_points[11] + frame_points[12]) / 2.0
    else:
        center = np.mean(frame_points, axis=0)
    frame_points = frame_points - center

    # 尺度：用肩宽归一化（左肩=5, 右肩=6）
    if frame_points.shape[0] >= 7:
        shoulder_width = np.linalg.norm(frame_points[5] - frame_points[6])
        if shoulder_width > 1e-6:
            frame_points = frame_points / shoulder_width

    return frame_points.flatten()

def json_to_tensor(json_path, max_frames=50, keypoint_num=17):
    """把单个视频的JSON转换为 [帧数, 特征维度]"""
    with open(json_path, 'r') as f:
        results = json.load(f)

    video_data = []
    for i, frame in enumerate(results):
        if i >= max_frames:
            break
        if len(frame["objects"]) == 0:  # 无检测
            video_data.append([0] * (keypoint_num * 2))
            continue

        # 取第一个人的关键点
        keypoints = frame["objects"][0]["keypoints"]
        frame_xy = []
        for (x, y) in keypoints:
            frame_xy.extend([x, y])

        norm_frame = normalize_keypoints(frame_xy)
        video_data.append(norm_frame)

    while len(video_data) < max_frames:
        video_data.append([0] * (keypoint_num * 2))

    return np.array(video_data, dtype=np.float32)

# 收集数据
X, y = [], []

for file in os.listdir(json_dir_cross):
    subdir = os.path.join(json_dir_cross, file)
    json_path = os.path.join(subdir, f"{file}.json")
    if os.path.exists(json_path):
        arr = json_to_tensor(json_path, max_frames, keypoint_num)
        X.append(arr)
        y.append(0)  # cross

for file in os.listdir(json_dir_notcross):
    subdir = os.path.join(json_dir_notcross, file)
    json_path = os.path.join(subdir, f"{file}.json")
    if os.path.exists(json_path):
        arr = json_to_tensor(json_path, max_frames, keypoint_num)
        X.append(arr)
        y.append(1)  # notcross

X = np.array(X)
y = np.array(y)

print("X.shape:", X.shape)
print("y.shape:", y.shape)

torch.save((torch.tensor(X), torch.tensor(y)), "dataset_norm.pt")
print("已保存为 dataset_norm.pt")
