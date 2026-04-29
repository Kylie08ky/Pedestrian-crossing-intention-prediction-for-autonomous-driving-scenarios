"""
inference.py — 行人过街意图预测端到端推理脚本
输入：单个视频文件路径
输出：cross（过街）/ notcross（未过街）+ 概率 + 可视化视频

Pipeline:
  视频 → YOLOv8-Pose逐帧检测 → 关键点归一化 → LSTM分类 → 结果可视化

用法:
  python inference.py --video path/to/video.mp4 --model LSTM_seq_best.pth
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ========================
# 超参数（必须和训练时一致）
# ========================
MAX_FRAMES = 50       # 序列长度
KEYPOINT_NUM = 17     # YOLOv8-Pose COCO格式关键点数
INPUT_DIM = 34        # 17 * 2
HIDDEN_DIM = 64
OUTPUT_DIM = 2
CONF_THRESHOLD = 0.25  # YOLO检测置信度阈值（降低以提高召回率）

LABELS = {0: "CROSSING", 1: "NOT CROSSING"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: 红=过街, 绿=不过街


# ========================
# LSTM模型定义（和训练时完全一致）
# ========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        return self.fc(x)


# ========================
# 关键点归一化（和训练时完全一致）
# ========================
def normalize_keypoints(frame_points):
    """
    输入: flat list [x0,y0,x1,y1,...] 长度=34
    输出: 归一化后的flat array，长度=34
    归一化方式: 以髋部中点为原点，肩宽为尺度
    """
    pts = np.array(frame_points).reshape(-1, 2)  # [17, 2]

    # 平移：以左右髋部中点为原点（COCO: 左髋=11, 右髋=12）
    if pts.shape[0] >= 13:
        center = (pts[11] + pts[12]) / 2.0
    else:
        center = np.mean(pts, axis=0)
    pts = pts - center

    # 缩放：以肩宽归一化（左肩=5, 右肩=6）
    if pts.shape[0] >= 7:
        shoulder_width = np.linalg.norm(pts[5] - pts[6])
        if shoulder_width > 1e-6:
            pts = pts / shoulder_width

    return pts.flatten().astype(np.float32)


# ========================
# 核心函数：视频 → 特征序列
# ========================
def video_to_sequence(video_path, pose_model):
    """
    逐帧处理视频，提取关键点序列
    返回: tensor shape=[1, MAX_FRAMES, INPUT_DIM], frames_list用于可视化
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    video_data = []   # 每帧的归一化关键点
    frames_list = []  # 原始帧用于可视化
    frame_results = []  # YOLO结果用于可视化
    detected_count = 0  # 成功检测到人的帧数（调试用）

    frame_idx = 0
    while len(video_data) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frames_list.append(frame.copy())

        # YOLOv8-Pose推理
        results = pose_model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        result = results[0]
        frame_results.append(result)

        # 提取第一个检测到的人的关键点
        if (result.keypoints is not None and
                len(result.keypoints.xy) > 0 and
                result.boxes is not None and
                len(result.boxes) > 0):

            # 取置信度最高的那个人（boxes已按conf排序）
            kps = result.keypoints.xy[0].cpu().numpy()  # [17, 2]
            flat = kps.flatten().tolist()
            norm = normalize_keypoints(flat)
            video_data.append(norm)
            detected_count += 1
        else:
            # 没检测到人 → 补零
            video_data.append(np.zeros(INPUT_DIM, dtype=np.float32))

        frame_idx += 1

    cap.release()
    print(f"检测统计: {detected_count}/{len(frames_list)} 帧成功检测到行人 ({detected_count/max(len(frames_list),1):.0%})")

    # 不足MAX_FRAMES则补零
    while len(video_data) < MAX_FRAMES:
        video_data.append(np.zeros(INPUT_DIM, dtype=np.float32))

    seq = np.array(video_data[:MAX_FRAMES], dtype=np.float32)
    tensor = torch.tensor(seq).unsqueeze(0)  # [1, 50, 34]
    return tensor, frames_list, frame_results


# ========================
# 可视化：把预测结果画到视频上
# ========================
def save_visualized_video(frames_list, frame_results, label, prob, output_path):
    """
    在每帧上画出：bbox、关键点骨架、预测标签和概率
    """
    if not frames_list:
        return

    h, w = frames_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 15, (w, h))

    color = COLORS[label]
    text = f"{LABELS[label]}: {prob:.1%}"

    for frame, result in zip(frames_list, frame_results):
        vis_frame = frame.copy()

        # 画检测框
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # 画关键点
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            for person_kps in result.keypoints.xy.cpu().numpy():
                for (x, y) in person_kps:
                    if x > 0 and y > 0:
                        cv2.circle(vis_frame, (int(x), int(y)), 3, (255, 255, 0), -1)

        # 画预测标签（左上角）
        cv2.rectangle(vis_frame, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(vis_frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        out.write(vis_frame)

    out.release()
    print(f"可视化视频已保存: {output_path}")


# ========================
# 主推理函数
# ========================
def predict(video_path, lstm_model_path, pose_model_path="yolov8n-pose.pt",
            save_video=True, save_json=False):
    """
    端到端推理
    返回: (label_str, prob_cross, prob_notcross)
    """
    print(f"\n{'='*50}")
    print(f"输入视频: {video_path}")
    print(f"{'='*50}")

    # 加载YOLO Pose模型
    print("加载 YOLOv8-Pose 模型...")
    pose_model = YOLO(pose_model_path)

    # 加载LSTM模型
    print("加载 LSTM 分类模型...")
    lstm = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    state = torch.load(lstm_model_path, map_location="cpu")
    lstm.load_state_dict(state)
    lstm.eval()

    # 提取特征序列
    print(f"提取关键点序列（最多{MAX_FRAMES}帧）...")
    seq_tensor, frames_list, frame_results = video_to_sequence(video_path, pose_model)
    print(f"实际处理帧数: {len(frames_list)}")

    # LSTM推理
    with torch.no_grad():
        logits = lstm(seq_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_label = int(np.argmax(probs))
    prob_cross = float(probs[0])
    prob_notcross = float(probs[1])

    print(f"\n预测结果: {LABELS[pred_label]}")
    print(f"  CROSSING    概率: {prob_cross:.1%}")
    print(f"  NOT CROSSING概率: {prob_notcross:.1%}")

    # 保存可视化视频
    if save_video and frames_list:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f"{base}_result.mp4"
        save_visualized_video(frames_list, frame_results,
                               pred_label, probs[pred_label], out_path)

    # 可选：保存JSON结果
    if save_json:
        result_dict = {
            "video": video_path,
            "prediction": LABELS[pred_label],
            "prob_crossing": prob_cross,
            "prob_notcrossing": prob_notcross,
            "frames_processed": len(frames_list)
        }
        json_out = os.path.splitext(video_path)[0] + "_result.json"
        with open(json_out, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"JSON结果已保存: {json_out}")

    return LABELS[pred_label], prob_cross, prob_notcross


# ========================
# 命令行入口
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="行人过街意图预测 — 端到端推理"
    )
    parser.add_argument("--video", required=True,
                        help="输入视频路径（.mp4）")
    parser.add_argument("--model", default="LSTM_seq_best.pth",
                        help="LSTM模型权重路径（默认: LSTM_seq_best.pth）")
    parser.add_argument("--pose", default="yolov8n-pose.pt",
                        help="YOLOv8-Pose模型路径（默认自动下载）")
    parser.add_argument("--no-video", action="store_true",
                        help="不保存可视化视频")
    parser.add_argument("--save-json", action="store_true",
                        help="保存JSON格式结果")
    args = parser.parse_args()

    label, p_cross, p_notcross = predict(
        video_path=args.video,
        lstm_model_path=args.model,
        pose_model_path=args.pose,
        save_video=not args.no_video,
        save_json=args.save_json
    )
