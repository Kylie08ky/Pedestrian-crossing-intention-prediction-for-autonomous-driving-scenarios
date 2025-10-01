import torch
import torch.nn as nn
import numpy as np
import json

# ======================
# 模型定义
# ======================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

# ======================
# 归一化函数（保持和训练一致）
# ======================
def normalize_keypoints(frame_points):
    frame_points = np.array(frame_points).reshape(-1, 2)
    if frame_points.shape[0] >= 12:
        center = (frame_points[11] + frame_points[12]) / 2.0
    else:
        center = np.mean(frame_points, axis=0)
    frame_points = frame_points - center
    if frame_points.shape[0] >= 7:
        shoulder_width = np.linalg.norm(frame_points[5] - frame_points[6])
        if shoulder_width > 1e-6:
            frame_points = frame_points / shoulder_width
    return frame_points.flatten()

# ======================
# JSON 转 Tensor (YOLOv8-Pose 格式)
# ======================
def json_to_tensor(json_path, max_frames=50, keypoint_num=17):
    """
    YOLOv8-Pose JSON 格式：
    [
        { "frame_id": 0, "objects": [
              { "bbox": [...], "keypoints": [[x,y], [x,y], ...] }
        ] },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    video_data = []
    for i, frame in enumerate(results):
        if i >= max_frames:
            break
        objects = frame.get("objects", [])
        if len(objects) == 0:
            # 没有检测到人 → 补零
            video_data.append([0] * (keypoint_num * 2))
            continue

        # 取第一个人（或者面积最大的人）
        obj = objects[0]
        kps = obj["keypoints"]  # [[x,y], ...]
        frame_points = []
        for kp in kps:
            frame_points.extend(kp)  # [x,y]

        norm_frame = normalize_keypoints(frame_points)
        video_data.append(norm_frame)

    # 补齐长度
    while len(video_data) < max_frames:
        video_data.append([0] * (keypoint_num * 2))

    arr = np.array(video_data, dtype=np.float32)
    return torch.tensor([arr], dtype=torch.float32)

# ======================
# 加载模型
# ======================
input_dim = 34   # 17 keypoints * 2
hidden_dim = 64
output_dim = 2
model = LSTMClassifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("LSTM_seq_best.pth", map_location="cpu"))
model.eval()

# ======================
# 预测函数
# ======================
def predict_video(json_path):
    x = json_to_tensor(json_path)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).numpy()
        pred = np.argmax(probs)
    print("预测概率:", probs)
    return "cross (过街)" if pred == 0 else "notcross (未过街)"

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    test_json = r"D:\Git Repo\Bachelor_thesis\3.2-YOLOv8Pose\test_json\test.json"
    result = predict_video(test_json)
    print(f"{test_json} 的预测结果: {result}")
