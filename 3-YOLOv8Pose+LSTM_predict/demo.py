import os
import json
from ultralytics import YOLO

# ==============================
# 1. 配置
# ==============================
video_path = "test.mp4"   # 输入视频路径
output_dir = "results_test"  # 输出目录
model_name = "yolov8n-pose.pt"  # 模型，可换 yolov8m/l-pose.pt

# ==============================
# 2. 加载模型
# ==============================
model = YOLO(model_name)

# ==============================
# 3. 推理（会生成带关键点的视频）
# ==============================
results = model.predict(
    source=video_path,
    save=True,           # 保存带关键点的视频
    project=output_dir,  # 输出主目录
    name="predict"       # 子目录名
)

# ==============================
# 4. 汇总 JSON（整合所有帧）
# ==============================
video_name = os.path.splitext(os.path.basename(video_path))[0]
json_output_path = os.path.join(output_dir, "predict", f"{video_name}.json")

all_frames = []
for i, result in enumerate(results):
    frame_data = []
    if result.boxes is not None and result.keypoints is not None:
        for box, keypoints in zip(result.boxes.xywh.cpu().numpy(), result.keypoints.xy.cpu().numpy()):
            frame_data.append({
                "bbox": box.tolist(),        # [x, y, w, h]
                "keypoints": keypoints.tolist()  # [[x1, y1], [x2, y2], ...]
            })

    all_frames.append({
        "frame_id": i,
        "objects": frame_data
    })

# 写入一个 JSON 文件
with open(json_output_path, "w") as f:
    json.dump(all_frames, f)

print(" 检测完成")
print(f"可视化视频: {os.path.join(output_dir, 'predict', os.path.basename(video_path))}")
print(f"汇总 JSON: {json_output_path}")
