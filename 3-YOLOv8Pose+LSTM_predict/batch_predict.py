import os
import json
from ultralytics import YOLO

# ==============================
# 1. 配置
# ==============================
video_root = "videos"      # 输入视频根目录（里面有 cross / notcross 文件夹）
output_root = "results_yolov8pose"  # 输出根目录
model_name = "yolov8n-pose.pt"      # 模型

# ==============================
# 2. 加载模型
# ==============================
model = YOLO(model_name)

# ==============================
# 3. 遍历 cross / notcross
# ==============================
for category in ["cross", "notcross"]:
    input_dir = os.path.join(video_root, category)
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith(".mp4"):
            continue
        video_path = os.path.join(input_dir, file)
        video_name = os.path.splitext(file)[0]

        # 判断是否已存在 JSON（跑完过了就跳过）
        json_output_path = os.path.join(output_dir, video_name, f"{video_name}.json")
        if os.path.exists(json_output_path):
            print(f" 跳过已完成: {video_name}")
            continue

        print(f"️ 处理 [{category}] {video_path} ...")

        # 运行 YOLOv8-Pose 检测（保存可视化视频）
        results = model.predict(
            source=video_path,
            save=True,
            project=output_dir,
            name=video_name,
            imgsz=640
        )

        # 汇总 JSON（一个视频对应一个 JSON）
        all_frames = []
        for i, result in enumerate(results):
            frame_data = []
            if result.boxes is not None and result.keypoints is not None:
                for box, keypoints in zip(result.boxes.xywh.cpu().numpy(),
                                          result.keypoints.xy.cpu().numpy()):
                    frame_data.append({
                        "bbox": box.tolist(),
                        "keypoints": keypoints.tolist()
                    })

            all_frames.append({
                "frame_id": i,
                "objects": frame_data
            })

        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, "w") as f:
            json.dump(all_frames, f)

print(" 所有未完成的视频已处理完毕")
print(f"结果在: {output_root}/cross 和 {output_root}/notcross")
