# Pedestrian Crossing Detection (YOLOv8 + DeepSORT + LSTM)

This project implements a **pedestrian crossing behavior recognition system**, combining **YOLOv8 object detection**, **DeepSORT tracking**, **YOLOv8-Pose keypoint extraction**, and an **LSTM classifier**.  
The pipeline detects pedestrians in surveillance videos, tracks their movement, extracts body keypoints, and predicts whether a pedestrian is **crossing the street** or **not crossing**.

*This project was part of my Bachelor thesis, demonstrating experience in **Computer Vision**, **Deep Learning**, and **Behavior Recognition**.*

---

##  Pipeline Overview

1. **YOLOv8 Object Detection**  
   - Trained YOLOv8 model to detect pedestrians.  
   - Exported weight file `best.pt`.  

2. **YOLOv8 + DeepSORT Tracking**  
   - Applied object detection + tracking on video datasets.  
   - Generated green-screen videos (pedestrian foreground only).  

3. **YOLOv8-Pose Keypoint Detection**  
   - Extracted human pose keypoints from videos.  
   - Saved as JSON for each video.  

4. **LSTM Sequence Classification**  
   - Input: keypoint sequences  
   - Output: pedestrian behavior classification (**cross / not cross**)  
   - Trained LSTM classifier → best checkpoint: `LSTM_seq_best.pth`.
## Pipeline Diagram

```mermaid
flowchart TD
    A["YOLOv8 Object Detection(trained in Colab,best.pt)"] --> B["YOLOv8 + DeepSORT Tracking"]
    B --> C["Green-Screen Pedestrian Videos"]
    C --> D["YOLOv8-Pose Keypoint Extraction(yolov8n-pose.pt)"]
    D --> E["Keypoints JSON Sequences"]
    E --> F["LSTM Classifier(LSTM_seq_best.pth)"]
    F --> G{Prediction}
    G -->|Class 0| H[Crossing]
    G -->|Class 1| I[Not Crossing]
```
---

##  Results

### YOLOv8 Object Detection
- Precision-Recall Curve 

  <img src="1-YOLOv8_Object_detection/docs/BoxPR_curve.png" width="500"/>
- F1-Confidence Curve

  <img src="1-YOLOv8_Object_detection/docs/BoxF1_curve.png" width="500"/>
- Confusion Matrix (YOLOv8 Detection)

  <img src="1-YOLOv8_Object_detection/docs/confusion_matrix.png" width="500"/>
- Training Curves  
  
  <img src="1-YOLOv8_Object_detection/docs/results.png" width="500"/> 

### LSTM Classification
- Confusion Matrix

  <img src="3-YOLOv8Pose+LSTM predict/figures/confusion_matrix.png" width="500"/>  
- Training & Validation Loss

  <img src="3-YOLOv8Pose+LSTM predict/figures/loss_curve.png" width="500"/>
- Classification Metrics

  <img src="3-YOLOv8Pose+LSTM predict/figures/metrics_bar.png" width="500"/>
### Example outputs:

- Green-screen with pose keypoints:  
  <img src="docs/example_pose.png" width="500"/>  

- Final behavior classification (crossing):  
  <img src="docs/example_cross.png" width="500"/>  
---

##  How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Dataset Preparation
Place your videos into:
```bash
2-Deepsort_Tracking/datasets/videos/cross/
2-Deepsort_Tracking/datasets/videos/notcross/
```
### 3. YOLOv8 + DeepSORT Tracking
The YOLOv8 object detection model was trained in Google Colab using Ultralytics' YOLOv8 framework.
```bash
python 2-Deepsort_Tracking/batch_track_and_cut.py
```
### 4. YOLOv8-Pose Keypoint Extraction
```bash
python 3-YOLOv8Pose+LSTM predict/batch_predict.py
```
### 5. Train LSTM
```bash
python 3-YOLOv8Pose+LSTM predict/1-LSTM train.py
```
### 6. Predict Single Video
```bash
python 3-YOLOv8Pose+LSTM predict/2-predict.py
```
---
## Notes

The original dataset (videos) is not included due to size limits.

Example results are provided.

YOLOv8 training was performed on Google Colab, using Ultralytics YOLOv8 framework.
