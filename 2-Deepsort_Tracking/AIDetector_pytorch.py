# AIDetector_pytorch.py — YOLOv8 detector (CPU-friendly, only "person")
# 直接覆盖原文件即可

import torch
import numpy as np
from ultralytics import YOLO
from utils.BaseDetector import baseDet
import cv2

class Detector(baseDet):
    def __init__(self, weights='weights/best.pt', img_size=640, conf_thres=0.4, iou_thres=0.5):
        super(Detector, self).__init__()
        self.weights = weights
        self.img_size = img_size
        # 保持原项目接口字段
        self.threshold = conf_thres
        self.iou_thres = iou_thres
        self.init_model()
        self.build_config()

    def init_model(self):
        # 强制 CPU，如果有 GPU 也可自动检测
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 加载 YOLOv8 模型
        self.model = YOLO(self.weights)
        # 类别名：list 或 dict
        self.names = self.model.names

    def preprocess(self, img):
        # YOLOv8 内部有 resize/letterbox，这里只返回原图和副本
        return img.copy(), img

    def detect(self, im):
        """
        返回 (im, pred_boxes)
        pred_boxes: [(x1, y1, x2, y2, label, conf), ...]
        只保留 'person'
        """
        im0, _ = self.preprocess(im)

        res = self.model.predict(
            source=im0,
            imgsz=self.img_size,
            conf=self.threshold,
            iou=self.iou_thres,
            device='cpu' if self.device.type == 'cpu' else 0,
            verbose=False
        )

        pred_boxes = []
        if len(res) > 0:
            r = res[0]
            if r.boxes is not None and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss):
                    c = int(c)
                    name = self.names[c] if isinstance(self.names, (list, tuple)) else self.names.get(c, str(c))
                    if name != 'person':
                        continue
                    pred_boxes.append((int(x1), int(y1), int(x2), int(y2), name, float(cf)))

        return im, pred_boxes
