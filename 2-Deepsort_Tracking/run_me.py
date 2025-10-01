import imutils
import cv2
import numpy as np
import os
from tqdm import tqdm
import os
import yaml
from easydict import EasyDict as edict
from tracker import vars
from AIDetector_pytorch import Detector


def image_cut(image, rectangles):
    # 创建一个与原图大小相同的绿色背景图像
    green_bg = np.zeros_like(image)
    green_bg[:, :, 1] = 255  # 将绿色通道设为255，使背景变为绿色

    for rect in rectangles:
        x1, y1, x2, y2 = rect
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        # 将矩形框内的图像复制到绿色背景上
        roi = image[y1:y2, x1:x2]
        green_bg[y1:y2, x1:x2] = roi

    # 返回抠图结果
    return green_bg


def get_filename_and_houzhui(full_path):
    path, file_full_name = os.path.split(full_path)
    file_name, 后缀名 = os.path.splitext(file_full_name)
    return path, file_name, 后缀名


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            # self.update(yaml.load(fo.read()))
            self.update(yaml.safe_load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)



videos_dir = 'try' # edit here
#video_name_list = os.listdir(videos_dir)
video_name_list = [f for f in os.listdir(videos_dir) if f.endswith('_out.mp4')]

for video_name in tqdm(video_name_list):
    vars.init()

    video_path = os.path.join(videos_dir, video_name)
    det = Detector()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    t = int(1000 / fps)

    video_writer_tracker_cut = None  # 新的视频写入对象
    path, file_name, 后缀名 = get_filename_and_houzhui(full_path=video_path)
    output_video_cut = f'results/try_cut/{file_name}_tracker_cut.mp4'  # 新的输出视频文件名
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer_tracker_cut = cv2.VideoWriter(output_video_cut, fourcc, fps, (frame_width, frame_height))

    video_writer_tracker = None  # 新的视频写入对象
    output_video_tracker = f'results/try_tracker/{file_name}_tracker.mp4'  # 新的输出视频文件名
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    frame_width, frame_height = 1137,640
    video_writer_tracker = cv2.VideoWriter(output_video_tracker, fourcc2, fps, (frame_width, frame_height))



    while True:
        _, im = cap.read()
        if im is None:
            break
        im0 = im.copy()
        result, boxes = det.feedCap(im)  # 得到检测结果
        # 定义多个矩形框的坐标
        image_cut_result = image_cut(im0, boxes)
        cv2.imshow('image_cut', image_cut_result)

        result = result['frame']
        result = imutils.resize(result, height=640)
        video_writer_tracker.write(result)  # 写入新的视频帧
        print("result.shape:", result.shape)
        video_writer_tracker_cut.write(image_cut_result)  # 写入新的视频帧

        cv2.imshow('demo', result)
        if cv2.waitKey(t) & 0xFF == ord('q'):
            break
    cap.release()
    if video_writer_tracker_cut is not None:
        video_writer_tracker_cut.release()

    if video_writer_tracker is not None:
        video_writer_tracker.release()

    cv2.destroyAllWindows()
