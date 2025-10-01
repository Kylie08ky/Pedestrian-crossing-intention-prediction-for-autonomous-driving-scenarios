from tracker import update_tracker
import cv2


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []


    def build_config(self):


        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1


        im, faces, face_bboxes = update_tracker(self, im)
        # print('update_tracker',update_tracker)
        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes
        # print('face_bboxes',face_bboxes)
        # print("retDict:", retDict)

        print("self.recorded:", self.recorded)
        print("self.currentCarID:", self.currentCarID)

        return retDict,face_bboxes



    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
