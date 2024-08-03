import json
from os import makedirs
from pathlib import Path
from typing import Optional, List, Tuple, Any

from PIL import Image
import numpy as np
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh
# from deepface import DeepFace
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import scipy.io as scio

from facevision.pose.sixdrepnet import SixDRepNet
# from fiqa.shufflenetv2 import ShuffleNetV2
# from fiqa.occlusion_detector import OcclusionDetector


class FaceAnalyzer:
    def __init__(self, image_path: Optional[str]):
        """
        Initializes the FIQA class with the given image path.
        Args:
            image_path (str, optional): The path to the image to be assessed. If not provided, certain attributes
            and methods related to the image will be set to None.
        """
        self.root_dir = Path(__file__).parent

        self.__image_path = image_path
        self.image = self.__image
        self.bbox = self.__bbox
        self.face = self.__face
        self.head = self.__head
        self.landmark = self.__landmark

        self.face_detector = FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
        self.pose_estimator = SixDRepNet(gpu_id=-1, dict_path='')

    def set_image(self, img_path: str):
        """
        Sets the image path and updates the internal image and landmark properties.
        Args:
            img_path (str): The path to the new image.
        """
        self.__image_path = img_path
        self.image = self.__image
        self.bbox = self.__bbox
        self.face = self.__face
        self.head = self.__head
        self.landmark = self.__landmark

    @property
    def __image(self) -> np.ndarray:
        """
        Loads and processes the image from the given image path.
        The image is read using OpenCV, flipped for selfie view, and converted to RGB.
        Returns:
            np.ndarray: The processed image.
        """
        image = cv2.imread(self.__image_path)
        # image = cv2.flip(image, 1)  # flipped for selfie view
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @property
    def __bbox(self):
        """
        Calculates the bounding box for the detected face landmarks.
        Returns:
            tuple: The bounding box coordinates (x1, y1, x2, y2).
        """
        detections = self.face_detector.process(self.image).detections

        if detections is None:
            return None

        bbox = detections[0].location_data.relative_bounding_box
        return bbox

    @property
    def __face(self):
        """
        Extracts the face region from the image based on the bounding box.
        Returns:
            np.ndarray: The cropped face region.
        """
        img_h, img_w, c = self.image.shape
        x1, y1 = int(self.bbox.xmin * img_w), int(self.bbox.ymin * img_h)
        x2, y2 = x1 + int(self.bbox.width * img_w), y1 + int(self.bbox.height * img_h)
        face = self.image[y1:y2, x1:x2]
        return face

    @property
    def __head(self):
        """
        Detects the person head and crops the image around the detected face.
        Returns:
            np.ndarray: The cropped image region containing the person.
        """
        h, w, c = self.image.shape
        x = int(0.5 * self.bbox.ymin * h)
        y = int(0.5 * self.bbox.xmin * w)
        l = int(max(self.bbox.width, self.bbox.height) * 1.4 * max(h, w))
        head = self.image[x: x + l, y: y + l]
        return head

    @property
    def __landmark(self) -> np.ndarray:
        """
        Detects and returns the face landmarks from the image.
        Returns:
            np.ndarray: An array of face landmark coordinates.
        Raises:
            excp.FaceNotDetected: If no face is detected in the image.
            excp.TooManyFaces: If more than one face is detected in the image.
        """
        self.image.flags.writeable = False
        results = self.face_mesh.process(self.image)

        if results.multi_face_landmarks is None:
            return None

        h, w, c = self.image.shape
        landmark = np.array([(lm.x, lm.y, lm.z) for lm in list(results.multi_face_landmarks[0].landmark)])
        landmark[:, 0] = np.round(landmark[:, 0] * w)
        landmark[:, 1] = np.round(landmark[:, 1] * h)
        return landmark.round().astype(np.int16)

    @property
    def check_size_standards(self) -> dict:
        """
        Assesses the ratio of the image that must be 3x4 and the location of the face in the image.
        Returns:
            dict: A dictionary with size information including ratio3X4, face location (x, y, z).
        """
        h, w, c = self.image.shape
        ratio_3x4 = h / w

        size_info = {'ratio3x4': 1.32 <= ratio_3x4 <= 1.34}

        left_eye, right_eye = self.__landmark[[468, 473]]
        middle_point = (right_eye + left_eye) // 2

        if middle_point[0] < 0.45 * w:
            face_loc_x = 'left'
        elif 0.55 * w < middle_point[0]:
            face_loc_x = 'right'
        else:
            face_loc_x = 'center'

        if 0.5 * h < middle_point[1]:
            face_loc_y = 'down'
        elif middle_point[1] < 0.3 * h:
            face_loc_y = 'up'
        else:
            face_loc_y = 'center'

        if self.face.shape[0] < 0.5 * w and self.face.shape[1] < 0.6 * h:
            face_loc_z = 'far'
        elif 0.75 * w < self.face.shape[0] and 0.9 * h < self.face.shape[1]:
            face_loc_z = 'close'
        else:
            face_loc_z = 'center'

        size_info.update({'face_loc_x': face_loc_x,
                          'face_loc_y': face_loc_y,
                          'face_loc_z': face_loc_z})
        return size_info

    @property
    def sharpness(self) -> float:
        """
        Calculates the sharpness of the face region using the Laplacian method.
        Returns:
            float: The sharpness value.
        """
        sharpness = cv2.Laplacian(self.face, ddepth=cv2.CV_64F, ksize=3).var()
        return str(sharpness)

    @property
    def pose(self) -> dict:
        """
        Estimates the head pose of the person in the image.
        Returns:
            dict: A dictionary containing the pitch, yaw, and roll angles.
        """
        pitch, yaw, roll = self.pose_estimator.predict(self.head)
        return {'pitch': str(pitch[0]),
                'yaw': str(yaw[0]),
                'roll': str(roll[0])}
