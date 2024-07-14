from os import listdir, makedirs
from pathlib import Path
from shutil import move

import cv2
from mediapipe.python.solutions.face_detection import FaceDetection

DETECTOR = FaceDetection()


def split_detected_faces(db_dir: str):
    for cls in listdir(db_dir):
        makedirs(Path(db_dir, 'undetected', cls), exist_ok=True)
        # makedirs(Path(db_dir, 'faces', cls), exist_ok=True)
        for img_path in Path(db_dir, cls).glob('*.jpg'):
            try:
                bgr_img = cv2.imread(str(img_path))

                if bgr_img is None:
                    results = None
                else:
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    results = DETECTOR.process(img).detections

                if results is None or len(results) == 0:
                    img_path_to_move = Path(db_dir, 'undetected', cls, img_path.name)

                    if not img_path_to_move.is_file():
                        move(img_path, img_path_to_move)

            except FileNotFoundError:
                print(img_path)
                continue

            # img_h, img_w, c = img.shape
            # bbox = results[0].location_data.relative_bounding_box
            #
            # y = int(abs(bbox.ymin * img_h))
            # x = int(abs(bbox.xmin * img_w))
            # face_w, face_h = int(bbox.width * img_w), int(bbox.height * img_h)
            # face = img[y: y + face_h, x: x + face_w]
            # cv2.imwrite(str(Path(db_dir, 'faces', cls, img_path.name)), face)


split_detected_faces(r"C:\Users\j.mokhtari\Downloads\datasets\glasses-and-coverings\occlusion_dataset")
