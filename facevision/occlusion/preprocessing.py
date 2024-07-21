from os import listdir, makedirs
from pathlib import Path
from shutil import move

import cv2
from mediapipe.python.solutions.face_detection import FaceDetection

DETECTOR = FaceDetection()


def rename_images(dataset_dir):
    for img_folder in listdir(dataset_dir):
        counter = 1
        for image_path in Path(dataset_dir, img_folder).glob('*.jpg'):
            try:
                if image_path.name.split('_')[0] == img_folder:
                    continue
                new_name = f"{image_path.parent.name}_{counter:04}.jpg"
                new_img_path = Path(dataset_dir, img_folder, new_name)
                counter += 1
                move(image_path, new_img_path)
            except FileNotFoundError:
                print(image_path)
                continue


def split_detected_faces(db_dir: str):
    for cls in listdir(db_dir):
        makedirs(Path(db_dir, 'undetected', cls), exist_ok=True)
        makedirs(Path(db_dir, 'faces', cls), exist_ok=True)
        for img_path in Path(db_dir, cls).glob('*.jpg'):
            try:
                bgr_img = cv2.imread(str(img_path))

                if bgr_img is None:
                    results = None
                else:
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    results = DETECTOR.process(img).detections

                    img_h, img_w, c = img.shape
                    bbox = results[0].location_data.relative_bounding_box

                    y = int(bbox.ymin * img_h) if bbox.ymin > 0 else 0
                    x = int(bbox.xmin * img_w) if bbox.xmin > 0 else 0
                    face_w, face_h = int(bbox.width * img_w), int(bbox.height * img_h)
                    face = bgr_img[y: y + face_h, x: x + face_w]
                    cv2.imwrite(str(Path(db_dir, 'faces', cls, img_path.name)), face)

                if results is None or len(results) == 0:
                    img_path_to_move = Path(db_dir, 'undetected', cls, img_path.name)

                    if not img_path_to_move.is_file():
                        move(img_path, img_path_to_move)

            except FileNotFoundError:
                print(img_path)
                continue


# split_detected_faces(r"C:\Users\j.mokhtari\Downloads\datasets\glasses-and-coverings\occluded_faces")
# rename_images(r"C:\Users\j.mokhtari\Downloads\datasets\glasses-and-coverings\occlusion_dataset")
