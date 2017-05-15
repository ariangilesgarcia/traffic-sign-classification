import os
import cv2
import glob
import uuid
import argparse
import numpy as np
from tqdm import tqdm
from darkflow.net.build import TFNet


cfg_path='yolo/yolo-obj.cfg'
yolo_input_size=416
yolo_weights_path='yolo/yolo-obj.weights'
yolo_threshold=0.1
crop_percent=0.5


def yolo_to_crops(video_path, output_folder, crop_folder):
    options = {"model": cfg_path, "config": cfg_path, "load": yolo_weights_path, "threshold": yolo_threshold, "gpu": 1}
    yolo = TFNet(options)

    cap = cv2.VideoCapture(video_path)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frames)):
        name = str(uuid.uuid4())

        ret, frame = cap.read()
        detections =  yolo.return_predict(frame)

        if detections:
            image_path = os.path.join(output_folder, name + '.jpg')
            cv2.imwrite(image_path, frame)

            label_path = os.path.join(output_folder, name + '.txt')
            label_file = open(label_path, 'w')

            img_height, img_width, _ = frame.shape

            for detection in detections:
                x1 = detection['topleft']['x']
                y1 = detection['topleft']['y']
                x2 = detection['bottomright']['x']
                y2 = detection['bottomright']['y']

                item = [0, x1, y1, x2, y2]

                # Write label
                label = ' '.join(map(str, item))
                label_file.write(label)
                label_file.write('\n')

                # Save crop
                # Calculate width and height of crop
                width = x2 - x1
                height = y2 - y1

                # Calculate center point of crop
                center_x = x1 + int(width/2)
                center_y = y1 + int(height/2)

                # Calculate the size of the square side
                size = max(width, height)
                size = int(size/2 + (1 + crop_percent))

                # Calculate new coords
                x1 = center_x - size
                x2 = center_x + size
                y1 = center_y - size
                y2 = center_y + size

                if x1 < 0:
                    x1 = 0
                if x2 > img_width:
                    x2 = img_width
                if y1 < 0:
                    y1 = 0
                if y2 > img_height:
                    y2 = img_height

                detection = x1, y1, x2, y2

                crop = frame[y1:y2, x1:x2]
                crop_name = str(uuid.uuid4()) + '.jpg'
                crop_path = os.path.join(crop_folder, crop_name)

                cv2.imwrite(crop_path, crop)

            label_file.close()


    cap.release()


if __name__ == '__main__':
    video_path = '/home/arian/test.mp4'
    output_folder = '/home/arian/Documents/proyecto-integrador/data/videos/test/out'
    crop_folder = '/home/arian/Documents/proyecto-integrador/data/videos/test/crops'

    yolo_to_crops(video_path, output_folder, crop_folder)
