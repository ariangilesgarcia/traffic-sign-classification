import os
import sys
import cv2
import json
import glob
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from models.cnn_models import *
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet


class Detector:
    def __init__(self,
                 cfg_path='yolo/yolo-obj.cfg',
                 yolo_input_size=416,
                 yolo_weights_path='yolo/yolo-obj.weights',
                 yolo_threshold=0.24,
                 classifier_thresh=0.1,
                 crop_percent=0.25,
                 classes_file='data/custom/classes.json',
                 weights_path='checkpoints/model_4_custom/model_4_custom-weights-10-1.00.hdf5'):

        # ##################### #
        # Initialize YOLO model #
        # ##################### #

        options = {"model": cfg_path, "config": cfg_path, "load": yolo_weights_path, "threshold": yolo_threshold, "gpu": 1}
        self.yolo = TFNet(options)

        self.yolo_input_size = (yolo_input_size, yolo_input_size)

        # ########################### #
        # Initialize classifier model #
        # ########################### #

        # Load classes names
        with open(classes_file, 'r') as classes_data:
            self.classes = json.load(classes_data)

        # Save threshold
        self.classifier_thresh = classifier_thresh

        # Load model and weights
        self.model = model_4(weights_path)

        # ##################### #
        # Initialize variables  #
        # ##################### #

        self.crop_percent = crop_percent


    def detect_traffic_sign(self, image):
        resized_image = cv2.resize(image, self.yolo_input_size)
        image_data = np.array(resized_image, dtype='float32')

        ts_detections = self.yolo.return_predict(image)

        detections = []

        for detection in ts_detections:
            x1 = detection['topleft']['x']
            y1 = detection['topleft']['y']
            x2 = detection['bottomright']['x']
            y2 = detection['bottomright']['y']

            item = [x1, y1, x2, y2]
            detections.append(item)

        return detections


    def classify_traffic_sign(self, image):
        # Resize image
        image = image[...,::-1]
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Predict class for image
        predictions = self.model.predict(image, verbose=False)[0]
        class_id = np.argmax(predictions)

        if predictions[class_id] > self.classifier_thresh:
            return [class_id, self.classes[str(class_id)]['name']]
        else:
            return None


    def test_image(self, image, return_image=False, show_image=False, output=None):
        # Detect location of traffic sign
        detections = self.detect_traffic_sign(image)
        img_height, img_width, _ = image.shape

        detected_signs = []

        # For each detected sign
        for detection in detections:
            x1, y1, x2, y2 = detection

            # Calculate width and height of crop
            width = x2 - x1
            height = y2 - y1

            # Calculate center point of crop
            center_x = x1 + int(width/2)
            center_y = y1 + int(height/2)

            # Calculate the size of the square side
            size = max(width, height)
            size = int(size/2 + (1 + self.crop_percent))

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

            crop = image[y1:y2, x1:x2]

            # Classify detected sign
            predicted_class = self.classify_traffic_sign(crop)

            if predicted_class:
                detected_sign = [detection, predicted_class]
                detected_signs.append(detected_sign)

        if output:
            image = self.get_image_with_bb(image, detected_signs)
            cv2.imwrite(output, image)

        if show_image:
            image = self.get_image_with_bb(image, detected_signs)
            cv2.imshow('results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if return_image:
            image = self.get_image_with_bb(image, detected_signs)
            return image

        return detected_signs


    def test_folder(self, input_folder, output_folder):
        file_filter = '*.jpg'
        file_list = glob.glob(os.path.join(input_folder, file_filter))

        for image_path in tqdm(file_list):
            image = cv2.imread(image_path)
            basename = os.path.basename(image_path)
            output = os.path.join(output_folder, basename[:-4] + '_test.jpg')
            self.test_image(image, output=output)


    def test_video(self, video_path, output=None):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))

        for i in tqdm(range(frames)):
            ret, frame = cap.read()
            image = self.test_image(frame, return_image=True)

            if output:
                out.write(image)
            else:
                cv2.imshow('Output', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        if output:
            out.release()

        cap.release()


    def test_webcam(self, feed=0, output=None):
        if output:
            cap = cv2.VideoCapture(0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

        while True:
            # Get feed frame
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            # Test image
            image = self.test_image(frame, return_image=True)
            cv2.imshow('frame', image)

            if output:
                out.write(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if output:
            out.release()

        cv2.destroyAllWindows()


    def get_image_with_bb(self, image, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection[0]
            label_id = detection[1][0]
            label = detection[1][1]

            color = self.classes[str(label_id)]['color']

            # Box thickness
            bbox_thickness = 10

            # Draw bounding box
            image = cv2.rectangle(image, (x2,y2), (x1,y1), color, bbox_thickness)

            # Font and text configuration
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1
            thickness = 2
            baseline = 0

            # Image shape
            img_height, img_width, _ = image.shape

            # Label text variables
            text = cv2.getTextSize(label, font, scale, thickness)
            text_width, text_height = text[0]

            x_text = x1
            y_text = y1 - 20

            if x_text + text_width > img_width:
                x_text = img_width - text_width

            # Draw rectangle as background text
            image = cv2.rectangle(image, (x_text, y1), (x_text+text_width, y1-30-text_height), color, bbox_thickness)
            image = cv2.rectangle(image, (x_text, y1), (x_text+text_width, y1-30-text_height), color, -1)

            # Draw text
            image = cv2.putText(image, label, (x_text, y_text), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), thickness)

        return image


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Traffic Sign Detector")
    test_sp = parser.add_subparsers(help='Test image/folder/video/webcam for traffic signs', dest='command')

    image_parser = test_sp.add_parser('image', help='Test image for traffic signs')
    image_parser.add_argument('-i', '--input-image', required=True, help='Path of the image to test')
    image_parser.add_argument('-o', '--output-image', help='Path where the output image is saved')

    folder_parser = test_sp.add_parser('folder', help='Test all images in a folder for traffic signs')
    folder_parser.add_argument('-i', '--input-folder', required=True, help='Path of the folder containing images to test')
    folder_parser.add_argument('-o', '--output-folder', help='Path where the tested images are saved')

    video_parser = test_sp.add_parser('video', help='Test video for traffic signs')
    video_parser.add_argument('-i', '--input-video', required=True, help='Path of the video to test')
    video_parser.add_argument('-o', '--output-video', help='Path where the output video is saved')

    webcam_parser = test_sp.add_parser('webcam', help='Test webcam feed for traffic signs')
    webcam_parser.add_argument('-i', '--input-feed', help='ID of the webcam to test')
    webcam_parser.add_argument('-o', '--output-video', help='Path where the feed video is saved')

    args = parser.parse_args()

    # Create Detector object
    detector = Detector()

    # Parse arguments
    if args.command == 'image':
        image_path = args.input_image
        output_path = args.output_image

        # Load image
        image = cv2.imread(image_path)

        # Test image
        if output_path:
            detector.test_image(image=image, output=output_path)
        else:
            detector.test_image(image=image, show_image=True)

    elif args.command == 'folder':
        # Check if argument is a folder
        if not os.path.isdir(args.input_folder):
            parser.print_help()
            sys.exit()

        if args.output_folder and not os.path.isdir(args.output_folder):
            parser.print_help()
            sys.exit()

        input_folder = args.input_folder
        output_folder = args.output_folder if args.output_folder else input_folder

        # Test folder
        detector.test_folder(input_folder=input_folder, output_folder=output_folder)

    elif args.command == 'video':
        video_path = args.input_video
        output_path = args.output_video

        # Test video
        detector.test_video(video_path=video_path, output=output_path)

    elif args.command == 'webcam':
        feed_id = args.input_feed
        output_path = args.output_video

        # Test webcam
        if feed_id:
            detector.test_webcam(feed=feed_id, output=output_path)
        else:
            detector.test_webcam(output=output_path)
