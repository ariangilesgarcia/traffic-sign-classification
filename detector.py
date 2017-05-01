import os
import sys
import cv2
import json
import glob
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras import backend as K
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_eval, yolo_head

from models.cnn_models import *


class Detector:
    def __init__(self,
                 model_path='yolo/yolo.h5',
                 anchors_path='yolo/yolo_anchors.txt',
                 classes_path='yolo/yolo_classes.txt',
                 yolo_thresh=0.24,
                 yolo_iou_thresh=0.5,
                 crop_percent=0.15,
                 classes_file='data/custom/classes.json',
                 weights_path='checkpoints/model_3_custom/model_3_custom-weights-18-1.00.hdf5'):

        # ##################### #
        # Initialize YOLO model #
        # ##################### #

        # Get tensorflow session
        self.sess = K.get_session()

        # Read YOLO classes names
        with open(classes_path) as f:
            self.class_names = f.readlines()
            self.class_names = [c.strip() for c in self.class_names]

        # Read anchors
        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1, 2)

        # Load keras model
        self.yolo_model = load_model(model_path)

        # Get input image size
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]

        # Generate output tensor targets for filtered bounding boxes.
        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))

        self.boxes, self.scores, self.classes_yolo = yolo_eval(self.yolo_outputs,
                                                               self.input_image_shape,
                                                               score_threshold=yolo_thresh,
                                                               iou_threshold=yolo_iou_thresh)

        self.crop_percent = crop_percent

        # ########################### #
        # Initialize classifier model #
        # ########################### #

        # Load classes names
        with open(classes_file, 'r') as classes_data:
            self.classes = json.load(classes_data)

        # Load model and weights
        self.model = model_3(weights_path)


    def detect_traffic_sign(self, image):
        resized_image = cv2.resize(image, self.model_image_size)
        image_data = np.array(resized_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes_yolo],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })

        detections = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            # item = [predicted_class, score, left, top, right, bottom]
            item = [left, top, right, bottom]
            detections.append(item)

        return detections


    def classify_traffic_sign(self, image):
        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Predict class for image
        predicted_class = self.model.predict_classes(image, verbose=False)

        return [predicted_class[0], self.classes[str(predicted_class[0])]['name']]


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

            crop = image[y1:y2, x1:x2]

            # Classify detected sign
            predicted_class = self.classify_traffic_sign(crop)

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
