import os
import cv2
import csv
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from keras import backend as K
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_eval, yolo_head

from models.cnn_models import *


class Detector:
    def __init__(self,
                 model_path='yolo/yolo.h5',
                 anchors_path='yolo/yolo_anchors.txt',
                 classes_path='yolo/yolo_classes.txt',
                 yolo_thresh=0.3,
                 yolo_iou_thresh=0.5,
                 classes_file='data/custom/classes.csv',
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

        # ########################### #
        # Initialize classifier model #
        # ########################### #

        # Load classes names
        with open(classes_file, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            self.classes = {rows[0]:rows[1] for rows in reader}

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

        return [predicted_class[0], self.classes[str(predicted_class[0])]]


    def test_image(self, image):
        # Detect location of traffic sign
        detections = self.detect_traffic_sign(image)

        detected_signs = []

        # For each detected sign
        for detection in detections:
            x1, y1, x2, y2 = detection
            crop = image[y1:y2, x1:x2]

            # Classify detected sign
            predicted_class = self.classify_traffic_sign(crop)

            detected_sign = [detection, predicted_class]
            detected_signs.append(detected_sign)

        return detected_signs


    def test_folder(self, input_folder, output_folder):
        pass


    def test_video(self, video_path):
        pass


    def test_webcam(self):
        pass


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='label-to-crops')

    parser.add_argument('input_folder', help='path for the input folder')
    parser.add_argument('output_folder', help='path for the output folder')

    args = parser.parse_args()
    """

    # Create Detector object
    detector = Detector()

    # Read image for detection
    image_path = '/home/arian/test.jpg'
    image = cv2.imread(image_path)

    # Show image for testing purposes
    """
    cv2.imshow('Image for detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Convert image to RGB
    #image = image[...,::-1]

    # Call Detector.detect_traffic_sign() function
    detections = detector.detect_traffic_sign(image)

    # Crop image
    x1, y1, x2, y2 = detections[0]
    crop = image[y1:y2, x1:x2]
    predicted_class = detector.classify_traffic_sign(crop)

    print('ID: {}\t LABEL: {}'.format(predicted_class[0], predicted_class[1]))
    """

    detections = detector.test_image(image)
    print(detections)
