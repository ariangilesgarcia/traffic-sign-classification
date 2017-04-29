import os
import cv2
import csv
import glob
import time
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


    def test_image(self, image, return_image=False, output=None):
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

        if output:
            image = self.get_image_with_bb(image, detected_signs)
            cv2.imwrite(output, image)

        if return_image:
            image = self.get_image_with_bb(image, detected_signs)
            return image

        return detected_signs


    def test_folder(self, input_folder, output_folder):
        file_filter = '*.jpg'
        file_list = glob.glob(os.path.join(input_folder, file_filter))

        for image_path in file_list:
            image = cv2.imread(image_path)
            basename = os.path.basename(image_path)
            output = os.path.join(output_folder, basename)
            self.test_image(image, output)


    def test_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()
            image = self.test_image(frame, return_image=True)
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def test_webcam(self):
        pass


    def get_image_with_bb(self, image, detections):
        color = (140, 242, 89)

        for detection in detections:
            x1, y1, x2, y2 = detection[0]
            label = detection[1][1]

            # Draw bounding box
            image = cv2.rectangle(image, (x2,y2), (x1,y1), color, 15)

            # Font and text configuration
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1
            thickness = 2
            baseline = 0

            # Draw rectangle as background text
            text = cv2.getTextSize(label, font, scale, thickness)
            text_width, text_height = text[0]
            image = cv2.rectangle(image, (x1, y1), (x1+text_width, y1-30-text_height), color, 15)
            image = cv2.rectangle(image, (x1, y1), (x1+text_width, y1-30-text_height), color, -1)

            # Draw text
            image = cv2.putText(image, label, (x1, y1-20), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), thickness)

        return image


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
    image_path = 'images/test-021032.jpg'
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


    detections = detector.test_folder('images/', None)
    print(detections)
    """

    detector.test_video('/home/arian/videoplayback.mp4')
