from cv2 import imread, CascadeClassifier, rectangle, imshow, waitKey, destroyAllWindows, VideoCapture
import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import numpy as np
import argparse
import imutils
import time
import random



class Analyse_image:

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
        self.gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')

        self.detector = MTCNN()

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.gender_list = ['Male', 'Female']
    
    
    def _draw_on_image(self, frame, result_list, reading_file = False):
        if reading_file:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        found_faces = len(result_list)
        print("Found {} faces".format(found_faces))

        pyplot.imshow(frame)
        ax = pyplot.gca()

        for result in result_list:

            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height

            rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            face_img = frame[y1:y2, x1:x2].copy()
            face_img = imutils.resize(face_img, 240)
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            
            
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)


            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (x1, y1 - 4), self.font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        info_text = "No. of Person: %s" % (found_faces)
        cv2.putText(frame, info_text, (10, 25), self.font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


        if found_faces:
            cv2.imwrite(f"outputs/output_image_{time.time()}.jpg", frame)

    

    def from_frame(self, frame):
        faces_result_list = self.detector.detect_faces(frame)
        self._draw_on_image(frame, faces_result_list, reading_file = False)
    

    
    def from_file(self, filename):
        frame = pyplot.imread(filename)
        faces_result_list = self.detector.detect_faces(frame)
        self._draw_on_image(frame, faces_result_list, reading_file = True)
    


###############################################################################################################################

if __name__ == '__main__':

    img_file = str(input("Enter relative path to the saved image: "))
    analyse = Analyse_image()
    analyse.from_file(img_file)
