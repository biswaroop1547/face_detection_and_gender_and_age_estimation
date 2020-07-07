from cv2 import imread, CascadeClassifier, rectangle, imshow, waitKey, destroyAllWindows, VideoCapture
import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import random


def draw_image_with_boxes(data, result_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    flag = True

    w_increase_percentage = 2
    h_increase_percentage = 2

    # if len(result_list) == 0:
    #     result_list = classifier.detectMultiScale(data, 1.05, 15)
    #     flag = True

    found_faces = len(result_list)
    print("Found {} faces".format(found_faces))


    for result in result_list:
        # if flag:
        #    x, y, width, height = result 
        # else:
        x, y, width, height = result['box']

        x2, y2 = x + width + int((w_increase_percentage/100)*width), y + height + int((h_increase_percentage/100)*height)
        x, y = x - int((w_increase_percentage/100)*width), y - int((h_increase_percentage/100)*height)
        rectangle(data, (x, y), (x2, y2), (0, 0, 255), 1)
        
        face_img = data[y:y+height, height:height+width].copy()
        face_img = imutils.resize(face_img, 240)
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)


        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)

        overlay_text = "%s %s" % (gender, age)
        cv2.putText(data, overlay_text, (x, y), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # if found_faces:
            # cv2.imwrite(f"output_image_{time.time()}.jpg", data)

def load_models(): 
    age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
    return(age_net, gender_net)

def video_detector(age_net, gender_net, from_webcam):

    while True:

        if from_webcam:
            frame = vs.read()
        
        else:
            ret, frame = vs.read()
        
            frame_width, frame_height = (
                int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            frame = imutils.resize(frame, width = frame_width-150, height = frame_height-150)

        # print(frame.shape)
        faces = detector.detect_faces(frame)
        draw_image_with_boxes(frame, faces)

        cv2.imshow('Frame', frame)

        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if from_webcam:
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()

################################################################ 
################################################################

if __name__ == "__main__":

    source = str(input("Enter relative path of the video file or press enter to use webcam: "))
    # frame_width = int(input("Enter frame width: "))
    if source == "":
        vs = VideoStream(src=0).start()
        from_webcam = True
    else:
        vs = VideoCapture(source)
        from_webcam = False
    print("[INFO] starting video stream...")
    time.sleep(1.0)
    fps = FPS().start()

    detector = MTCNN()
    classifier = CascadeClassifier('models/haarcascade_frontalface_default.xml')
    age_net, gender_net = load_models()

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    video_detector(age_net, gender_net, from_webcam)