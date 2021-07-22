from audioop import avg
import conf as conf
import cv2
import imutils as imutils
import numpy as np
from cv2 import getStructuringElement, morphologyEx, createBackgroundSubtractorKNN



def demo():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while True:
        ret,frame = cap.read()
        if ret is True:

            fgmask = fgbg.apply(frame)
            dilate = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            cnts, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)  # 计算轮廓线的外框
                if cv2.contourArea(c) < 2000:  #计算轮廓面积
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
#

            cv2.imshow("frame",frame)
            c = cv2.waitKey(10)
            if c==27 :
                break
    cap.release()
    cv2.destroyAllWindows()


demo()
