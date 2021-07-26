import sys
import numpy as np
from PIL import Image
import os
import cv2


# 人脸数据路径
path = 'D:/test/face'

recognizer = cv2.face.LBPHFaceRecognizer_create()
#detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
# 使用opencv预置人脸检测的模型
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

def getImage():

    name = input('输入你的名字（每个字的首字母）：')    #名字
    id = input('输入你的id（数字）：')       #标签
    cap = cv2.VideoCapture(0)    #"E:/demo.mp4"
    # 摄像头打开失败
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()
    count = 0

    while True:
        ret, image = cap.read() #读取视频的一帧
        # 视频读取失败
        if not ret:
            print('Cannot read video file')
            sys.exit()
        (h, w) = image.shape[:2]
        # 将原图画转换为灰阶图像
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #人脸检测
        face = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(24, 24))
        for (fx, fy, fw, fh) in face:
            cv2.putText(image, name, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 1)
            count += 1
            if count <= 100:
                faceImageName = "D:/test/face/#" + name + '#.' + str(id)+"."+str(count) + '.jpg'#str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(faceImageName, gray[fy: fy + fh, fx: fx + fw])      #保存灰阶图像进行训练
            else:
                cv2.putText(image, "The face information is collected", (fx-fw, fy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #  显示
        cv2.imshow('人脸采集', image)
        k = cv2.waitKey(100)& 0xff
        # 27对应Esc，当点击该键时退出
        if k == 27:
            break
        elif k == 32:
            while cv2.waitKey(0) != 32:
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  #读取照片素材所有文件路径
    faceSamples = []
    imageNums = []
    for imagePath in imagePaths:
        print(imagePath)
        print(type(imagePath))
        PIL_img = Image.open(imagePath).convert('L')   # 转成灰度图
        img_numpy = np.array(PIL_img, 'uint8')
        imageNum = int(os.path.split(imagePath)[-1].split(".")[1])
        print(imageNum)
        faces = face_classifier.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            imageNums.append(imageNum)
    return faceSamples, imageNums

print('采集数据')
getImage()
print('训练需要一定时间，请耐心等待....')
faces, imageNums = getImagesAndLabels(path)
recognizer.train(faces, np.array(imageNums))   #训练
recognizer.write(r'D:/test/face_trainer/trainer.yml')
print("{0} faces trained. Exiting Program".format(len(imageNums)))

