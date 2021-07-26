import os
import sys
import uuid
import cv2
from cv2 import getStructuringElement

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:/test/face_trainer/trainer.yml')

def imageClassification(video_path, names):
    # 判断是否有入侵行为，如果有则生成视频
    invade = False
    isHaveMovingObject = False  # 检测是否有移动物体
    # 模型路径 需要下载模型文件
    model_bin = "D:/ssd/MobileNetSSD_deploy.caffemodel"
    config_text = "D:/ssd/MobileNetSSD_deploy.prototxt"

    # 类别信息
    objName = ["background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"]

    # 加载模型
    net = cv2.dnn.readNetFromCaffe(config_text, model_bin)
    #使用opencv预置人脸检测的模型
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(video_path)#"D:/example video.avi"
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    # 视频打开失败
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()

    vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #宽度
    vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #高度
    fps = cap.get(cv2.CAP_PROP_FPS)         #帧率

    fileName = "D:/test/video" + str(uuid.uuid4()) + ".mp4"     #视频保存文件名
    out = cv2.VideoWriter(fileName, cv2.CAP_ANY, int(cap.get(cv2.CAP_PROP_FOURCC)), fps, (int(vw), int(vh)), True)  #保存视频
    while True:
        ret, image = cap.read() #读取视频的一帧

        # 视频读取失败
        if not ret:
            print('Cannot read video file')
            sys.exit()

        (h, w) = image.shape[:2]
        # 将原图画转换为灰阶图像
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        fgmask = fgbg.apply(image)
        dilate = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cnts, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for c in cnts:
            if isHaveMovingObject:
                break
            (x, y, w, h) = cv2.boundingRect(c)  # 计算轮廓线的外框
            if cv2.contourArea(c) < 2000:  # 计算轮廓面积
                continue
            elif w<30 or h<30:
                continue
            isHaveMovingObject = True

        # 如果有移动物体则进行目标检测
        if isHaveMovingObject:
            blobImage = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
            net.setInput(blobImage)
            cvOut = net.forward()

            for detection in cvOut[0, 0, :, :]:
                score = float(detection[2])
                objIndex = int(detection[1])
                if score > 0.6:
                    left = detection[3]*w
                    top = detection[4]*h
                    right = detection[5]*w
                    bottom = detection[6]*h

                    # 绘制
                    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                    cv2.putText(image, "score:%.2f, %s"%(score, objName[objIndex]),
                            (int(left) - 10, int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 8)
                    invade = True

            # 人脸识别
            face = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(24, 24))
            for (x, y, w, h) in face:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 68:     #如果置信度小于68则说明该人有很大可能是本人
                    name = names[idnum]
                    print(idnum)
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    name = "unknown"
                    print(confidence)
                    print(idnum)
                    confidence = "{0}%".format(round(100 - confidence))

                cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(image, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        # 如果有人入侵则录制视频
        if invade:
            out.write(image)
        #  显示
        cv2.imshow('demo', image)

        k = cv2.waitKey(100)& 0xff
        # 27对应Esc，当点击该键时退出
        if k == 27:
            break
        elif k == 32:
            while cv2.waitKey(0) != 32:
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

#获取名字
def findName(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # 读取照片素材所有文件路径
    names = []
    for imagePath in imagePaths:
        name = os.path.split(imagePath)[-1].split("#")[1]
        if name in names:
            continue
        else:
            names.append(name)
    #names.reverse()
    return names


if __name__ == '__main__':
    path = 'D:/test/face'
    print(findName(path))
    names = findName(path)
    video_path = 0#"E:/demo.mp4"#"D:/example video.avi"#"D:/test2.mp4"
    imageClassification(video_path, names)
