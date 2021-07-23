import os
import sys
import uuid
import cv2

def imageClassification(video_path):
    # 判断是否有入侵行为，如果有则生成视频与照片
    invade = False
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
    # 使用opencv预置人脸识别的模型
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    cap = cv2.VideoCapture(video_path)#"D:/example video.avi"
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
        # 获得所有层名称与索引
        #layerNames = net.getLayerNames()
        #lastLayerId = net.getLayerId(layerNames[-1])
        #lastLayer = net.getLayer(lastLayerId)
        #print(lastLayer.type)

        #人脸识别
        face = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(24, 24))
        for (fx, fy, fw, fh) in face:
            cv2.putText(image, "face", (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 1)

        # 检测
        blobImage = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
        net.setInput(blobImage)
        cvOut = net.forward()
        #print(cvOut)
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
                imageName = "D:/test/image" + str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(imageName, image)

        # 如果有人入侵录制视频
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

if __name__ == '__main__':
    video_path = 0#"D:/example video.avi"#"D:/test2.mp4"
    imageClassification(video_path)