import cv2
import time
import os
import Hand_Detector_Module as hdm

camW , camH = 1000, 1400
cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)
pre_time = 0

imgFolderPath = "Images"
imgList = os.listdir(imgFolderPath)
# print(imgList)

detector = hdm.HandDetector()
newList = []
for imgPath in imgList:
    img = cv2.imread(f'{imgFolderPath}/{imgPath}')
    # print(img)
    newList.append(img)
# print(len(newList)) 
tipIds = [8, 12, 16, 20]

while True:
    success, frame = cap.read()
    # frame[0:200, 0:200] = newList[0]
    frame = detector.FindHands(frame, False)
    lmList = detector.FindPosition(frame, False)

    if len(lmList) != 0:
        fingers = []

        ### Thumb
        # For Left Hand
        if lmList[12][1] < lmList[16][1]:
            if lmList[4][1] < lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # For Right Hand
        elif lmList[12][1] > lmList[16][1]:
            if lmList[4][1] < lmList[3][1]:
                fingers.append(0)
            else:
                fingers.append(1)

        ### Other 4 fingers    
        for id in range(0, 4):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0) 

        # print(sum(fingers))  

        ### For showing The images of hand sign
        sumFig = 0
        sumFig = sum(fingers)

        h, w, c = newList[sumFig].shape
        frame[0:h, 0:w] = newList[sumFig]

        cv2.rectangle(frame, (20, 225), (170, 425), (255, 0, 255), cv2.FILLED)
        cv2.putText(
            frame,
            str(sumFig),
            (50, 385),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5,
            (0, 0, 0),
            5
        )

    current_time = time.time()
    fps = 1/ (current_time-pre_time)
    pre_time = current_time

    # FPS Box
    start_point = (500, 0)
    end_point = (650, 32)
    color = (205, 0, 0)
    thickness = -1

    FPS_Box = cv2.rectangle(frame, start_point, end_point, color, thickness) 

    # FPS value
    cv2.putText(
        FPS_Box,
        str(int(fps)),
        (575, 26),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,
        (255, 201, 255),
        2
    )

    # Text 'FPS'
    cv2.putText(
        FPS_Box,
        "FPS: ",
        (520, 24),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
        (255, 201, 255),
        2
    )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == ord('0'):
        break
    
cap.release()
cv2.destroyAllWindows() 