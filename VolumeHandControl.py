import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# print(volRange)
minVol = volRange[0]
maxVol = volRange[1]
# print(volRange[1])
#
# # setting webcam parameters:
wCam, hCam = [640, 480]
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
#
# # initializing prevTime which will be used to calculate fps:
prevTime = 0

# creating an object that will incorporate the functionalities of handDetector class within htm module:
detector = htm.handDetector(detectionCon=0.7)

while True:
    # storing video frames:
    success, img = cap.read()

    # # finding hand landmarks in frame:
    img = detector.findHands(img)

    # finding list of landmark positions:
    lmList = detector.findPosition(img, draw=False)

    # THUMB_TIP has id=4 i.e. lmList[4] & INDEX_FINGER_TIP has id=8 i.e. lmList[8], we'll extract these two for volume
    # control:
    if len(lmList) != 0:
        # THUMB_TIP x & y coordinate:
        x1, y1 = lmList[4][1], lmList[4][2]
        # INDEX_FINGER_TIP x & y coordinate:
        x2, y2 = lmList[8][1], lmList[8][2]

        # Creating bigger circles against the above points which will serve as a differentiator:
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # connecting the circles via a line:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # calculating centre points:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        # finding connecting line length:
        length = math.hypot(x2-x1, y2-y1)

        # We print the length to determine tha maximum and minimum length(max & min extension of fingers)
        # print(length)

        # Using the numpy interpolation fnc we've interpolated (min and max hand extension) with speaker min & max vol.
        vol = np.interp(length, [40, 200], [minVol, maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        # changing the colour (as an indicator) that we've reached minimum length:
        if length < 40:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Calculating fps:
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    # Displaying fps:
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)



