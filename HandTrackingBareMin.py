import cv2
import mediapipe as mp
import time


# Mediapipe? -> open source framework designed by Google aimed to build "world-class machine learning solutions". The
# code is written in C++ , but it can easily be deployed to any platform from web assembly to Android to macOS.
# Has its applications in -> persistent object tracking, AR hair colouring, pose tracking etc.


# MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D
# landmarks of a hand from just a single frame.


# MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that
# operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the
# cropped image region defined by the palm detector and returns high-fidelity 3D hand key points.


cap = cv2.VideoCapture(0)

# solutions.hands : indicates that we are going to use the hand tracking module.
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# To draw connections and trace indexes in hand landmarks would've involved a lot of maths and would've been very
# tedious, so we've a predefined utility fnc for drawing the landmarks:
mpDraw = mp.solutions.drawing_utils


# We are taking the current as well as previous time to determine the fps:
currTime = 0
prevTime = 0


while True:
    success, img = cap.read()
    # We need to send the RGB image to the object "hands" so we'll first convert (by default it's BGR):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # The process method will process the frame for us & give the required results.
    results = hands.process(imgRGB)
    # Now , taking into consideration the parameters in the Hands() fnc we can have multiple hands & therefore results
    # can have multiple hands.
    # To check if something is detected or not , we can write:
    # print(results.multi_hand_landmarks)
    # results.multi_hand_landmarks : will return all the hand marks with the index.

    # results.multi_hand_landmarks=true -> indexes where hand marks where detected:
    if results.multi_hand_landmarks:
        # results.multi_hand_landmarks can have multiple hands so for each handLms in results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw the landmarks detected for a single hand on the corresponding image via the draw_landmarks within
            # mp.solutions.drawing_utils
            # mpDraw.draw_landmarks(img, handLms)
            # The above commented line will only draw the points for the 21 landmarks.To draw the connections as well:
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # Now we are going to get the information within each hand (i.e. each handLms) -> we'll get the id no. as
            # the landmark information (x,y,z coordinate).
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                # Now we are going to utilize the x & y coordinate to the find the position of the landmark on the hand
                # (i.e. width pixels & height pixels)
                # height, width, colour channel of the image
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # This will give the x & y posn of the landmark corresponding to the id.
                print(id, cx, cy)
                # To differentiate a particular landmark index , we can draw diff shapes on a particular id.
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    # Displaying the fps:
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
