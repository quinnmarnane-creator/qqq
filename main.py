import cv2
import mediapipe as mp
import numpy as np
import time

from filters import filters, filter_names


camera = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


mp_draw = mp.solutions.drawing_utils


current_filter = 0

last_switch = 0

cooldown = 5

show_skeleton = True



def create_hand_mask(frame, results):

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )


    if not results.multi_hand_landmarks:
        return mask


    h, w = frame.shape[:2]


    # Only use fingertips
    fingertip_ids = [
        4,   # thumb
        8,   # index
        12,  # middle
        16,  # ring
        20   # pinky
    ]


    points = []


    for hand in results.multi_hand_landmarks:

        for tip in fingertip_ids:

            landmark = hand.landmark[tip]


            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )


            points.append(
                [x, y]
            )


    if len(points) >= 3:

        points = np.array(
            points,
            dtype=np.int32
        )


        hull = cv2.convexHull(
            points
        )


        cv2.fillConvexPoly(
            mask,
            hull,
            255
        )


    mask = cv2.GaussianBlur(
        mask,
        (21,21),
        0
    )


    return mask




def detect_peace(results):

    if not results.multi_hand_landmarks:
        return False


    for hand in results.multi_hand_landmarks:

        lm = hand.landmark


        index = lm[8].y < lm[6].y

        middle = lm[12].y < lm[10].y

        ring = lm[16].y > lm[14].y

        pinky = lm[20].y > lm[18].y


        if index and middle and ring and pinky:
            return True


    return False





while True:


    success, frame = camera.read()


    if not success:
        break



    frame = cv2.flip(
        frame,
        1
    )



    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )



    results = hands.process(
        rgb
    )



    if show_skeleton and results.multi_hand_landmarks:


        for hand in results.multi_hand_landmarks:


            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )




    if detect_peace(results):


        if time.time() - last_switch > cooldown:


            current_filter += 1


            if current_filter >= len(filters):

                current_filter = 0


            last_switch = time.time()





    mask = create_hand_mask(
        frame,
        results
    )



    filtered = filters[current_filter](
        frame
    )



    output = frame.copy()



    output[mask == 255] = filtered[mask == 255]




    cv2.putText(
        output,
        "S = Toggle Skeleton",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )



    cv2.putText(
        output,
        "Peace Sign = Next Filter",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )



    cv2.putText(
        output,
        "ESC = Close",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )



    cv2.putText(
        output,
        "FILTER: " + filter_names[current_filter],
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )



    cv2.imshow(
        "Gesture Filter",
        output
    )



    key = cv2.waitKey(1)



    if key == ord("s"):

        show_skeleton = not show_skeleton



    if key == 27:

        break





camera.release()

cv2.destroyAllWindows()