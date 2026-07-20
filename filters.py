import cv2
import numpy as np
import time


def y2k_blue_filter(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = (hsv[:,:,0] + 115) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.8, 0, 255)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # RGB glitch
    b,g,r = cv2.split(result)

    shift = int(time.time()*8) % 10

    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)

    result = cv2.merge([b,g,r])


    # scanlines
    result[::4] = result[::4] * 0.7


    glow = cv2.GaussianBlur(
        result,
        (25,25),
        0
    )

    return cv2.addWeighted(
        result,
        0.8,
        glow,
        0.2,
        0
    )



def anime_filter(frame):

    smooth = cv2.bilateralFilter(
        frame,
        15,
        80,
        80
    )


    gray = cv2.cvtColor(
        smooth,
        cv2.COLOR_BGR2GRAY
    )


    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        4
    )


    edges = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )


    anime = cv2.bitwise_and(
        smooth,
        edges
    )


    hsv = cv2.cvtColor(
        anime,
        cv2.COLOR_BGR2HSV
    )


    hsv[:,:,1] = np.clip(
        hsv[:,:,1]*1.7,
        0,
        255
    )


    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )



def rainbow_flash_filter(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    hue = int(time.time()*40) % 180

    hsv[:,:,0] = (
        hsv[:,:,0] + hue
    ) % 180


    hsv[:,:,1] = np.clip(
        hsv[:,:,1]*2,
        0,
        255
    )


    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    # RGB split effect
    b,g,r = cv2.split(result)

    r = np.roll(r,5,axis=1)
    b = np.roll(b,-5,axis=1)

    return cv2.merge(
        [b,g,r]
    )



def hacker_filter(frame):

    green = np.zeros_like(frame)

    green[:,:,1] = frame[:,:,1]


    result = cv2.addWeighted(
        frame,
        0.25,
        green,
        1,
        0
    )


    # digital lines
    result[::6] = (
        result[::6] * 0.5
    )


    noise = np.random.randint(
        0,
        25,
        frame.shape,
        dtype=np.uint8
    )


    return cv2.add(
        result,
        noise
    )



def cyberpunk_filter(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    hsv[:,:,0] = (
        hsv[:,:,0]+150
    ) % 180


    hsv[:,:,1] = np.clip(
        hsv[:,:,1]*2,
        0,
        255
    )


    neon = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    glow = cv2.GaussianBlur(
        neon,
        (35,35),
        0
    )


    return cv2.addWeighted(
        neon,
        0.75,
        glow,
        0.25,
        0
    )



filters = [
    y2k_blue_filter,
    anime_filter,
    rainbow_flash_filter,
    hacker_filter,
    cyberpunk_filter
]


filter_names = [
    "cool blue",
    "ANIME VISION",
    "RAINBOW FLASH",
    "Green",
    "NEON"
]