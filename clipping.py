#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import cv2


def iter_cap(cap: cv2.VideoCapture):
    count = -1
    while True:
        ret, frame = cap.read()
        count += 1

        if ret:
            yield count, frame
        else:
            break


def main():
    root = Path("") # directory to dataset
    path = root / "y1.MTS"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Failed to open video file {path}.")
        exit(0)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(w, h, n)
    avg = np.zeros((h, w, 3), dtype=float)

    for c, frame in iter_cap(cap):
        avg += frame / n

        if c % 1000 == 0:
            print(c)

    print(np.min(avg), np.max(avg))

    cv2.imwrite("average.jpg", avg)


if __name__ == '__main__':
    main()
