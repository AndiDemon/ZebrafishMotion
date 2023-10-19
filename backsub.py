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
    root = Path("/net/nfs2/export/dataset/morita/mie-u/zebrafish/20220527/")
    path = root / "y1.MTS"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Failed to open video file {path}.")
        exit(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter("y1sub.avi", fourcc, 30, (1920, 1080))

    avg = cv2.imread(str(root / 'y1avg.jpg'))
    avg = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY).astype(int)
    print(avg.dtype)

    for c, frame in iter_cap(cap):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)

        frame = avg - frame
        frame[frame < 0] = 0

        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        writer.write(frame)

        # if c % 100 == 0:
        #     cv2.imwrite(f"f{c}.jpg", frame)
        #     print(c)
        # if c == 600:
        #     break

    writer.release()


if __name__ == '__main__':
    main()
