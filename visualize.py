import cv2 as cv
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn

from utils.dataset import *

# Set CUDA device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=8,  # 4 * (x, y)
            hidden_size=128,  # Any
            num_layers=1,  # Number of LSTM stack, usually 1.
            batch_first=True  # (batch, t, feature)
        )
        self.out = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            # nn.ReLU(),
            nn.Linear(128, 2),
            # nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden0=None):
        out, (hidden, cell) = self.lstm(x, hidden0)  # LSTM Layer

        # print("out   :", out.shape)
        # print("hidden:", hidden.shape)
        # print("cell  :", cell.shape)

        # Many-to-one
        #   https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/06/01-RNN-Many-to-one.html

        out = self.out(out[:, -1, :])  # FC Layer
        out = self.softmax(out)

        return out


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.gru = nn.GRU(
            input_size=8,  # 4 * (x, y)
            hidden_size=128,  # Any
            num_layers=1,  # Number of LSTM stack, usually 1.
            batch_first=True  # (batch, t, feature)
        )
        self.out = nn.Sequential(
            nn.Linear(self.gru.hidden_size, 128),
            # nn.ReLU(),
            nn.Linear(128, 2),
            # nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden0=None):
        out, _ = self.gru(x, hidden0)  # LSTM Layer

        # print("out   :", out.shape)
        # print("hidden:", hidden.shape)
        # print("cell  :", cell.shape)

        # Many-to-one
        #   https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/06/01-RNN-Many-to-one.html

        out = self.out(out[:, -1, :])  # FC Layer
        out = self.softmax(out)

        return out


def load_file(folder, index):
    print("index = ", index)
    df = pd.read_csv(folder, delimiter="\t")
    df = df[["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"]]

    df = TrajectoriesData.normalize_coordinate(df)
    result = np.zeros((2, 300, 8))
    result[0] = df[index:index + 300]
    result[1] = df[index + 300:index + 600]
    result = torch.tensor(result, dtype=torch.float32)
    return result


def class_interpret(pred):
    result = []
    for i in range(len(pred)):
        r = "OLD"
        if pred[i][0] > pred[i][1]:
            r = "YOUNG"
        result.append(r)
    return result


def main():
    print("vis")
    folder = "../../dataset/20220527/"
    files = ["o1", "y1"]

    cap_o = cv.VideoCapture(folder + files[0] + ".MTS")
    cap_y = cv.VideoCapture(folder + files[1] + ".MTS")
    # Check if camera opened successfully
    if (cap_o.isOpened() == False):
        # print(cap)
        print("Error opening video stream or file")

    size = (int(cap_o.get(3)), int(cap_o.get(4)))
    result = cv.VideoWriter('./vis/' + files[0] + files[1] + '__.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, size)

    print("cap = ", cap_o.get(cv.CAP_PROP_FRAME_COUNT))
    rand_ind_o = random.randint(600, int(cap_o.get(cv.CAP_PROP_FRAME_COUNT) - 600))
    rand_ind_y = random.randint(600, int(cap_y.get(cv.CAP_PROP_FRAME_COUNT) - 600))
    print("rand o = ", rand_ind_o)
    print("rand y = ", rand_ind_y)

    model_name = "GRU"
    model_dirname = "./checkpoints/" + model_name

    model = torch.load(model_dirname, map_location=device)

    """load data from txt"""
    data_root = Path("/net/nfs2/export/home/andi/Documents/Zebrafish/dataset/20220527/idTracker/")
    old = data_root / "old_01_trajectories_nogaps.txt"
    young = data_root / "waka_01_trajectories_nogaps.txt"

    input_old = load_file(old, rand_ind_o)
    input_young = load_file(young, rand_ind_y)
    print("input old = ", input_old.shape)
    print("input young = ", input_young.shape)

    pred_old = model(input_old.to(device))
    pred_old = pred_old.to("cpu")
    real_old = class_interpret(pred_old)
    print("pred = ", pred_old.shape)
    print("pred = ", pred_old)
    print("pred = ", real_old)

    pred_young = model(input_young.to(device))
    pred_young = pred_young.to("cpu")
    real_young = class_interpret(pred_young)
    print("pred = ", pred_young.shape)
    print("pred = ", pred_young)
    print("pred = ", real_young)

    # describe the type of font
    # to be used.
    font = cv.FONT_HERSHEY_SIMPLEX

    # Read until video is completed
    count = 0
    torch.set_printoptions(precision=3)
    while (cap_o.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap_o.read()
        if ret == True:
            if (count >= rand_ind_o) & (count < rand_ind_o + 300):
                print(count)
                cv.putText(frame, 'GT = ' + 'OLD', (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_4)
                cv.putText(frame, 'Prediction = ' + real_old[0], (50, 90), font, 1, (255, 0, 0), 2, cv.LINE_4)
                cv.putText(frame, 'Weight = ' + str(round(pred_old[0][0].item(), 3)) + ', ' + str(
                    round(pred_old[0][1].item(), 3)),
                           (50, 130),
                           font, 1, (0, 255, 0), 2, cv.LINE_4)

                result.write(frame)
            elif (count >= rand_ind_o + 300) & (count < rand_ind_o + 600):
                print(count)
                cv.putText(frame, 'GT = ' + 'OLD', (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_4)
                cv.putText(frame, 'Prediction = ' + real_old[1], (50, 90), font, 1, (0, 0, 255), 2, cv.LINE_4)
                cv.putText(frame,
                           'Weight = ' + str(round(pred_old[1][0].item(), 3)) + ', ' + str(round(pred_old[1][1].item(), 3)),
                           (50, 130),
                           font, 1, (0, 255, 0), 2, cv.LINE_4)

                result.write(frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        count += 1
    count = 0
    while (cap_y.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap_y.read()
        if ret == True:
            if (count >= rand_ind_y) & (count < rand_ind_y + 300):
                print(count)
                cv.putText(frame, 'GT = ' + 'YOUNG', (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_4)
                cv.putText(frame, 'Prediction = ' + real_young[0], (50, 90), font, 1, (255, 0, 0), 2, cv.LINE_4)
                cv.putText(frame, 'Weight = ' + str(round(pred_young[0][0].item(), 3)) + ', ' + str(
                    round(pred_young[0][1].item(), 3)), (50, 130),
                           font, 1, (0, 255, 0), 2, cv.LINE_4)

                result.write(frame)
            elif (count >= rand_ind_y + 300) & (count < rand_ind_y + 600):
                print(count)
                cv.putText(frame, 'GT = ' + 'YOUNG', (50, 50), font, 1, (0, 255, 255), 2, cv.LINE_4)
                cv.putText(frame, 'Prediction = ' + real_young[1], (50, 90), font, 1, (0, 0, 255), 2, cv.LINE_4)
                cv.putText(frame, 'Weight = ' + str(round(pred_young[1][0].item(), 3)) + ', ' +str(
                    round(pred_young[1][1].item(), 3)), (50, 130),
                           font, 1, (0, 255, 0), 2, cv.LINE_4)

                result.write(frame)

                # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
        count += 1
    # When everything done, release the video capture object
    result.release()

    # Closes all the frames


if __name__ == '__main__':
    main()
