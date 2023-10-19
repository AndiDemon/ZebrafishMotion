#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn
# from torch.optim import SGD, Adam
from sklearn.metrics import confusion_matrix

from cnn.metrics import cmat_f1, cmat_accuracy, cmat_recall, cmat_specificity


# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


class TrajectoriesData(torch.utils.data.Dataset):
    def __init__(self, files: [(Path, int)]):
        super(TrajectoriesData, self).__init__()

        # 1clip = 10sec = 300frames
        self.__steps = 300

        to1hot = np.eye(2)
        self.__dataset = []
        for f, label in files:
            # print(f, label)
            self.__dataset += [
                (d, to1hot[label])
                for d in self.load_file(f, steps=self.steps)
            ]

    def __getitem__(self, index: int) -> (np.ndarray, int):
        vec, label = self.__dataset[index]

        return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.__dataset)

    @property
    def steps(self) -> int:
        return self.__steps

    @property
    def shape1(self) -> (int, int):
        """
        Feature shape
        """
        arr, _ = self.__dataset[0]
        return arr.shape

    @staticmethod
    def load_file(path: Path, steps: int):
        df = pd.read_csv(path, delimiter="\t")
        df = df[["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"]]
        # print(df.shape)

        df = TrajectoriesData.normalize_coordinate(df)

        result = []
        for t in range(0, df.shape[0], steps):
            clip = df.iloc[t:t + steps, :]

            # Ignore when not enough number of frames included
            if clip.shape[0] < steps:
                continue

            if clip.isnull().sum().sum() == 0:
                result.append(clip.to_numpy())

        return result

    @staticmethod
    def find_minmax(df: pd.DataFrame, label: str) -> (float, float):
        df = df.loc[:, df.columns.str.match(label)]

        return df.min().min(), df.max().max()

    @staticmethod
    def normalize_coordinate(df: pd.DataFrame) -> pd.DataFrame:
        x_min, x_max = TrajectoriesData.find_minmax(df, "X.+")
        y_min, y_max = TrajectoriesData.find_minmax(df, "Y.+")
        # print(x_min, x_max, ",", y_min, y_max)

        # Normalize by frame-size
        df[["X1", "X2", "X3", "X4"]] = (df[["X1", "X2", "X3", "X4"]] - x_min) / (x_max - x_min)
        df[["Y1", "Y2", "Y3", "Y4"]] = (df[["Y1", "Y2", "Y3", "Y4"]] - y_min) / (y_max - y_min)

        return df


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=8,       # 4 * (x, y)
            hidden_size=128,    # Any
            num_layers=1,       # Number of LSTM stack, usually 1.
            batch_first=True    # (batch, t, feature)
        )
        self.out = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            # nn.ReLU(),
            nn.Linear(128, 2),
            # nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden0=None):
        out, (hidden, cell) = self.lstm(x, hidden0)     # LSTM Layer

        # print("out   :", out.shape)
        # print("hidden:", hidden.shape)
        # print("cell  :", cell.shape)

        # Many-to-one
        #   https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/06/01-RNN-Many-to-one.html

        out = self.out(out[:, -1, :])                   # FC Layer
        out = self.softmax(out)

        return out


def main():
    # data_root = Path("~/workspace/zebrafish/_data/").expanduser()
    data_root = Path("/net/nfs2/export/dataset/morita/mie-u/zebrafish/20220527/idTracker/")

    # print(TrajectoriesData([
    #         # ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
    #         # ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
    #         # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
    #         ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
    # ]).shape1)
    print(len(TrajectoriesData([
        # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
        # ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
        # ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
        ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
    ])))
    # exit(0)

    train_loader = torch.utils.data.DataLoader(
        TrajectoriesData([
            # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
        ]),
        batch_size=64, shuffle=True, num_workers=os.cpu_count() // 2
    )
    valid_loader = torch.utils.data.DataLoader(
        TrajectoriesData([
            ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
            # ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
            # ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
            ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
        ]),
        batch_size=64, shuffle=False, num_workers=os.cpu_count() // 2
    )

    """
    x_train = np.array(x_train)[:, :, np.newaxis]
    y_train = np.array(y_train)[:, np.newaxis]
    """

    # Training
    model = LSTMModel().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)     # TODO: !!
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(100):
        train_loss = 0
        train_acc = 0
        cmat = np.zeros(shape=(2, 2))

        model.train()
        # for batch, i in enumerate(range(0, len(y_train), batch_size)):
        for batch, (x, y_true) in enumerate(train_loader):
            # x = torch.tensor(x_train[i:i + batch_size], dtype=torch.float)
            # y_true = torch.tensor(y_train[i:i + batch_size], dtype=torch.float)

            y_pred = model(x.to(device))
            y_pred = y_pred.to("cpu")

            # print(x.shape, "->", y_pred.shape, "vs.", y_true.shape)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_loader)
            # train_acc += np.sum(np.abs((y_pred.data - y_true.data).numpy()) / len(train_loader))

            cmat += confusion_matrix(
                torch.argmax(y_true, dim=1),
                torch.argmax(y_pred, dim=1)
            )
            train_acc = cmat_f1(cmat)

            print(
                "\rEpoch ({:03}), Batch({:03}/{}): loss={:.3}, f1={:.3}".format(
                    epoch + 1, batch+1, len(train_loader),
                    train_loss, train_acc
                ), end=""
            )
            # print("")
            # print(cmat)
            # break

        # print("")
        model.eval(

        )
        with torch.no_grad():
            valid_loss = 0.0
            cmat = np.zeros(shape=(2, 2))
            for batch, (x, y_true) in enumerate(valid_loader):
                y_pred = model(x.to(device))
                y_pred = y_pred.to("cpu")

                # print(x.shape, "->", y_pred.shape, "vs.", y_true.shape)

                loss = criterion(y_pred, y_true)
                valid_loss += loss.item() / len(train_loader)
                # train_acc += np.sum(np.abs((y_pred.data - y_true.data).numpy()) / len(train_loader))

                cmat += confusion_matrix(
                    torch.argmax(y_true, dim=1),
                    torch.argmax(y_pred, dim=1)
                )

            print(
                "\tValid Loss={:.3}, acc={:.3}, f1={:.3}, recall={:.3}, spec={:.3})".format(
                    valid_loss, cmat_accuracy(cmat), cmat_f1(cmat),
                    cmat_recall(cmat), cmat_specificity(cmat)
                ), end=""
            )
        print("")
        # break


if __name__ == '__main__':
    main()
