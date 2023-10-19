#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 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
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
import matplotlib.pyplot as plt

from utils.dataset import *
from models.rnn import *
from models.attention import *
from models.Poolformer import *

from cnn.metrics import cmat_f1, cmat_accuracy, cmat_recall, cmat_specificity

import warnings
import seaborn as sns
import cv2 as cv

warnings.simplefilter('ignore')

# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


def main():
    # no_gaps()
    # data_root = Path("~/workspace/zebrafish/_data/").expanduser()
    """"
    There are 2 dataset provided:
    1. 20220527 -> taken last year May
    2. 20220921 -> taken last year September
    """
    EPOCH = 100
    data_root = Path("/net/nfs2/export/home/andi/Documents/Zebrafish/dataset/20220527/idTracker/")
    data_root_new = Path("/net/nfs2/export/home/andi/Documents/Zebrafish/dataset/20220921/convert/")

    # print(TrajectoriesData([
    #         # ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
    #         # ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
    #         # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
    #         ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
    # ]).shape1)
    # no_gaps(data_root / "liang_y1.txt")
    print("TEST NEW DATA = ", len(TrajectoriesData([
        ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
        ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
        ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
        ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
        # ((data_root_new / "liang_y1.txt").expanduser(), 0),
        # ((data_root_new / "liang_y2.txt").expanduser(), 0),
        # ((data_root_new / "liang_o1.txt").expanduser(), 1),
        # ((data_root_new / "liang_o2.txt").expanduser(), 1),
    ])))

    train_loader = torch.utils.data.DataLoader(
        TrajectoriesData([
            ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
            # ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
            # ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
            ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root_new / "liang_y1.txt").expanduser(), 0),
            # ((data_root_new / "liang_y2.txt").expanduser(), 0),
            # ((data_root_new / "liang_o1.txt").expanduser(), 1),
            # ((data_root_new / "liang_o2.txt").expanduser(), 1),
        ]),
        batch_size=64, shuffle=True, num_workers=os.cpu_count() // 2
    )
    valid_loader = torch.utils.data.DataLoader(
        TrajectoriesData([
            # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root_new / "liang_y1.txt").expanduser(), 0),
            # ((data_root_new / "liang_y2.txt").expanduser(), 0),
            # ((data_root_new / "liang_o1.txt").expanduser(), 1),
            # ((data_root_new / "liang_o2.txt").expanduser(), 1),
        ]),
        batch_size=64, shuffle=False, num_workers=os.cpu_count() // 2
    )

    """
    x_train = np.array(x_train)[:, :, np.newaxis]
    y_train = np.array(y_train)[:, np.newaxis]
    """

    """
    Choose Model to use
    """
    # model_name = "LSTM"
    # learning_rate = 0.01
    # model = LSTMModel().to(device)

    # model_name = "GRU"
    # learning_rate = 0.01
    # model = GRUModel().to(device)

    model_name = "SelfAttention_4096_noExit"
    num_layers = 1
    d_model = 256
    embed_dim = 8
    dff = 4096
    num_heads = 4
    dropout_rate = 0.5
    learning_rate = 0.001
    model = Attention(d_model, embed_dim, num_heads, num_layers, dropout=dropout_rate,
                      dff=dff, device=device).to(device)

    # model_name = "Poolformer"
    # in_channel = 300
    # out_channel = 1
    # embed_dim = 8
    # out_class = 2
    # pool_size = 3
    # num_layers = 2
    # dff = 4096
    # dropout_rate = 0.5
    # learning_rate = 0.001
    # model = Poolformer(in_channel, out_channel, embed_dim, out_class, pool_size=pool_size, num_layer=num_layers,
    #                    dropout=dropout_rate,
    #                    dff=dff, device=device).to(device)

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO: !!
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters in the model:", total_params)

    model_dirname = "./checkpoints/" + model_name + ".pt"

    train_loss_all = []
    valid_loss_all = []

    train_acc_all = []
    valid_acc_all = []

    train_f1_all = []
    valid_f1_all = []

    train_recall_all = []
    valid_recall_all = []

    best_valid_f1 = 0
    for epoch in range(EPOCH):
        train_loss, train_f1, train_acc, train_recall = 0, 0, 0, 0
        cmat = np.zeros(shape=(2, 2))

        model.train()
        # for batch, i in enumerate(range(0, len(y_train), batch_size)):
        for batch, (x, y_true) in enumerate(train_loader):
            # x = torch.tensor(x_train[i:i + batch_size], dtype=torch.float)
            # y_true = torch.tensor(y_train[i:i + batch_size], dtype=torch.float)

            y_pred = model(x.to(device))
            y_pred = y_pred.to("cpu")

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
            train_f1 = cmat_f1(cmat)
            train_acc = cmat_accuracy(cmat)
            train_recall = cmat_recall(cmat)

            print(
                "\rEpoch ({:03}), Batch({:03}/{}): loss={:.3}, f1={:.3}".format(
                    epoch + 1, batch + 1, len(train_loader),
                    train_loss, train_f1
                ), end=""
            )

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            cmat = np.zeros(shape=(2, 2))
            for batch, (x, y_true) in enumerate(valid_loader):
                y_pred = model(x.to(device))
                y_pred = y_pred.to("cpu")

                loss = criterion(y_pred, y_true)
                valid_loss += loss.item() / len(valid_loader)
                # train_acc += np.sum(np.abs((y_pred.data - y_true.data).numpy()) / len(train_loader))

                cmat += confusion_matrix(
                    torch.argmax(y_true, dim=1),
                    torch.argmax(y_pred, dim=1)
                )
            valid_f1 = cmat_f1(cmat)
            valid_acc = cmat_accuracy(cmat)
            valid_recall = cmat_recall(cmat)
            print(
                "\tValid Loss={:.3}, acc={:.3}, f1={:.3}, recall={:.3}, spec={:.3})".format(
                    valid_loss, cmat_accuracy(cmat), cmat_f1(cmat),
                    cmat_recall(cmat), cmat_specificity(cmat)
                ), end=""
            )

        print("")
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)

        train_f1_all.append(train_f1)
        valid_f1_all.append(valid_f1)

        train_acc_all.append(train_acc)
        valid_acc_all.append(valid_acc)

        train_recall_all.append(train_recall)
        valid_recall_all.append(valid_recall)
        # break

        if valid_f1 >= best_valid_f1:
            # save model if the validation F1 score is better than the previous saved one
            best_valid_f1 = valid_f1
            torch.save(model, model_dirname)
            print("best valid f1 = ", best_valid_f1, ", best acc = ", valid_acc, ", best recall = ", valid_recall)


    fig1 = plt.figure(1)
    plt.plot(train_loss_all, 'r', label='Train loss')
    plt.plot(valid_loss_all, 'g', label='Val loss')
    plt.legend()
    # plt.show()
    plt.savefig("./eval/" + model_name + "_loss.eps")

    fig2 = plt.figure(2)
    plt.plot(train_f1_all, 'r', label='Train f1')
    plt.plot(valid_f1_all, 'g', label='Val f1')
    plt.legend()
    # plt.show()
    plt.savefig("./eval/" + model_name + "_f1.eps")

    fig3 = plt.figure(3)
    plt.plot(train_acc_all, 'r', label='Train acc')
    plt.plot(valid_acc_all, 'g', label='Val acc')
    plt.legend()
    # plt.show()
    plt.savefig("./eval/" + model_name + "_acc.eps")

    fig4 = plt.figure(4)
    plt.plot(train_recall_all, 'r', label='Train recall')
    plt.plot(valid_recall_all, 'g', label='Val recall')
    plt.legend()
    # plt.show()
    plt.savefig("./eval/" + model_name + "_recall.eps")


if __name__ == '__main__':
    main()
