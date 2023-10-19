import cv2 as cv
import os
import numpy as np
import math
import torch


class GenerateData(torch.utils.data.Dataset):
    def __init__(self, sources):
        # print(sources.shape)
        to1hot = np.eye(2)
        self.__dataset = []
        for f, label in sources:
            l = to1hot[label]
            self.__dataset.append([f, l])

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, index):
        # print(self.sources[index].shape)
        vec, label = self.__dataset[index]
        return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class VideoData():
    def __init__(self, step, height, width):
        self.step = step
        self.height = height
        self.width = width

    def openVid(self, x):
        image_list = []
        vid = cv.VideoCapture(x)
        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
        print("frame = ", frame_count)
        print("-----LOADING DATA!---------")
        for i in range(3000):
            # print("==",i,"=")
            ret, frame = vid.read()
            frame = cv.resize(frame, (self.width, self.height))
            image_list.append(frame)
        print("-----------DATA LOADED!------------")
        return image_list

    def sliding_window(self, x):
        slided = []
        for i in range(int(len(x)/self.step)):
            # print("-------------------------", i, "-----------------------")
            slided.append((x[i*self.step:(i*self.step)+self.step]))
        return np.array(slided)


    def load_data(self, x):
        vid = np.array(self.openVid(x))
        vid = self.sliding_window(vid)

        vid_rearranged = np.einsum('klijc->kclij', vid)

        # print(vid.shape)
        print(vid_rearranged.shape)

        return vid_rearranged
    #
    # def forward(self, x, label):
    #     """
    #
    #     Args:
    #         x: file
    #         label: young or old
    #     Returns:
    #         x (channels, steps, height, width)
    #     """
    #
    #     #conver label to 1hotencoding
    #     to1hot = np.eye(2)
    #
    #     return torch.tensor(self.load_data(x), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    """
    VideoData(steps, height, width)
    return: torch tensor vector images (sliding window, channels, frames, width, height) and label(sliding window, one hot encoding[2] ) 
    """
    vid = VideoData(300, 100, 100)
    vec = vid.load_data("../../dataset/20220921/o1.mp4")
    print(vec.shape)