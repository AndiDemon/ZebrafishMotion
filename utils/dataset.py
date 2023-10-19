import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

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
        # print(np.array(self.__dataset[0][0]).shape)
        # print(len(self.__dataset))

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
        """
        @param path: Path to dataset folder
        @param steps: Number of clip size
        @return: List contained the structured sliding window data for input
        """
        df = pd.read_csv(path, delimiter="\t")
        df = df[["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"]]
        if "liang" in os.path.basename(path):
            """
            if new dataset is used, linear interpolation is applied 
            to get rid of the NaN value in the dataset
            """
            print("Liang true")
            df.interpolate(inplace=True)

        df = TrajectoriesData.normalize_coordinate(df)
        print("df shape : ", df.shape)

        result = []
        for t in range(0, df.shape[0], steps):
            clip = df.iloc[t:t + steps, :]

            # Ignore when not enough number of frames included
            if clip.shape[0] < steps:
                # print('< step')
                continue

            # at least 1 NAN is contained in a clip
            # Data is ignored
            if clip.isnull().sum().sum() == 0:
                # print('isnull')
                result.append(clip.to_numpy())

        # print("result shape : ", np.array(result).shape)
        return result

    @staticmethod
    def find_minmax(df: pd.DataFrame, label: str) -> (float, float):
        """
        @param df: Dataframe of dataset
        @param label: Column label of dataframe
        @return: min and max value
        """
        df = df.loc[:, df.columns.str.match(label)]
        return df.min().min(), df.max().max()

    @staticmethod
    def normalize_coordinate(df: pd.DataFrame) -> pd.DataFrame:
        """
        @param df: Dataframe from dataset
        @return: Normalized data of 4 fish coordinate x and y
        """
        x_min, x_max = TrajectoriesData.find_minmax(df, "X.+")
        y_min, y_max = TrajectoriesData.find_minmax(df, "Y.+")

        # Normalize by frame-size
        df[["X1", "X2", "X3", "X4"]] = (df[["X1", "X2", "X3", "X4"]] - x_min) / (x_max - x_min)
        df[["Y1", "Y2", "Y3", "Y4"]] = (df[["Y1", "Y2", "Y3", "Y4"]] - y_min) / (y_max - y_min)

        return df
