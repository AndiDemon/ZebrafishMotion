import torch
import torch.nn as nn


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
