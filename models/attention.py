import torch.autograd
import torch
import torch.nn as nn
import math
from torch import Tensor


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        @param x: input shape (batch size, channel, feature)
        @return: Squeeze and Excitation on channel dimension
                 output shape (batch size, channel, feature)
        """
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, input, expand, Dropout=0.5):
        super().__init__()
        self.Block = nn.Sequential(
            nn.Conv1d(input, expand, 1, stride=1),
            nn.ReLU(),
            nn.Conv1d(expand, input, 1, stride=1),
            nn.ReLU(),
            nn.Dropout(Dropout)
        )

    def forward(self, x):
        x = self.Block(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.5, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class Attention(nn.Module):
    def __init__(self, d_model: int, embed_dim, num_head, num_layer, dropout=0.5, dff=2048, device='cuda:1'):
        super(Attention, self).__init__()
        self.num_layers = num_layer
        self.output_dim = 2
        self.dropout = nn.Dropout(p=dropout).to(device)
        self.seq_len = 300
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_head, bias=False).to(device)

        # ConvBlock
        self.convblock = self.Block = nn.Sequential(
            nn.Conv1d(self.seq_len, dff, 1, stride=1),
            nn.ReLU(),
            nn.Conv1d(dff, self.seq_len, 1, stride=1),
            nn.ReLU(),
        )
        # Squueze and Excitation
        self.se = SELayer(self.seq_len, r=4, use_max_pooling=True)

        # MLP Exit
        self.MLPExit = nn.Sequential(
            nn.Linear(embed_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, embed_dim),
            nn.ReLU(),
        )
        self.lin_exit = nn.Linear(embed_dim, self.output_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.soft = nn.Softmax(dim=1)


    def att_arch(self, x):
        x = self.dropout(x)

        for _ in range(self.num_layers):
            y = self.convblock(x)
            y = self.se(y)
            x = x + y

        for _ in range(self.num_layers):
            y, _ = self.MHA(x, x, x)
            x = x + self.norm(y)

            y = self.convblock(x)
            # y = self.MLPExit(x)
            y = self.se(y)
            x = x + self.norm(y)

        out = x[:, -1, :]
        for _ in range(1):
            out = self.MLPExit(out)
        out = self.soft(self.lin_exit(out))

        return out

    def forward(self, x):
        return self.att_arch(x)


class MLP_mixer(nn.Module):
    def __init__(self, d_model: int, embed_dim, num_head, num_layer, dropout=0.5, dff=2048, device='cuda:1'):
        super(MLP_mixer, self).__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.num_layers = num_layer
        self.output_dim = 2
        self.dff = dff
        self.dropout = nn.Dropout(p=dropout).to(device)
        self.device = device
        self.hidden_dim = embed_dim
        self.seq_len = 300
        # self.MHA = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head, bias=False, dropout=dropout).to(device)

        # self.att = MHAttention(dim=self.embed_dim, heads=self.num_head, dim_head=embed_dim, dropout=dropout)
        # self.ff = FeedForward(self.embed_dim, self.output_dim, dropout)

        # self.gcn = nn.

        # ConvBlock
        self.convblock = ConvBlock(input=self.seq_len, expand=self.dff, Dropout=dropout)

        # conv1D
        self.conv = nn.Conv2d(1, self.hidden_dim, (1, self.embed_dim), stride=1)
        # self.conv = nn.Conv1d(in_channels=1, out_channels=dff, kernel_size=(1, self.d_model), stride=1)
        self.conv_one = nn.Conv1d(in_channels=1, out_channels=dff, kernel_size=1, stride=1)
        self.conv_two = nn.Conv1d(in_channels=dff, out_channels=1, kernel_size=1, stride=1)
        self.batchnorm = nn.BatchNorm1d(25)
        self.max_pool = nn.MaxPool1d(2)

        self.se = SELayer(self.seq_len, r=4, use_max_pooling=True)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_head)
        # self.encoder_layer = nn.TransformerEncoderLayer(self.dff, nhead=self.num_head)
        self.transEncoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)

        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=self.num_head)
        self.transDecoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

        # Transformer Full
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.num_head, num_encoder_layers=self.num_layers,
                                          num_decoder_layers=self.num_layers)

        self.pos_one = PositionalEncoding(self.embed_dim)
        self.pos_two = PositionalEncoding(self.embed_dim)

        self.norm_two = nn.LayerNorm(self.d_model)
        self.norm_one = nn.LayerNorm(self.hidden_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.lin_one = nn.Linear(self.embed_dim, self.dff)
        self.lin_two = nn.Linear(self.embed_dim, self.output_dim)
        self.lin_three = nn.Linear(self.dff, self.dff)
        self.lin_four = nn.Linear(self.dff, self.embed_dim)
        self.lin_five = nn.Linear(self.embed_dim, self.d_model)
        self.lin_six = nn.Linear(self.embed_dim, self.dff)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.soft = nn.Softmax(dim=1)

        self.fc_out = nn.Linear(self.hidden_dim, self.embed_dim)

        # self.conv_out = nn.Conv1d(self.seq_len, self.pred_len, 1, stride=1)

    def mlp(self, x):
        # MLP
        x = self.lin_one(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin_four(x)
        x = self.relu(x)
        return x

    def mlp_exit(self, x):
        # MLP
        x = self.lin_six(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin_four(x)
        x = self.dropout(x)

        return x

    def mlp_mixer(self, x):
        # print("1 = ", x.shape)
        x = self.pos_one(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(dim=3).transpose(1, 2)

        # x = self.convblock(x)

        # print("x shape = ", x.shape)
        for _ in range(self.num_layers):
            y = self.norm(x)
            # print("y shape = ", y.shape)
            y = self.mlp(y)
            y = self.se(y)

            x = x + y

            y = self.norm(x)
            y = self.mlp_exit(y)
            y = self.se(y)

            x = x + y

        x = x[:, -1, :]
        out = self.soft(self.lin_two(x))

        return out

    def forward(self, x):
        return self.mlp_mixer(x)
