#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn
import torchvision


class ResNetAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torchvision.models.resnet18(pretrained=True)
        # Remove layers
        self.encoder.avgpool = torch.nn.Identity()
        self.encoder.fc = torch.nn.Identity()

        # print(self.encoder)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            # torch.nn.BatchNorm2d(1024),
            # torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y


def main():
    ResNetAE()


if __name__ == '__main__':
    main()
