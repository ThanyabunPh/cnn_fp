import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            # input 1 x (24 x 32) output 24 x (24 x 32), kernel_size= 2
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 2), padding=1),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # Input = 24 x 24 x 32, Output = 24 x 12 x 16
            # torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.AvgPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 2), padding=1),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(3, 2), padding=1),
            # torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(48 * 3 * 4, 288),
            torch.nn.ReLU(),
            torch.nn.Linear(288, 4)
        )

    def forward(self, x):
        return self.model(x)