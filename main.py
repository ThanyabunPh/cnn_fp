import time

import numpy as np
import torch

import torch.utils.data as data_utils

from model import CNN

import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console
from tqdm import tqdm

console = Console()

lines = "-------------------------------------------------------------------------------\n"


# from rich_dataframe import prettify

class dataUitls:
    def __init__(self, file_train, file_test):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        self.testing_files = [i for i in file_test]
        self.training_files = [i for i in file_train]

        self.files = self.testing_files + self.training_files

        self.x_trainImages = []
        self.x_testImages = []
        self.y_trainLabels = []
        self.y_testLabels = []

    def LoadingData(self):
        # reading the csv files testings set
        console.print(f'\n{lines}Loading data...')
        for i in self.files:
            df = pd.read_csv(f'.\\Dataset\\{i}.csv', encoding="utf8", on_bad_lines='warn')
            if (i in self.training_files):
                # training
                for row in tqdm(range(len(df)), desc=f"Loading {i}\t", total=len(df), unit=" rows"):
                    self.y_train.append(df.iloc[row, -1])
                    self.x_train.append(df.iloc[row, 1:-1])
            else:
                for row in tqdm(range(len(df)), desc=f"Loading {i}\t", total=len(df), unit=" rows"):
                    self.y_test.append(df.iloc[row, -1])
                    self.x_test.append(df.iloc[row, 1:-1])

        return self.x_train, self.x_test, self.y_train, self.y_test

    def Preprocessing(self):
        x_train, x_test, y_train, y_test = self.LoadingData()
        print(f'{lines}Preprocessing data...')
        print(f'x_train Raw Training Images size: {len(x_train)}\t|\tShape:{np.shape(x_train)}')
        print(f'y_train Raw Training Label  size: {len(y_train)}\t|\tShape:{np.shape(y_train)}\t|\tUnique Labels:{np.unique(y_train)}')
        print(f'x_test  Raw Testing  Images size: {len(x_test)}\t|\tShape:{np.shape(x_test)}')
        print(f'y_test  Raw Testing  Label  size: {len(y_test)}\t|\tShape:{np.shape(y_test)}\t|\tUnique Labels:{np.unique(y_test)}\n')

        Raw_Data = [x_train, x_test]
        Images = [self.x_trainImages, self.x_testImages]

        # Preprocessing
        num = 0
        time.sleep(1.5)
        print('Starting Preprocessing data...')
        for index, f in enumerate(Raw_Data):
            for i in tqdm(f, desc=f'Preprocessing {index}', total=len(f), unit=' rows'):
                frame2D = []
                for h in range(24):
                    frame2D.append([])
                    for w in range(32):
                        t = i[h * 32 + w]
                        frame2D[h].append(t)
                Images[num].append([frame2D])
            num += 1

        print('\nTransforming data...')
        self.x_trainImages = torch.FloatTensor(Images[0])
        self.y_trainLabels = torch.LongTensor(y_train)
        self.x_testImages = torch.FloatTensor(Images[1])
        self.y_testLabels = torch.LongTensor(y_test)

        print('Transformed X_trainImages Images size: ', self.x_trainImages.size())
        print('Transformed X_testImages Images size: ', self.x_testImages.size())
        print('Transformed Y_trainLabels Labels size: ', self.y_trainLabels.size())
        print('Transformed Y_testLabels Labels size: ', self.y_testLabels.size())

        return self.x_trainImages, self.x_testImages, self.y_trainLabels, self.y_testLabels


class Training:
    def __init__(self, file_train=[], file_test=[], epochs=10, n_iters=3000, batch_size=32, learning_rate=0.001,
                 roundLoop=1):
        self.dataTransformed = dataUitls([i for i in file_train], [i for i in file_test]).Preprocessing()
        self.epochs = epochs
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.roundLoop = roundLoop

        self.train_scores = []
        self.test_scores = []

    def train(self):


        epochs = self.n_iters / (len(self.dataTransformed[0]) / self.batch_size)
        # epochs = int(self.epochs)
        batch_size = self.batch_size
        learning_rate = self.learning_rate

        roundLoop = self.roundLoop

        x_trainImages, x_testImages, y_trainLabels, y_testLabels = self.dataTransformed

        train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)
        train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = data_utils.TensorDataset(x_testImages, y_testLabels)
        test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

        # setting Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'{lines}Parameters:')
        print(
            f'Epochs: {int(epochs)}\t|\tBatch Size: {batch_size}\t|\tLearning Rate: {learning_rate}\t|\tDevice: {device}')

        model = CNN()
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        print(f'{lines}Model Summary:')
        for i in range(len(list(model.parameters()))):
            print(f'{list(model.parameters())[i].size()}')

        print(f'{lines}Training...')
        roundloop = [int(i) for i in range(1, roundLoop)]

        for round in roundloop:
            print(lines)
            for epoch in range(int(epochs)):

                train_total = 0.0
                train_loss = 0.0
                train_correct = 0.0

                for batch_id, (data, label) in enumerate(train_loader):
                    data = data.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    train_total += label.size(0)
                    train_correct += (predicted == label).sum().item()

                train_accuracy = 100 * train_correct / train_total
                self.train_scores.append(train_accuracy)

                test_total = 0.0
                test_loss = 0.0
                test_correct = 0.0

                with torch.no_grad():
                    for batch_id, (test_data, test_label) in enumerate(test_loader):
                        test_data = test_data.to(device)
                        test_label = test_label.to(device)
                        output = model(test_data)
                        loss = criterion(output, test_label)
                        test_loss += loss.item()

                        _, predicted = torch.max(output.data, 1)
                        test_total += test_label.size(0)
                        test_correct += (predicted == test_label).sum().item()

                    test_accuracy = 100 * test_correct / test_total
                    self.test_scores.append(test_accuracy)

                if epoch % 2 == 0:
                    train_loss = train_loss / train_total
                    print(f'Round {round} Epoch {epoch}\t | Train Loss: {train_loss:.10f} | Train Accuracy: {train_accuracy:.10f} | Test Accuracy: {test_accuracy:.10f}')
                    plotting(epoch, train_loss, train_accuracy, test_accuracy).plot_on_test()

        print('\nTraining Finished!')
        plt.grid(b=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
        plt.axhline(0.90, color="gold", linestyle="--", alpha=0.5, linewidth=1.0, label='base line')
        plt.title("cnn")
        plt.plot(roundloop, self.train_scores, '--', label='Train', color="darkgreen", alpha=0.5, linewidth=1.0)
        plt.plot(roundloop, self.test_scores, '--', label='Test', color="maroon", alpha=0.5, linewidth=1.0)
        plt.xticks(rotation=45, fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xlabel('Times', fontsize=10)
        plt.legend(fontsize=12, loc='lower right')
        plt.show()


if __name__ == '__main__':
    # 'training-merge'
    # 'testing-merge'
    Training(
        file_train=['even-merge'],
        file_test=['odd-merge'],

        epochs=200,
        n_iters=3000,
        batch_size=1000,
        learning_rate=0.001,

        roundLoop=50

    ).train()
