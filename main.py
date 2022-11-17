import pandas as pd
import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
import csv
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold

line = '\n-----------------------------------------------------------------------'

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 2), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 2), padding=1),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.Dropout(0.25),
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


# list of all data
x_train = []
x_test = []

y_train = []
y_test = []




lines = '\n-----------------------------------------------------------------------\n'

# setting dataset for test && train 
testing_files = ['test']
training_files = ['train']

#load data from csv && data preprocessing
print(lines)

print('Loading data...')
for i in testing_files:
    df = pd.read_csv('D:\\Alize\\MyWorkSpace\\Python\\Python_split_test\\Datasets\\' + str(i) + '.csv', encoding="utf8", on_bad_lines='warn')
    for row in range(len(df)):
        y_test.append(df.iloc[row, -1])
        x_test.append(df.iloc[row, 1:-1])

for j in training_files:
    df = pd.read_csv('D:\\Alize\\MyWorkSpace\\Python\\Python_split_test\\Datasets\\' + str(j) + '.csv', encoding="utf8", on_bad_lines='warn')
    for row in range(len(df)):
        y_train.append(df.iloc[row, -1])
        x_train.append(df.iloc[row, 1:-1])

print('Raw Training Images size: ', len(x_train))
print('Raw Training Label size: ', len(y_train), ' | Unique Label: ', np.unique(np.array(y_train)))
print('Raw Testing Images size: ', len(x_test))
print('Raw Testing Label size: ', len(y_test), ' | Unique Label: ', np.unique(np.array(y_test)))


print(lines)
print('Data preprocessing...')
x_trainImages = []
x_testImages = []
y_trainLabels = []
y_testLabels = []

Rawdata = [x_train, x_test]
data = [x_trainImages, x_testImages]

num = 0
for d in Rawdata:
    # Data Transforming
    for i in d:
        frame2D = []
        for h in range(24):
            frame2D.append([])
            for w in range(32):
                t = i[h * 32 + w]
                frame2D[h].append(t)

        data[num].append([frame2D])

    num += 1


x_trainImages = torch.FloatTensor(x_trainImages)
y_trainLabels = torch.LongTensor(y_train)
x_testImages = torch.FloatTensor(x_testImages)
y_testLabels = torch.LongTensor(y_test)
    
# Data Loader
print('Transformed X_trainImages Images size: ', x_trainImages.size())
print('Transformed Y_trainLabels Labels size: ', y_trainLabels.size())
print('Transformed X_testImages Images size: ', x_testImages.size())
print('Transformed Y_testLabels Labels size: ', y_testLabels.size())

# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(lines)
print('Using {} device'.format(device))
model = CNN().to(device)

# Hyperparameters
num_epochs = 200
learning_rate = 0.001
weight_decay = 0.01
batch_size = 1000
criterion = torch.nn.CrossEntropyLoss()

# test loop


# Model Setting
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# fold for the training set
kf = KFold(n_splits=10, shuffle=True)

# get the training set
train_set = data_utils.TensorDataset(x_trainImages, y_trainLabels)
train_loader = data_utils.DataLoader(train_set, batch_size=1000, shuffle=True)

# get the test set
# test_set = data_utils.TensorDataset(x_testImages, y_testLabels)
# test_loader = data_utils.DataLoader(test_set, batch_size=1000, shuffle=True)

training_set = ConcatDataset([train_loader.dataset])
# testing_set = ConcatDataset([test_loader.dataset])

# Training loss and accuracy
train_loss_list = []
test_acc_list = []

# Test loss and accuracy

results = {}

# start training
print(lines)
print('Start training...')

for i in range(1):
    for fold, (images, Label) in enumerate(kf.split(training_set)):

        print(f'{lines}FOLD {fold}')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(images)
        test_subsampler = torch.utils.data.SubsetRandomSampler(Label)

        # Define data loaders for training and testing data in this fold
        train_load = torch.utils.data.DataLoader(training_set, batch_size=100, sampler=train_subsampler)
        test_load = torch.utils.data.DataLoader(training_set, batch_size=100, sampler=test_subsampler)

        for epoch in range(num_epochs):

            train_loss = 0.0
            for i, data in enumerate(train_load, 0):
                inputs, labels = data
                optimizer.zero_grad()

                # tell model to train
                model.train()

                # Extracting images and target labels for the batch being iterated
                inputs, labels = inputs.to(device), labels.to(device)

                # Calculating the model output and the cross entropy loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Updating weights according to calculated loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss_list.append(train_loss / len(train_load))
            if epoch % 2 == 0:
                print('Epoch: {}\{} \tTraining Loss: {:.6f}'.format(epoch + 2, num_epochs, train_loss / len(train_load)))
            # print(f"Training loss = {train_loss_list[-1]}")


        test_acc = 0.0
        total = 0.0



        model.eval()

        with torch.no_grad():
            print('Starting testing...')
            for i, data in enumerate(test_load, 0):

                inputs, labels = data

                # Extracting images and target labels for the batch being iterated
                inputs, y_true = inputs.to(device), labels.to(device)

                # Calculating the model output and the cross entropy loss
                outputs = model(inputs)

                # Calculating the accuracy of the model
                _, predicted = torch.max(outputs.data, 1)
                total += (predicted == y_true).sum().item()


                test_acc_list.append(total / len(test_load))
            print(f"Test acc for fold: {fold+1}: {sum(test_acc_list)/len(test_acc_list)}")
        results[fold] = sum(test_acc_list)/len(test_acc_list)


    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {fold} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')

    print(' ---- Model Testing End ---- ', '\n')



