# encoding=utf-8
"""
    Created on 21:11 2018/11/8 
    @author: Jindong Wang
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 64
N_EPOCH = 50
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l = []


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data():
    import os
    if os.path.isfile('data/data_har.npz') == True:
        data = np.load('data/data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = 'Your root folder' + 'UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def load(x_train, y_train, x_test, y_test):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0,0,0,0,0,0,0), std=(0,0,0,0,0,0,0,0,0))
    ])
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)),
            # nn.BatchNorm1d()
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=6)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(-1, 64 * 26)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out,dim=1)
        return out


def train(model, optimizer, train_loader, test_loader):
    model.train()
    n_batch = len(train_loader.dataset) // BATCH_SIZE
    criterion = nn.CrossEntropyLoss()

    for e in range(N_EPOCH):
        correct, total_loss = 0, 0
        total = 0
        for index, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            output = model(sample)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            if index % 20 == 0:
                print('Epoch: [{}/{}], Batch: [{}/{}], loss:{:.4f}'.format(e + 1, N_EPOCH, index + 1, n_batch,
                                                                           loss.item()))
        acc_train = float(correct) * 100.0 / (BATCH_SIZE * n_batch)
        print('Epoch: [{}/{}], loss: {:.4f}, train acc: {:.2f}%'.format(e + 1, N_EPOCH, total_loss * 1.0 / n_batch, acc_train))


        with torch.no_grad():

            correct, total = 0, 0
            for sample, target in test_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                output = model(sample)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
        print('Epoch: [{}/{}], test acc: {:.2f}%'.format(e + 1, N_EPOCH, float(correct) * 100 / total))
        acc_test = float(correct) * 100 / total
        l.append([acc_train, acc_test])

def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)

if __name__ == '__main__':
    torch.manual_seed(10)
    x_train, y_train, x_test, y_test = load_data()
    print(y_train.shape)
    train_loader, test_loader = load(x_train.reshape((-1, 9, 1, 128)), y_train,
                                     x_test.reshape((-1, 9, 1, 128)), y_test)
    model = Network().to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    train(model, optimizer, train_loader, test_loader)
    ll = np.array(l, dtype=float)
    np.savetxt('result.csv',ll,fmt='%.2f',delimiter=',')
