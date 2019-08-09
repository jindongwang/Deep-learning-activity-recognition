# encoding=utf-8
"""
    Created on 21:11 2018/11/8 
    @author: Jindong Wang
"""

import data_preprocess
import matplotlib.pyplot as plt
import network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import config_info


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []


def train(model, optimizer, train_loader, test_loader):
    n_batch = len(train_loader.dataset) // config_info['batch_size']
    criterion = nn.CrossEntropyLoss()

    for e in range(config_info['epoch']):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for index, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)
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
                print('Epoch: [{}/{}], Batch: [{}/{}], loss:{:.4f}'.format(e + 1, config_info['epoch'], index + 1, n_batch,
                                                                           loss.item()))
        acc_train = float(correct) * 100.0 / \
            (config_info['batch_size'] * n_batch)
        print(
            'Epoch: [{}/{}], loss: {:.4f}, train acc: {:.2f}%'.format(e + 1, config_info['epoch'], total_loss * 1.0 / n_batch,
                                                                      acc_train))

        # Testing
        model.train(False)
        with torch.no_grad():
            correct, total = 0, 0
            for sample, target in test_loader:
                sample, target = sample.to(
                    DEVICE).float(), target.to(DEVICE).long()
                sample = sample.view(-1, 9, 1, 128)
                output = model(sample)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
        acc_test = float(correct) * 100 / total
        print('Epoch: [{}/{}], test acc: {:.2f}%'.format(e + 1,
                                                         config_info['epoch'], float(correct) * 100 / total))
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        np.savetxt('result.csv', result_np, fmt='%.2f', delimiter=',')


def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(10)
    train_loader, test_loader = data_preprocess.load(
        batch_size=config_info['batch_size'])
    model = net.Network().to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(
    ), lr=config_info['lr'], momentum=config_info['momemtum'])
    train(model, optimizer, train_loader, test_loader)
    result = np.array(result, dtype=float)
    np.savetxt(config_info['result_file'], result, fmt='%.2f', delimiter=',')
    plot()
