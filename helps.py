import os # шапка с импортами 
import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from time import perf_counter
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output, display

def plot_stats(train_loss_accuracy, test_loss_accuracy): # класс для рисования графиков loss и accuracy
    fig = plt.figure(figsize = (16,7))
    epoch = len(np.array(train_loss_accuracy)[:, 0])
    plt.subplot(1, 2, 1) 
    plt.plot(range(1, epoch + 1), np.array(train_loss_accuracy)[:, 0], label = 'Train loss')
    plt.plot(range(1, epoch + 1), np.array(test_loss_accuracy)[:, 0], label = 'Test loss')
    plt.ylabel('Loss'); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), np.array(train_loss_accuracy)[:, 1], label = 'Train accuracy')
    plt.plot(range(1, epoch + 1), np.array(test_loss_accuracy)[:, 1], label = 'Test accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid();
    
    clear_output(wait=True)
    display(plt.gcf())
    plt.close()
class NN_name(nn.Module): # определение модели со всеми необходимыми функциями
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.steplr_enable = 25
        pass 
        
    def forward(self, x):
        pass
    
    def fit(self, train_loader):
        self.train()
        torch.enable_grad()
        self.train_loss = 0
        total = 0
        correct = 0
        for x, y in tqdm(train_loader, desc = 'Train'):
            x, y = x.to(device), y.to(device)
            self.optim.zero_grad()
            out = self(x)
            loss = self.loss_fn(out, y)
            self.train_loss += loss.item()
            loss.backward()
            self.optim.step()
            
            _, y_pred = out.max(dim=1)
            total += y.numel()
            correct += (y == y_pred).sum().item()
        self.train_loss  /= len(train_loader)
        self.train_accuracy = correct / total
        return self.train_loss, self.train_accuracy
    
    @torch.inference_mode()
    def predict(self, test_loader):
        self.eval()
        torch.no_grad()
        self.test_loss = 0
        total = 0
        correct = 0
        for x, y in tqdm(test_loader, desc = 'Test'):
            x, y = x.to(device), y.to(device)
            out = self(x)
            loss = self.loss_fn(out, y)
            self.test_loss += loss.item()
            
            _, y_pred = out.max(dim=1)
            total += y.numel()
            correct += (y == y_pred).sum().item()
        self.test_loss  /= len(test_loader)
        self.test_accuracy = correct / total
        return self.test_loss, self.test_accuracy
    
    def octava(self, train_loader, test_loader, epochs = 25): # Полный цикл обучения
        self.to(device)
        self.train_loss_accuracy, self.test_loss_accuracy = [], []
        self.optim = torch.optim.Adam(self.parameters(), 1e-3)
        if self.steplr_enable: scheduler = StepLR(self.optim, step_size = self.steplr_enable)
        d = perf_counter()
        for e in range(epochs):
            self.train_loss_accuracy += [self.fit(train_loader)]
            self.test_loss_accuracy += [self.predict(test_loader)]
            visualize(self, next(iter(test_loader)))
            plot_stats(self.train_loss_accuracy, self.test_loss_accuracy)
            if self.steplr_enable: scheduler.step()
            
        print(f'Time taken (secs): {int(perf_counter() - d)}')
        return self