import torch
from torch import nn
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score


class MnistNet(nn.Module):

    def __init__(self, in_channels = 3, pre_trained_weights = False):

        super().__init__()

        self.FeatureExtractor = nn.Sequential(nn.Conv2d(in_channels, 8, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.Conv2d(8, 16, (2, 2), stride=2, padding=1), nn.ReLU(),
                                              nn.Conv2d(16, 20, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.Conv2d(20, 24, (2, 2), stride=2, padding=1), nn.ReLU())

        self.Classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(24*9*9, 256), nn.Dropout(0.3), nn.ReLU(),
                                        nn.Linear(256, 64), nn.Dropout(0.3), nn.ReLU(),
                                        nn.Linear(64, 10))

        if pre_trained_weights != False:
            self.load(pre_trained_weights)

    def load(self, pre_trained_weights):
        self.load_state_dict(torch.load(pre_trained_weights))

    def forward(self, x):
        cnn_x = self.FeatureExtractor(x)
        out = self.Classifier(cnn_x)
        return out

    def train_step(self, trainloader, loss_fn, optimizer, device, gaussian_noise = 0):
         ### 1. Train
        self.train()
        ### 1.1 Define vars
        loss, accuracy = [], []

        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            ### 1.2 Feed the network
            X, y = batch[0].to(device), batch[1].to(device)
            if gaussian_noise > 0:
                X, y = self.augment_with_noise(X, y, gaussian_noise)
                # X, y = X.to(device), y.to(device)
            out = self(X)
            ### 1.3 Compute loss and back-propagate
            crn_loss = loss_fn(out, y)
            crn_loss.backward()
            optimizer.step()
            ### 1.4 Save results
            loss.append(crn_loss.data.item())
            accuracy.append(accuracy_score(y.cpu().numpy(), np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)

    def augment_with_noise(self, X, y, gaussian_noise):
        X = torch.cat([X, X + torch.randn(X.size()).to(device) * gaussian_noise])
        y = torch.cat([y, y])
        return X, y

    def eval_step(self, testloader, loss_fn, device, tqdm_disable=True):
        ### 1. Eval
        self.eval()
        ### 1.1 Define vars
        loss, accuracy = [], []

        with torch.no_grad():
            for batch in tqdm(testloader, disable=tqdm_disable):
                ### 1.2 Feed the network
                X, y = batch[0].to(device), batch[1].to(device)
                out = self(X)
                ### 1.3 Compute loss
                crn_loss = loss_fn(out, y)
                ### 1.4 Save results
                loss.append(crn_loss.data.item())
                accuracy.append(accuracy_score(batch[1], np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)
