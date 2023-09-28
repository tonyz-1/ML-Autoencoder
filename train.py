import argparse
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from torchvision import datasets, transforms

from model import autoencoderMLP4Layer


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, weightsPath, lossPlot):
    print('training...')
    model.train()
    losses_train = []
    device = torch.device('cpu')

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()
        losses_train += [loss_train / len(train_loader)]
        print('{} Epoch {}, Training loss {}'.format(datetime.now(), epoch, loss_train/len(train_loader)))

    xs = [x for x in range(len(losses_train))]
    plt.plot(xs, losses_train, label="Train")
    plt.draw()

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(lossPlot)
    torch.save(model.state_dict(), weightsPath)
    torchsummary.summary(autoencoderMLP4Layer(), (1, 28*28))


train_transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)
parser = argparse.ArgumentParser()
parser.add_argument('-z', '--bottleneck', type=int)
parser.add_argument('-e', '--epoch', type=int)
parser.add_argument('-b', '--batchSize', type=int)
parser.add_argument('-s', '--weight_file')
parser.add_argument('-p', '--loss_plot')
args = parser.parse_args()

model = autoencoderMLP4Layer()
bottleneck = args.bottleneck
epochs = args.epoch
batchSize = args.batchSize
trainLoader = DataLoader(train_set, batchSize, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
weightsPath = args.weight_file
lossPlot = args.loss_plot

train(epochs, optimizer, model, loss_fn, trainLoader, scheduler, weightsPath, lossPlot)



