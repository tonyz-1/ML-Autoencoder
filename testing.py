import argparse
import random

import torch
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from model import autoencoderMLP4Layer
from torchvision import datasets, transforms


def interpolate(weight):
    model = autoencoderMLP4Layer()
    model.load_state_dict(torch.load(weight))
    model.eval()

    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = datasets.MNIST('./data/mnist', train=False, download=False, transform=eval_transform)

    idx1 = random.randint(0, len(eval_set) - 1)
    idx2 = random.randint(0, len(eval_set) - 1)

    img1 = eval_set.data[idx1].to(dtype=torch.float32)
    img1 = img1.flatten() / 255.0
    img2 = eval_set.data[idx2].to(dtype=torch.float32)
    img2 = img2.flatten() / 255.0

    output1 = model.encode(img1)
    output2 = model.encode(img2)
    output = model.decode(output1 + output2)
    output = output.view(28, 28)
    output = output.detach().numpy()

    steps = 10
    g = plt.figure()
    g.add_subplot(1, steps, 1)
    plt.imshow(output, cmap='gray')
    for i in range(steps):
        n1 = 1 - i/steps
        n2 = i/steps
        output = model.decode((n1 * output1) + (n2 * output2))
        output = output.view(28, 28)
        output = output.detach().numpy()

        g.add_subplot(1, steps, i + 1)
        plt.imshow(output, cmap='gray')
    plt.show()


def noisyVisualize(weight):
    model = autoencoderMLP4Layer()
    model.load_state_dict(torch.load(weight))
    model.eval()

    # idx = int(input("Enter index between 0 and 59999: "))
    # if idx > 59999 or idx < 0:
    #     print("Invalid Index")
    #     return
    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = datasets.MNIST('./data/mnist', train=False, download=False, transform=eval_transform)

    idx = random.randint(0, len(eval_set) - 1)
    img = eval_set.data[idx].to(dtype=torch.float32)
    img = img.flatten() / 255.0
    noisyImg = torch.rand(img.size())

    noisyOutput = model(noisyImg)
    noisyOutput = noisyOutput.view(28, 28)
    noisyOutput = noisyOutput.detach().numpy()

    h = plt.figure()
    h.add_subplot(1, 3, 1)
    plt.imshow(img.view(28, 28), cmap='gray')
    h.add_subplot(1, 3, 2)
    plt.imshow(noisyImg.view(28, 28), cmap='gray')
    h.add_subplot(1, 3, 3)
    plt.imshow(noisyOutput, cmap='gray')
    # plt.show()


def visualize(weight):
    model = autoencoderMLP4Layer()
    model.load_state_dict(torch.load(weight))
    model.eval()

    # idx = int(input("Enter index between 0 and 59999: "))
    # if idx > 59999 or idx < 0:
    #     print("Invalid Index")
    #     return
    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = datasets.MNIST('./data/mnist', train=False, download=False, transform=eval_transform)

    idx = random.randint(0, len(eval_set) - 1)
    img = eval_set.data[idx].to(dtype = torch.float32)
    img = img.flatten() / 255.0

    output = model(img)
    output = output.view(28, 28)
    output = output.detach().numpy()

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img.view(28, 28), cmap='gray')
    f.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--weight_path')
args = parser.parse_args()

print(args)

weights = args.weight_path

visualize(weights)
noisyVisualize(weights)
interpolate(weights)
