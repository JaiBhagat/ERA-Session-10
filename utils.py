# Albumentation 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision


train_transform = A.Compose(
    [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # A.CropAndPad(4),
        A.RandomCrop(32, 32),
        A.HorizontalFlip(),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False),
        ToTensorV2()
    ]
)

test_transform = A.Compose([A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ToTensorV2()])

# One cycle LR 

# Plot a graph between LR and iterations
lr_range = []


def triangular_plot(iterations, stepsize, lr_max, lr_min):
    for i in range(iterations):
        cycle = math.floor(1 + (i / (2 * stepsize)))
        x = abs((i / stepsize) - (2 * (cycle)) + 1)
        lr_t = lr_max + (lr_min - lr_max) * x
        lr_range.append(lr_t)

    plt.plot(range(iterations), lr_range)


# dataset loader
use_cuda = torch.cuda.is_available()


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def data_loaders(batch_size, train_transform, test_transform):
    torch.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset = Cifar10SearchDataset(transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    sampleloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, **kwargs)
    testset = Cifar10SearchDataset(train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)

    return trainloader, testloader, sampleloader


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

#     test_acc.append(100. * correct / len(test_loader.dataset))

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F


train_losses = []
test_losses = []
test_acc1= []
test_acc2= []
criterion = nn.CrossEntropyLoss()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
            
    test_acc_single = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(test_acc_single)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc_single))
    return test_loss


# training
criterion = nn.CrossEntropyLoss()
lr_list = []

train_losses = []
test_losses = []
train_acc = []
test_acc = []
lr_list = []


def train(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    total = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        lr_list.append(optimizer.param_groups[0]['lr'])
        acc = 100*correct/total
        train_acc.append(acc)
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')
