import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ssl

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim = 256, depth= 8, kernel_size=7, patch_size=2, n_classes=10):
    return nn.Sequential(
        # replace patch embedding with pixel-shuffle down-sampling
        nn.PixelUnshuffle(patch_size),
        nn.Conv2d(12, dim, kernel_size=1, stride=1),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=True, num_workers=2)


model = ConvMixer()

checkpoint = torch.load('cifar10 classification/ConvMixer_ks7.pt')
model.load_state_dict(checkpoint)

model= model.cuda()


model.eval()

with torch.no_grad():
    for i, (X, y) in enumerate(testloader):
        X, y = X.cuda(), y.cuda()
        with torch.cuda.amp.autocast():
            output = model(X)
        test_acc+ = (output.max(1)[1] == y).sum().item()
        m += y.size(0)

print(f'Test Acc: {test_acc/m:.4f}')
