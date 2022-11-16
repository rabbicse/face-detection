import cv2 as cv2
import torch
from torch import nn
from torchvision.transforms import transforms

# from resnet_18 import Resnet
from resnet import ResNet, Bottleneck

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# net = Resnet(3, 10)
net = ResNet(Bottleneck, [3, 4, 6, 3])
# self.net = ResNet([2, 2, 2, 2]) # resnet 18
net = ResNet(Bottleneck, [3, 4, 6, 3])  # resnet 50
# fine tune
# Freeze the layers
for param in net.parameters():
    param.requires_grad = False
# Change the last layer to cifar10 number of output classes.
# Also unfreeze the penultimate layer. We will finetune just these two layers.
net.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.load_state_dict(torch.load('./cifar_net.pth'))

data = cv2.imread('./data/cifar10/train/airplane/0001.png', cv2.IMREAD_UNCHANGED)
# Convert BGR image to RGB image
data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)

transform = transforms.Compose(
    # [
    #     transforms.ToTensor(),
    #     transforms.Resize((112, 112)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ]
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

with torch.no_grad():
    tensor = transform(data)
    tensor = torch.reshape(tensor, (1, 3, 224, 224))
    # tensor = torch.reshape(tensor, (1, 3, 32, 32))

    result = net(tensor.cuda())
    print(result)
    _, predicted = torch.max(result, 1)
    print(predicted)
    print(classes[predicted[0]])
