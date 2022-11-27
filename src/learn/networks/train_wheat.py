import gc
import os
import time
import warnings
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import V
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc, FrozenBatchNorm2d

warnings.filterwarnings("ignore")


def _ovewrite_value_param(param: Optional[V], new_value: V) -> V:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {param} instead.")
    return new_value


class WheatNet(FasterRCNN):
    def __init__(self, *,
                 weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
                 progress: bool = True,
                 num_classes: Optional[int] = None,
                 weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
                 trainable_backbone_layers: Optional[int] = None,
                 **kwargs: Any, ):
        weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
        weights_backbone = ResNet50_Weights.verify(weights_backbone)

        if weights is not None:
            weights_backbone = None
            num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        elif num_classes is None:
            num_classes = 91

        is_trained = weights is not None or weights_backbone is not None
        trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
        norm_layer = misc.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

        backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        super(WheatNet, self).__init__(backbone, num_classes=num_classes, **kwargs)

        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=progress))
            if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
                overwrite_eps(self, 0.0)

        num_classes = 2  # 1 class (person) + background
        in_features = self.roi_heads.box_predictor.cls_score.in_features

        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


class WheatDataset(Dataset):
    def __init__(self, root, folder='train', transform=None):
        self.transforms = []
        if transform is not None:
            self.transforms.append(transform)
        self.root = root
        self.folder = folder
        box_data = pd.read_csv(os.path.join(root, "train.csv"))
        self.box_data = pd.concat(
            [box_data, box_data.bbox.str.split('[').str.get(1).str.split(']').str.get(0).str.split(',', expand=True)],
            axis=1)
        self.imgs = list(os.listdir(os.path.join(root, self.folder)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.join(self.root, self.folder), self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        df = self.box_data[self.box_data['image_id'] == self.imgs[idx].split('.')[0]]
        if df.shape[0] != 0:
            df[2] = df[0].astype(float) + df[2].astype(float)
            df[3] = df[1].astype(float) + df[3].astype(float)
            boxes = df[[0, 1, 2, 3]].astype(float).values
            labels = np.ones(len(boxes))
        else:
            boxes = np.asarray([[0, 0, 0, 0]])
            labels = np.ones(len(boxes))
        for i in self.transforms:
            img = i(img)

        targets = {}
        targets['boxes'] = torch.from_numpy(boxes).double()
        targets['labels'] = torch.from_numpy(labels).type(torch.int64)
        # targets['id']=self.imgs[idx].split('.')[0]
        return img.double(), targets


class WheatTrain:
    def __init__(self):
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        self.root = os.path.abspath('../../data/global-wheat-detection/')

        # set hyper parameters
        self.img_size = 224
        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.batch_size = 1

        # Number of training epochs
        self.num_epochs = 20

        # Learning rate
        self.lr = 0.0001

        # Initiate net
        self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) #WheatNet()

        num_classes = 2  # 1 class (person) + background
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features

        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # set optimizer
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.01)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # set criterion to calculate loss
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        # Learning Rate Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def load_dataset(self):
        train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(self.img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.means,
            #                      std=self.stds)
        ])

        dataset = WheatDataset(root=self.root, folder='train', transform=train_transforms)

        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        train_dataset = torch.utils.data.Subset(dataset, indices[:-2500])
        test_dataset = torch.utils.data.Subset(dataset, indices[-2500:])

        self.train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                          collate_fn=lambda x: list(zip(*x)))
        self.test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                         collate_fn=lambda x: list(zip(*x)))

        print('Load data done!')

    def train_data(self):
        best_val_acc = 0
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # Training
            print("Training")
            losess = 0
            for images, targets in self.train_iterator:
                try:
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    model = self.net.double()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()

                    self.optimizer.zero_grad()
                    self.optimizer.step()
                except Exception as x:
                    print(x)

            print(f"Epoch: {epoch} Loss = {losses.item():.4f}")

        print('Finished Training')

        # save model
        model_path = os.path.abspath('models/fruits_net.pth')
        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = WheatTrain()
    trainer.load_dataset()
    trainer.train_data()
