import os
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from xml.etree import ElementTree as et
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

warnings.filterwarnings("ignore")


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes, pretrained=True):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    elif obj.find('name').text == "without_mask":
        return 3
    return 0


def get_transform():
    return transforms.Compose([transforms.PILToTensor(),
                               transforms.ConvertImageDtype(torch.float)])


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, image_list, transform):
        self.transform = transform
        self.imgs = image_list

        self.img_dir, self.ann_dir = img_dir, ann_dir

    def __getitem__(self, idx):
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'

        img_path = os.path.join(self.img_dir, file_image)
        label_path = os.path.join(self.ann_dir, file_label)

        img = Image.open(img_path).convert("RGB")
        target = self.__generate_target(idx, label_path)

        # if self.transform is not None:
        #     # img, target = self.transform(img, target)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def __generate_target(image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            objects = soup.find_all('object')

            num_objs = len(objects)

            # get bounding box coordinates for each mask
            boxes = []
            labels = []
            for i in objects:
                boxes.append(generate_box(i))
                labels.append(generate_label(i))

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            img_id = torch.tensor([image_id])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Annotation is in dictionary format
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = img_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            return target


class FaceMaskTrain:
    def __init__(self):
        self.root = os.path.abspath('../../data/shoe_sandal_boot/')
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # set hyper parameters
        self.img_size = 64
        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.batch_size = 12

        # Number of training epochs
        self.num_epochs = 20

        # Learning rate
        self.lr = 0.0001

        # Initiate net
        self.net = get_model_instance_segmentation(4)

        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # set optimizer
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        # self.optimizer = torch.optim.SGD(params, lr=0.005,
        #                                  momentum=0.9, weight_decay=0.0005)

        # set criterion to calculate loss
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion.to(self.device)

        # Learning Rate Scheduler
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def load_dataset(self):
        imgs = list(sorted(
            os.listdir("/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/face-masks/images/")))
        train_ims, test_ims = train_test_split(imgs)
        print(len(train_ims), len(test_ims))

        train_datasets = MaskDataset(
            img_dir='/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/face-masks/images/',
            ann_dir='/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/face-masks/annotations/',
            image_list=train_ims,
            transform=get_transform())

        self.train_iterator = DataLoader(dataset=train_datasets,
                                         shuffle=True,
                                         num_workers=8,
                                         batch_size=self.batch_size,
                                         collate_fn=collate_fn)

        # examples = enumerate(self.train_iterator)
        # print(next(examples))
        # batch_idx, (example_data, example_targets) = next(examples)
        #
        # print(batch_idx)
        # print(example_targets)

        print('Load data done!')

    def train_data(self):
        best_val_acc = 0
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            self.net.train()

            i = 0
            epoch_loss = 0
            for imgs, annotations in self.train_iterator:
                i += 1
                imgs = list(img.to(self.device) for img in imgs)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                loss_dict = self.net([imgs[0]], [annotations[0]])
                losses = sum(loss for loss in loss_dict.values())

                epoch_loss += losses.item()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
            print(f'Epoch: {epoch} Loss: {epoch_loss}')

        print('Finished Training')

        # save model
        model_path = os.path.abspath('models/face_mask_net.pth')
        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = FaceMaskTrain()
    trainer.load_dataset()
    trainer.train_data()
