import os
import warnings
from glob import glob

import torch
import torchvision
from PIL import Image
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    x_min = int(obj.find('xmin').text)
    y_min = int(obj.find('ymin').text)
    x_max = int(obj.find('xmax').text)
    y_max = int(obj.find('ymax').text)
    return [x_min, y_min, x_max, y_max]


def generate_label(obj):
    if 'rikshaw' in obj.find('name').text.lower():
        return 1
    return 0


def get_transform():
    return transforms.Compose([transforms.PILToTensor(),
                               transforms.ConvertImageDtype(torch.float)])


class RickshawDataset(Dataset):
    def __init__(self, dataset_dir, transform):
        self.transform = transform
        self.dataset_dir = dataset_dir

        self.images = list(sorted(glob(os.path.join(self.dataset_dir, '*.jpg'))))
        self.annotations = list(sorted(glob(os.path.join(self.dataset_dir, '*.xml'))))

    def __getitem__(self, index):
        # im = cv2.imread(self.images[index])
        # cv2.namedWindow('FRS', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('FRS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('FRS', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img = Image.open(self.images[index]).convert("RGB")
        target = self.__generate_target(index, self.annotations[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def __generate_target(image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, 'lxml')
            objects = soup.find_all('object')

            # get bounding box coordinates for each mask
            boxes = []
            labels = []
            for obj in objects:
                boxes.append(generate_box(obj))
                # print(obj)
                # print(f'Label: {generate_label(obj)}')
                labels.append(generate_label(obj))

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            img_id = torch.tensor([image_id])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

            # Annotation is in dictionary format
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = img_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            # print(target)
            return target


class TrainRickshaw:
    def __init__(self):
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # set hyper parameters
        self.img_size = 64
        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.batch_size = 1

        # Number of training epochs
        self.num_epochs = 10

        # Learning rate
        self.lr = 0.0001

        # Initiate net
        self.net = get_model_instance_segmentation(2)

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
        dataset_dir = "/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/rickshaw_data/"
        train_datasets = RickshawDataset(
            dataset_dir=dataset_dir,
            transform=get_transform())

        self.train_iterator = DataLoader(dataset=train_datasets,
                                         shuffle=True,
                                         num_workers=8,
                                         batch_size=self.batch_size,
                                         collate_fn=collate_fn)

        # examples = enumerate(self.train_iterator)
        # nx = next(examples)
        # batch_idx, (example_data, example_targets) = nx
        #
        # print(batch_idx)
        # print(example_targets)

        print('Load data done!')

    def train_data(self):
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            self.net.train()

            i = 0
            epoch_loss = 0
            for imgs, annotations in self.train_iterator:
                i += 1
                imgs = list(img.to(self.device) for img in imgs)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]

                # loss_dict = self.net([imgs[0]], [annotations[0]])
                loss_dict = self.net(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                epoch_loss += losses.item()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                self.lr_scheduler.step()
            print(f'Epoch: {epoch} Loss: {epoch_loss}')

        print('Finished Training')

        # save model
        model_path = os.path.abspath(
            '/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/learn/models/rickshaw_net.pth')
        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = TrainRickshaw()
    trainer.load_dataset()
    trainer.train_data()
