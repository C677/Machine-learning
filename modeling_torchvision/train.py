import os
import csv
import numpy as np
import torch
from engine import train_one_epoch, evaluate
import torch.utils.data
from PIL import Image, ImageFile
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
import pandas as pd
import collections
from tqdm import tqdm

"""
faster rcnn+mask rcnn 으로 구성
둘의 차이? faster rcnn은 object detection으로 끝나지만, mask rcnn은 instance segmentation까지 수행!
즉, faster rcnn은 물체가 있는 위치에 네모 박스만 쳤지만 mask rcnn은 물체의 mask를 따낸다.
따라서, object detection까지는 faster rcnn을 수행하고 instance segmentation을 위해 mask rcnn을 수행
<<two stage>>
ROI(물체가 있을지도 모르는 위치의 후보 영역) 제안 -> ROI에 대해 클래스 분류 및 bbox 회귀
따라서, 느리지만 성능은 좋음
"""

sample_root = 'C:\\Users\\hyoj_\\OneDrive\\Desktop\\sample\\'

def get_transform(train):
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model

def parse_one_annot(filepath, filename):
    data = pd.read_csv(filepath)
    boxes_array = data[data["filename"] == filename][["minX", "minY", "maxX", "maxY"]].values
    classnames = data[data["filename"] == filename][["classname"]]
    classes = []
    for i in range(len(classnames)) :
        if classnames.iloc[i, 0] =='covid-19' : classes.append(1)
        elif classnames.iloc[i, 0] =='nodule' : classes.append(2)
        elif classnames.iloc[i, 0] =='cancer' : classes.append(3)
    return boxes_array, classes
 
class OpenDataset(torch.utils.data.Dataset):
# 데이터셋을 생성하고 Dataloader로 데이터셋을 불러오는 클래스
# height와 width는 resize할 크기, transforms는 이미지 전처리(좌우 변환 등)를 의미
    def __init__(self, root, df_path, height, width, transforms=None):
        self.root = root
        self.transforms = transforms
        self.height = height
        self.width = width
        self.df = df_path
        names = pd.read_csv(df_path)[['filename']]
        names = names.drop_duplicates()
        self.imgs = list(np.array(names['filename'].tolist()))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
        if img_path.split('.')[-1] != 'png' : img_path += '.png'
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        box_list, classes = parse_one_annot(self.df, self.imgs[idx])
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        boxes = torch.as_tensor(box_list, dtype=torch.float32)

        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    num_classes = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train = OpenDataset(sample_root,sample_root+'annotations.csv', 512, 512, transforms = get_transform(train=True))
    dataset_test = OpenDataset(sample_root,sample_root+'annotations.csv', 512, 512, transforms = get_transform(train=False))

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-3])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-3:])

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset_train), len(dataset_test)))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=0.1)

    num_epochs = 10
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "model.pth")