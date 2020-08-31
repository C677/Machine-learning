#The model using detectron2

We will conduct machine learning experiment through Faster R-CNN provided by [detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).
To speed up time of train and evaluate, we use Colab's GPU.

### 1. Set up the Colab environment

- 1) First, we need to enable GPUs for the notebook :

Navigate to Edit → Notebook Settings

Select GPU from the Hardware Accelerator drop-down

        2) Install dependencies and detectron2 :

!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html


        Now, you can do all necessary imports

import os
# detectron2 logger 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# library
import torch, torchvision
import numpy as np
import cv2
import json
import random
from google.colab.patches import cv2_imshow

# detectron2 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

        3) Mount google drive to Colab :

from google.colab import drive
drive.mount('/content/drive')
        When we do this, our current directory becomes '/content/drive/My Drive/'.





2. Define the custom Dataset

  we want to use a custom dataset, so we need to register our dataset(CT images, information of RoI).

we load the original dataset into list[dict] with a specification similar to COCO’s json annotations.

We used following fields.

- file_name: the full path to the image file.

- height, width: interger. The shape of the image.

- image_id (str or int): a unique id that identifies this image. 

- annotations (list[dict]): each dict corresponds to annotations of one instance in this image.

- bbox (list[float]): list of 4 numbers representing the bounding box of the instance. --> where RoI is

- bbox_mode (int): the format of bbox. It must be a member of structures.BoxMode. Supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.

- category_id (int): an integer in the range[0, num_categories-1] representing the category label.


we register our dataset through DatasetCatalog.register, so we also add its corresponding metadata through MetadataCatalog.get(dataset_name).some_key = some_value, to enable any features that need the metadata.

For example, below is a part of our train/val.json and code to register train/val data.

- train/val.json

"1.3.6.1.4.1.14519.5.2.1.6655.2359.102500633407588554681658808214.png": {
        "image_id": 0,
        "filename": "1.3.6.1.4.1.14519.5.2.1.6655.2359.102500633407588554681658808214.png",
        "height": 512,
        "width": 512,
        "annotations": [
            {
                "bbox": [
                    286,
                    310,
                    355,
                    402
                ],
                "bbox_mode": 0,
                "category_id": 0
            }
        ]
    }
bbox_mode :0 means BoxMode.XYXY_ABS
 category_id = 0 is cancer,  category_id = 1 is covid, and category_id = 2 is nodules.


- register train/val/test data

def get_diseases_dicts(imgdir, fn):
  json_file = os.path.join(imgdir, fn)
  with open(json_file) as f:
    imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
      record={}

      filename = os.path.join(imgdir, v["filename"])
    
      record["file_name"] = filename
      record["image_id"] = idx
      record["height"] = v["height"]
      record["width"] = v["width"]

      annos = v["annotations"]
      for i in annos:
              i["bbox_mode"]= BoxMode.XYXY_ABS
              i["category_id"]= int(i["category_id"])
         
      record["annotations"] = annos
      dataset_dicts.append(record)
  return dataset_dicts
       
from detectron2.data import DatasetCatalog, MetadataCatalog
#Registering the Dataset
for d in ["train", "val"]:
    DatasetCatalog.register("diseases/" + d, lambda d=d: get_dieases_dicts("/content/gdrive/My Drive/colab_data/" + d, d+".json"))
    MetadataCatalog.get("diseases/" + d).set(thing_classes=["cancer", "covid", "nodules"])
dieases_metadata = MetadataCatalog.get("diseases/train")
We created get_dieases_dicts(imgdir, fn) function to match the format in which data set information(train/val/test.json) is loaded in detectron2. 



If you want to make sure the data is loaded properly...

import random
import matplotlib.pyplot as plt
%matplotlib inline

dataset_dicts = get_diseases_dicts("/content/gdrive/My Drive/colab_data/train", "train.json")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=diseases_metadata, scale=0.8)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])




3. Train the model (ex. 3 epochs)

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
#Get the basic model configuration from the model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  
#Passing the Train and Validation sets
cfg.DATASETS.TRAIN = ("diseases/train",)
cfg.DATASETS.TEST = ()
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
 # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.005  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 8243  #No. of iterations   
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64 # default: 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # number of classes (cancer, covid, nodules) 
cfg.TEST.EVAL_PERIOD = 500 # No. of iterations after which the Validation Set is evaluated.
 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # save the model
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
We use faster_rcnn_R_50_FPN_3x.yaml and also use faster_rcnn_X_101_32x8d_FPN_3x.yaml. 

When training is finished, the model(model_final.pth) is saved on this path, cfg.OUTPUT_DIR.



If you want more informations of model_zoo_config_file, Take a look referneces.

How to calculate 'num of epochs'...
      one_epoch = num of data / IMS_PER_BATCH

      num_of_epochs = MAX_ITER / one_epoch



* we have 10989 datas for training so MAX_ITER is 8234 for 3 epochs


4. Test the model

        1) Load the model

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
Now it's time to test with the lung diseases validation dataset.

First, load the trained model and create a predictor.



       2) Draw prediction on image

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_dieases_dicts("/content/gdrive/My Drive/colab_data/val", "val.json")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=dieases_metadata, 
                   scale=0.8
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])

This shows the predicted results for any of the 3 images.

       3) Test the model

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("diseases/val", cfg, False, output_dir="/content/gdrive/My Drive/colab_data/output/")
val_loader = build_detection_test_loader(cfg, "diseases/val")
inference_on_dataset(trainer.model, val_loader, evaluator)

[references]
 Use Custom Datasets: (https://detectron2.readthedocs.io/tutorials/datasets.html)
