from time import sleep
from fastai.vision.all import *
import pandas as pd

import torch
from torch import nn

from fastcore.meta import use_kwargs_dict

from fastai.callback.fp16 import to_fp16
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_one_cycle

from fastai.data.block import MultiCategoryBlock, DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import RandomSplitter, ColReader

from fastai.metrics import accuracy_multi, BaseLoss

from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner

from torchvision.models import resnet34
from UnrealUtilities import UE5EnvWrapper
from pathlib import WindowsPath

env = UE5EnvWrapper()

def main():
    learn = load_learner("./models/trained.pkl")
    
    
    while(env.isconnected()):
        imagePath = env.request_image(0)
        clas, clas_idx, probs = learn.predict(imagePath)
        print(clas)
        if(clas == 'right'):
            env.right(45)
        elif(clas == 'left'):
            env.left(45)
        elif(clas == 'forward'):
            env.forward(45)
        elif(clas == 'back'):
            env.back(45)
        sleep(2)

if __name__ == "__main__":
    main()