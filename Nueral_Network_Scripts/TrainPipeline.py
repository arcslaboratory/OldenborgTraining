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

from pathlib import WindowsPath

def main():
    path = './TestData'
    set_seed(2)
    fnames = get_image_files(path)
    pat = r'/([^/]+)_\d+.*'
    item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
    batch_tfms = [*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
    bs=4

    images = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=RandomSplitter(0.2),
                    get_y=RegexLabeller(pat = pat),
                    item_tfms=item_tfms,
                    batch_tfms=batch_tfms)

                    # path_im = path
    dls = images.dataloaders(path, bs=bs)

    # path_im = path
    dls = images.dataloaders(path, bs=bs)
    dls.vocab
    learn = vision_learner(dls, resnet34, pretrained=True, metrics=error_rate).to_fp16()
    learn.fit_one_cycle(1)
    learn.lr_find()
    learn.unfreeze()
    learn.fit_one_cycle(4, lr_max=slice(1e-4, 1e-3))
    learn.export("./models/trained.pkl")
    
if __name__ == "__main__":
    main()