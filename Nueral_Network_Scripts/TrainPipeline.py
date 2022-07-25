# from fastai.vision.all import *
# import pandas as pd

# import torch
# from torch import nn

# from fastcore.meta import use_kwargs_dict

# from fastai.callback.fp16 import to_fp16
# from fastai.callback.progress import ProgressCallback
# from fastai.callback.schedule import lr_find, fit_one_cycle

# from fastai.data.block import MultiCategoryBlock, DataBlock
# from fastai.data.external import untar_data, URLs
# from fastai.data.transforms import RandomSplitter, ColReader

# from fastai.metrics import accuracy_multi, BaseLoss

# from fastai.vision.augment import aug_transforms
# from fastai.vision.data import ImageBlock
# from fastai.vision.learner import cnn_learner

# from torchvision.models import resnet34

# from pathlib import WindowsPath

from fastai.vision.all import *
# from torchsummary import summary

def main():

    path = Path("./DataSets/UE5Images")
    print(path.ls())

    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224), num_workers=16, bs=2)
    # print(dls.show_batch())

    print("Validation dataset size:", len(dls.valid_ds))
    print("Training dataset size:", len(dls.train_ds))

    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # learn.lr_find()

    learn.fine_tune(4)

    # learn.show_results()

    interp = ClassificationInterpretation.from_learner(learn)
    # summary(learn.model)

    # plot_top_losses_fix(interp, 9)

    # interp.plot_confusion_matrix(figsize=(10, 10))

    learn.export("./models/resnet18.pkl")

    # path = './TestData'
    # set_seed(2)
    # fnames = get_image_files(path)
    # pat = r'/([^/]+)_\d+.*'
    # item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
    # batch_tfms = [*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
    # bs=4

    # images = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                 get_items=get_image_files,
    #                 splitter=RandomSplitter(0.2),
    #                 get_y=RegexLabeller(pat = pat),
    #                 item_tfms=item_tfms,
    #                 batch_tfms=batch_tfms)

    #                 # path_im = path
    # dls = images.dataloaders(path, bs=bs)

    # # path_im = path
    # dls = images.dataloaders(path, bs=bs)
    # dls.vocab
    # learn = vision_learner(dls, resnet34, pretrained=True, metrics=error_rate).to_fp16()
    # learn.fit_one_cycle(1)
    # learn.lr_find()
    # learn.unfreeze()
    # learn.fit_one_cycle(4, lr_max=slice(1e-4, 1e-3))
    # learn.export("./models/trained.pkl")
# def plot_top_losses_fix(interp, k, largest=True, **kwargs):
#     losses,idx = interp.top_losses(k, largest)
#     if not isinstance(interp.inputs, tuple): interp.inputs = (interp.inputs,)
#     if isinstance(interp.inputs[0], Tensor): inps = tuple(o[idx] for o in interp.inputs)
#     else: inps = interp.dl.create_batch(interp.dl.before_batch([tuple(o[i] for o in interp.inputs) for i in idx]))
#     b = inps + tuple(o[idx] for o in (interp.targs if is_listy(interp.targs) else (interp.targs,)))
#     x,y,its = interp.dl._pre_show_batch(b, max_n=k)
#     b_out = inps + tuple(o[idx] for o in (interp.decoded if is_listy(interp.decoded) else (interp.decoded,)))
#     x1,y1,outs = interp.dl._pre_show_batch(b_out, max_n=k)
#     if its is not None:
#         plot_top_losses(x, y, its, outs.itemgot(slice(len(inps), None)), interp.preds[idx], losses,  **kwargs)

if __name__ == "__main__":
    main()