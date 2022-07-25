from fastai.vision.all import *

def main():

    path = Path("./DataSets/UE5Images")
    print(path.ls())

    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224), num_workers=16, bs=2)

    print("Validation dataset size:", len(dls.valid_ds))
    print("Training dataset size:", len(dls.train_ds))

    learn = vision_learner(dls, resnet18, metrics=accuracy)

    learn.fine_tune(4)

    interp = ClassificationInterpretation.from_learner(learn)

    learn.export("./DataSets/UE5Images/models/resnet18.pkl")

if __name__ == "__main__":
    main()