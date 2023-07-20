from fastai.vision.all import *
from pathlib import Path
from argparse import ArgumentParser
import os

compared_models = {
    "resnet18": resnet18,
    # "xresnext50": xresnext50,
    # "xresnext18": xresnext18,
    # "alexnet": alexnet,
    # "densenet121": densenet121,
}
#TODO fix
def get_action_from_filename(filename):
    return filename.split("_")[0]


def main():
    arg_parser = ArgumentParser("Train a model")
    arg_parser.add_argument("--dataset", type=str, help="dataset location")
    arg_parser.add_argument("--model", type=str, help="pre-trained model")
    arg_parser.add_argument("--valid_pct", type=float, default=0.2, help="percentage of images used for validation")
    arg_parser.add_argument("--image_resize", type=int, default=224, help="Image transform.")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="Change batch size.")
    arg_parser.add_argument("--num_epochs", type=int, default=5, help="Change number of epochs.")
    args = arg_parser.parse_args()

    dataset_path = Path(args.dataset)
    path = Path(os.getcwd())
    filenames = get_image_files(dataset_path)
    dls = ImageDataLoaders.from_name_func(
        path,
        valid_pct=args.valid_pct,
        item_tfms=Resize(args.image_resize),
        bs=args.batch_size,
        label_func=get_action_from_filename,
        fnames=filenames,
    )

    print("Validation dataset size:", len(dls.valid_ds))
    print("Training dataset size:", len(dls.train_ds))

    learn = vision_learner(dls, compared_models[args.model], metrics=accuracy)

    learn.fine_tune(args.num_epochs)

    learn.export("./models/" + args.model + ".pkl")


if __name__ == "__main__":
    main()