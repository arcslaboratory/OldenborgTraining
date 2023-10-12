from pathlib import Path
from fastai.vision.models import resnet18
from fastai.vision.learner import Learner, accuracy, vision_learner


def y_from_filename(rotation_threshold: float, filename: str) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"
    
def y_from_filename(rotation_threshold, filename) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))

    # threshold augments cutoff to mitigate sharp switches in direction
    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"


def wandb_load_learner(pth_file: str):
    # TODO: Delete metrics?
    model = vision_learner(dls, resnet18, metrics=accuracy)
    model.load(pth_file)
    return model


# create data loaders
def get_dls(args: Namespace, data_path: str):
    # NOTE: not allowed to add a type annotation to the input

    image_filenames: list = get_image_files(data_path)  # type:ignore

    # Using a partial function to set the rotation_threshold from args
    label_func = partial(y_from_filename, args.rotation_threshold)

    if args.use_command_image:
        return get_image_command_category_dataloaders(
            args, data_path, image_filenames, label_func
        )
    else:
        return ImageDataLoaders.from_name_func(
            data_path,
            image_filenames,
            label_func,
            valid_pct=args.valid_pct,
            shuffle=True,
            bs=args.batch_size,
            # item_tfms=Resize(224)
        )
