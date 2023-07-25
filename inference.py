from time import sleep
from fastai.vision.all import *
from ue5osc import Communicator
import os
import pathlib
from math import radians
from argparse import ArgumentParser, Namespace
import time

# Need it otherwise we get an Error PoxisPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def label_func(filename):
    direction = filename.split("/")[-1].split(".")[0].split("_")[-1]

    if direction[0] == "+":
        return "left"
    elif direction[0] == "-":
        return "right"
    else:
        return "forward"


def get_image(filename: str) -> bytes:
    """Requests the image we saved."""
    from PIL import Image

    image = Image.open(filename)
    display_image = image.show()
    return image


def main():
    argparser = ArgumentParser("Run inference on model.")
    argparser.add_argument("--model", type=str, help="Choose which model to load in.")
    argparser.add_argument(
        "file_path", type=str, help="Choose where to save your image to."
    )
    argparser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Address")
    argparser.add_argument(
        "--ue_port", type=int, default=7447, help="Unreal Engine OSC server port."
    )
    argparser.add_argument(
        "--py_port", type=int, default=7001, help="Python OSC server port."
    )
    args = argparser.parse_args()

    learn = load_learner("c:\\Users\\alnj2022\\Downloads\\model.pkl")

    movement_increment = 50
    rotation_increment = radians(10)

    file_path = Path(args.file_path).resolve() if args.file_path else None

    if file_path:
        file_path.mkdir(parents=True, exist_ok=True)

    image_number = 0

    with Communicator(ip="127.0.0.1", ue_port=7447, py_port=7001) as osc_communicator:
        image_number += 1
        image_filepath = f"{file_path}/" f"{image_number:06}.jpg"

        osc_communicator.save_image(image_filepath)
        sleep(2.0)
        get_image(image_filepath)

        im = PILImage.create(image_filepath)

        # Remove the following line; no need to resize the image here
        # im = im.thumbnail((256, 256))

        predicted_direction, _, probs = learn.predict(im)
        print(f"We should go: {predicted_direction}")
        print(f"Probability we need to go {predicted_direction}: {probs[0]:.4f}")


if __name__ == "__main__":
    main()
