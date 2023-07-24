"""
This script will run on the desktop. 

This can use packaged game or editor.

Robot can run into the wall.

"""
from time import sleep
from fastai.vision.all import *
from ue5osc import Communicator

import os

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

import sys

from math import radians
from argparse import ArgumentParser

import time


# # TODO fix (use diff naming)
# def get_action_from_filename(filename):
#     return filename.split("_")[0]


def get_action_from_filename(filename):
    direction = f.split("/")[-1].split(".")[0].split("_")[-1]

    if direction[0] == "+":
        return "left"
    elif direction[0] == "-":
        return "right"
    else:
        return "forward"


def main():
    argparser = ArgumentParser("Run inference on model.")
    argparser.add_argument("model", type=str, help="Choose which model to load in.")
    argparser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Address")
    argparser.add_argument(
        "--ue_port", type=int, default=7447, help="Unreal Engine OSC server port."
    )
    argparser.add_argument(
        "--py_port", type=int, default=7001, help="Python OSC server port."
    )
    # argparser.add_argument(
    #     "path_to_projects", type=str, help="path to where Unreal Engine 5 projects are stored"
    # )
    args = argparser.parse_args()

    # TODO change to Communicator
    with ue5osc.Communicator(
        args.ip,
        args.ue_port,
        args.py_port,
    ) as osc_communicator:
        sleep(0.5)

    learn = load_learner(f"./models/{args.model}")

    # path to where UE5 saves projects
    # project_path = args.path_to_projects
    # get the connected project's name
    # project_name = env.get_project_name()
    # path in project folder to saved folder
    # project_saved_path = project_name + "\\Saved"
    # path in project folder to screenshots according to default configurations
    # screenshot_folder = "Screenshots"
    # project_screenshot_path = project_saved_path + "\\" + screenshot_folder

    # check existence of screenshots folder and error out if folder exists
    # image_folder = os.path.join(project_path, project_screenshot_path)
    # if (os.path.isdir(image_folder)):
    #     sys.exit("ERROR: The \"Screenshots\" folder exists in the Saved directory of UE5 connected project file, please delete this folder and run again.")

    # TODO thoughtfully think about what these values should be
    movement_increment = 50
    rotation_increment = radians(10)

    image_num = 0
    # use context manager with communicator here; see UE5OSC demo for example
    """
    With Communicator 
        save an image (saves a file)
        sleep(add arg for sleep amount -> how long does UE take to take ss)
        read in image file
        pass to the NN
        act on NN decision 
    """
    # while env.is_connected():
    #     env.save_image(0)
    #     image_path = os.path.join(image_folder, 'WindowsEditor\HighresScreenshot{num:0{width}}.png'.format(num=image_num, width=5))
    #     clss, clss_idx, probs = learn.predict(image_path)
    #     print(clss)
    #     if clss == "right":
    #         env.right(rotation_increment)
    #     elif clss == "left":
    #         env.left(rotation_increment)
    #     elif clss == "forward":
    #         env.forward(movement_increment)
    #     elif clss == "back":
    #         env.back(movement_increment)
    #     image_num += 1
    #     sleep(1)
    # env.reset()

    # create new folder for screenshots based on time of inference completion and move images there
    # os.chdir(os.path.join(project_path, project_saved_path))
    # current_time = time.localtime()
    # new_folder_name = screenshot_folder + time.strftime("-%Y-%m-%d-%H%M", current_time)
    # os.rename(screenshot_folder, pathlib.Path(new_folder_name))
    # print(f"Inference images saved to: {project_saved_path}\\{new_folder_name}")


if __name__ == "__main__":
    label_func("artifacts/perfect-dataset-no-texture:v0/data/001_000117_+0p21.png")
