# !/usr/bin/env python

from concurrent.futures import thread
from multiprocessing.connection import wait
from time import sleep
import matplotlib.pyplot as plt

import sys

from UnrealUtilities import UE5EnvWrapper
import fastai
import numpy
import pandas
import os
import shutil

env = UE5EnvWrapper()
fig, ax = plt.subplots()

numForwardImages = 0
numBackImages = 0
numLeftImages = 0
numRightImages = 0

def main():

    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.title("Unreal Engine View")
    plt.axis("off")
    ax.imshow(env.request_image(cameraNum=0))
    plt.show()
        
def onpress(event):
    path = "none"
    global numForwardImages
    global numBackImages
    global numLeftImages
    global numRightImages

    if event.key == "w" or event.key == "up":
        numForwardImages += 1
        path = env.save_image(0, f"forward_{numForwardImages}")
        env.forward(45)

    elif event.key == "a" or event.key == "left":
        numLeftImages += 1
        path = env.save_image(0, f"left_{numLeftImages}")
        env.left(45)

    elif event.key == "d" or event.key == "right":
        numRightImages += 1
        path = env.save_image(0, f"right_{numRightImages}")
        env.right(45)
    elif event.key == "s" or event.key == "down":
        numBackImages += 1
        path = env.save_image(0, f"down_{numBackImages}")
        env.back(45)
    else:
        return
    shutil.move(path, ".\TestData")
    ax.imshow(env.request_image(cameraNum=0))
    fig.canvas.draw()

if __name__ == "__main__":
    main()