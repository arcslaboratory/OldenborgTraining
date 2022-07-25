# !/usr/bin/env python

from concurrent.futures import thread
from multiprocessing.connection import wait
from random import Random
from time import sleep
import matplotlib.pyplot as plt

import sys

from sklearn import datasets

from UnrealUtilities import UE5EnvWrapper
import fastai
import numpy
import pandas
import os
import shutil
import random

# print(type(imageName))
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

    num = str(random.random())
    num = num.split('.')
    imageName = num[1]

    datasetPath = "C:/Users/simon/OneDrive/Documents/ArcsLab/ArcLabPrograms/OldenborgTraining/Nueral_Network_Scripts/DataSets/UE5Images"

    if event.key == "w" or event.key == "up":
        numForwardImages += 1
        path = env.save_image(0, f"{imageName}")
        shutil.move(path, f"{datasetPath}/forward")
        env.forward(30)

    elif event.key == "a" or event.key == "left":
        numLeftImages += 1
        path = env.save_image(0, f"{imageName}")
        shutil.move(path, f"{datasetPath}/left")
        env.left(30)

    elif event.key == "d" or event.key == "right":
        numRightImages += 1
        path = env.save_image(0, f"{imageName}")
        shutil.move(path, f"{datasetPath}/right")
        env.right(30)
    elif event.key == "s" or event.key == "down":
        numBackImages += 1
        path = env.save_image(0, f"{imageName}")
        shutil.move(path, f"{datasetPath}/back")
        env.back(30)
    else:
        return
    
    ax.imshow(env.request_image(cameraNum=0))
    fig.canvas.draw()

if __name__ == "__main__":
    main()