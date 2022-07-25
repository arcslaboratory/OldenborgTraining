# !/usr/bin/env python
import matplotlib.pyplot as plt
from UnrealUtilities import UE5EnvWrapper
import shutil
import random

env = UE5EnvWrapper()
fig, ax = plt.subplots()

def main():

    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.title("Unreal Engine View")
    plt.axis("off")
    ax.imshow(env.request_image(cameraNum=0))
    plt.show()
"""
Called when user presses a key when plot window is focused
""" 
def onpress(event):
    path = "none"
    # Generate unique file name, for now simply a float between 0 and 1 with the '0.' removed
    num = str(random.random())
    num = num.split('.')
    imageName = num[1]
    
    datasetPath = "./Data/UE5Images"

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