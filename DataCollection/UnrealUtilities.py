#!/usr/bin/env python

from ast import Str
from tokenize import String
from numpy import pi, cos, sin
import matplotlib.pyplot as plt

import sys
from traitlets import Int

sys.path.append("/home/eoca2018/unrealcv/client/python")
import unrealcv
# from unrealcv import client as ue5  # type: ignore
from unrealcv.util import read_png  # type: ignore

ue5 = unrealcv.Client(("localhost", 8500))
class UE5EnvWrapper:
    def __init__(self):
        ue5.connect()
        if ue5.isconnected():
            print(ue5.request("vget /unrealcv/status"))
        else:
            print("Failed to Connect to UnrealCV server")
            exit(0)

        loc = ue5.request("vget /camera/0/location")
        rot = ue5.request("vget /camera/0/rotation")

        self.initial_x,self.initial_y,self.initial_z = loc.split()
        self.initial_pitch,self.initial_yaw,self.initial_roll = rot.split()

        # self.x = self.initial_x
        # self.y = self.initial_y
        # self.z = self.initial_z

        # self.pitch = self.initial_pitch
        # self.yaw = self.initial_yaw
        # self.roll = self.initial_roll
        
        self.cameraIDs = ["0", "1", "2"]

    def isconnected(self):
        return self.connected

    def reset(self):
        ue5.request(f"vset /action/keyboard backspace 1")
        for cameraID in self.cameraIDs:
            # ue5.request(f"vset /camera/{cameraID}/location {self.initial_x} {self.initial_y} {self.initial_z}")
            ue5.request(f"vset /camera/{cameraID}/rotation {self.initial_pitch} {self.initial_yaw} {self.initial_roll}")

    """
    Returns a string of form "pitch yaw roll"
    """
    def getRotation(self) -> Str:
        # Treat camera 0 as the robot for now
        return ue5.request("vget /camera/0/rotation")
        # ue5.receive

    def left(self, degreeRot:float):
        # IMPORTANT: In future all cameras on robot will be bounded to different locations on robot.
        # For network testing initially we will simply change the camera's POV for now
        rotVector = self.getRotation().split()
        currentPitch = rotVector[0]
        currentYaw = rotVector[1]
        currentRoll = rotVector[2]
        ue5.request(f"vset /camera/0/rotation {currentPitch} {float(currentYaw) - degreeRot} {currentRoll}")
        
    def right(self, degreeRot:float):
        # IMPORTANT: In future all cameras on robot will be bounded to different locations on robot.
        # For network testing initially we will simply change the camera's POV for now

        rotVector = self.getRotation().split()
        currentPitch = rotVector[0]
        currentYaw = rotVector[1]
        currentRoll = rotVector[2]

        ue5.request(f"vset /camera/0/rotation {currentPitch} {float(currentYaw) + degreeRot} {currentRoll}")
        # ue5.request(f"ke * right {value}")

    def forward(self, value:float):
        # Forward is a custom event for the press of the "up" arrow key in the UE5 Bluprint
        # This event is programmed locally int he UE5 blueprint for the robot, so that the robot still interacts and bumps into objects

        ue5.request(f"vset /action/keyboard up 1")
    
    def back(self, value:float):
        # Back is a custom command in the objects blueprint
        # This event is programmed locally int he UE5 blueprint for the robot, so that the robot still interacts and bumps into objects
        ue5.request(f"vset /action/keyboard down 1")

    def open_level(levelName:Str):
        ue5.request(f"open {levelName}")

    def request_image(self, cameraNum:Int):
        image_data = ue5.request(f"vget /camera/{cameraNum}/lit jpg")
        return read_png(image_data)
        
    def save_image(self, cameraNum:Int, annotation:Str):
        path = ue5.request(f'vget /camera/{cameraNum}/lit {annotation}.jpg')
        return path

    def show(self):
        plt.imshow(self.request_image())