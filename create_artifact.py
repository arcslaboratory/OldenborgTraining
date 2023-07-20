"""
This is the code for creating an artifact through wandb.
"""
import wandb
from fastai.vision.all import *

import zipfile as zf
# Change zip file name to the folder used to save screenshots
files = zf.ZipFile('UE_Screenshots.zip', 'r')
files.extractall('UE_Screenshots')
files.close()

run = wandb.init(
    # Change project and notes to be specific to the artifact you're creating
    project="Multirun-testing-4K",
    entity="arcslaboratory",
    notes="A dataset of images collected over 20 different runs for testing.",
    job_type="dataset-upload"
)

artifact = wandb.Artifact(
    # Name your artifact something specific about the dataset
    name="perfect-dataset-no-texture",
    type="dataset"
)

# Change to the folder name used to save screenshots

artifact.add_dir("UE_Screenshots/UE_Screenshots", name="data")

run.log_artifact(artifact)






