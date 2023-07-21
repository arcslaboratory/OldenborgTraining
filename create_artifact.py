"""
This is the code for creating an artifact through wandb.
"""
import wandb
from fastai.vision.all import *
from argparse import ArgumentParser, Namespace
import zipfile as zf


# Not necessary if you drag a folder of images; useful if you're downloading data from online
def zip_file():
    files = zf.ZipFile("UE_Screenshots.zip", "r")
    files.extractall("UE_Screenshots")
    files.close()


def wandb_stuff(args: Namespace):
    run = wandb.init(
        # Change project and notes to be specific to the artifact you're creating
        project="Multirun-testing-4K",
        entity="arcslaboratory",
        notes="A dataset of images collected over 20 different runs for testing.",
        job_type="dataset-upload",
    )

    artifact = wandb.Artifact(
        # Name your artifact something specific about the dataset
        name="perfect-dataset-no-texture",
        type="dataset",
    )

    # Change to the folder name used to save screenshots

    artifact.add_dir(args.directory_name, name="data")

    run.log_artifact(artifact)


def main():
    argparser = ArgumentParser("Choose a directory to add the artifact to.")
    argparser.add_argument(
        "directory_name", type=str, help="Choose artifact directory."
    )
    args = argparser.parse_args()
    wandb_stuff()


if __name__ == "__main__":
    main()
