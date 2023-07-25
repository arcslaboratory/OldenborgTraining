"""
This is the code for creating an artifact through wandb.
"""
import wandb
from fastai.vision.all import *
from argparse import ArgumentParser, Namespace
import zipfile as zf


def zip_file():
    files = zf.ZipFile("UE_Screenshots.zip", "r")
    files.extractall("UE_Screenshots")
    files.close()


def artifact_setup(args: Namespace):
    run = wandb.init(
        # Change project and notes to be specific to the artifact you're creating; follow project name format.
        # Example: "randomized_hidden_perfect"
        project="env_visibility_path",
        entity="arcslaboratory",
        notes="A dataset of images collected over 20 different runs for testing.",
        job_type="dataset-upload",
    )

    artifact = wandb.Artifact(
        # Name your artifact something specific about the dataset; follow artifact name format
        # Example name: "07-21_20_4K"
        name="date_num-trials_num-images",
        type="dataset",
    )

    artifact.add_dir(args.directory_name, name="data")

    run.log_artifact(artifact)


def main():
    """Argument Parser for choosing artifact directory path."""
    argparser = ArgumentParser("Choose a directory to add the artifact to.")
    argparser.add_argument(
        "directory_name", type=str, help="Choose artifact directory."
    )
    args = argparser.parse_args()
    artifact_setup(args)


if __name__ == "__main__":
    main()
