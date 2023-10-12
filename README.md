# OldenborgModel

Train and perform inference on Oldenborg datasets.
Create one Project in WandB that contains Datasets, models, and inference testing

## Windows

For inference on Windows, I had to create an environment with the following:

~~~bash
conda create --name oldenborg
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
mamba install fastai
~~~
## Mac
On mac just using conda: 
conda create --name oldenborg
conda activate oldenborg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
conda install -c fastai fastai 
conda install -c conda-forge wandb

In script, Must comment out: 
with set_posix_windows(): 
def set_posix_windows():
     posix_backup = pathlib.PosixPath
     try:
         pathlib.PosixPath = pathlib.WindowsPath
         yield
     finally:
         pathlib.PosixPath = posix_backup

