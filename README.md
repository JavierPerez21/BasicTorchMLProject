# BasicTorchMLProject

This repository is meant to be used as a first draft for machine learning projects using PyTorch in the most organized, effective and succint form possible.

# Usage
This repository consists of four main folders:
* Architectures: It contains all the architectures used in the project. Prefarably they should all inherit from nn.module.
* Dataset generator: It contains the dataset class which inherits from torch.utils.data.Dataset and generates the datasets for the dataloaders.
* Experiment_logs: It containes folders for each experiment. In each experiment folder there are several log files and saved models for the different iterations and configurations of the experiment.
