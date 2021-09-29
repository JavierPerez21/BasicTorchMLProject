# BasicTorchMLProject

This repository is meant to be used as a first draft for machine learning projects using PyTorch in the most organized, effective and succint form possible.

# Usage
This repository consists of three main folders:
* Architectures: It contains all the architectures used in the project. Prefarably they should all inherit from nn.module.
* Dataset generator: It contains the dataset class which inherits from torch.utils.data.Dataset and generates the datasets for the dataloaders.
* Experiment_logs: It containes folders for each experiment. In each experiment folder there are several log files and saved models for the different iterations and configurations of the experiment.

The Model class brings everything together. Its inputs are: _architecture_, which should be a model that inherits from nn.module, and _config_, which should be a dictionary with all the required parameters for the model: optimizer, scheduler, number of steps, device, pretrained_model and others.

Once the model class is created, it can be trained using the .train() method.

Once training is completed, the performance of the model can be evaluated using the .evaluate() method.

Logs from training and evaluation are saved automatically in Experiment_logs.
