from Architectures.ResNet9 import *
from Model import *
from Dataset_generator.DatasetGenerator import *
from torch.utils.data.dataloader import DataLoader
import torchvision
from  torchvision.transforms  import ToTensor
import os

# Settings
config = {
    'batch_size': 200,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': torch.optim.Adam,
    'optimkwargs': {},
    'scheduler': None,
    'schedulerkwargs': {},
    'epochs': 20,
    'in_channels': 3,
    'num_classes': 10,
    'experiment_name': "nominal-mnist",
    'pretrained_model': str(os.getcwd()) + "/Experiment_logs/nominal-cifar10/nominal-cifar10_2021-09-16_13-08-26.pth",
    # These params are used for adversarial training
    'attack': torchattacks.FGSM,
    'thres': 0.5,
    'attackkwargs': {'eps':4/255}
}
config['model_name'] = config['experiment_name'] + "_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:19]
config['experiment_path'] = str(os.getcwd()) + "/Experiment_logs" + "/" + config['experiment_name'] + "/"
if not os.path.isdir(config['experiment_path']):
    os.mkdir(config['experiment_path'])


# Load dataset
train_data = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=ToTensor())
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.5), len(test_data) - int(len(test_data)*0.5)])
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)


resnet9 = ResNet9(in_channels=config['in_channels'], num_classes=config['num_classes'])
classifier = Model(resnet9, config)


#classifier.train(train_loader, test_loader)


fgsm = torchattacks.FGSM(copy.deepcopy(classifier.model), eps=8/255)
pgd_linf = torchattacks.PGD(copy.deepcopy(classifier.model), eps=8/255, alpha=0.1, steps=50, random_start=False)
autoattack = torchattacks.AutoAttack(copy.deepcopy(classifier.model), norm='Linf', eps=8/255, version='standard', n_classes=config['num_classes'])
list_of_attacks = [None, fgsm, pgd_linf, autoattack]
classifier.validate(val_loader, list_of_attacks=list_of_attacks)