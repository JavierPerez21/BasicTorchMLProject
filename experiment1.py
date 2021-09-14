from Architectures.CNN import *
from Model import *
from Dataset_generator.DatasetGenerator import *
from torch.utils.data.dataloader import DataLoader
import torchvision
from  torchvision.transforms  import ToTensor
import os
import wandb
wandb.init(entity="javierpereztobia", project="ags-mnist")
wandb.watch_called = False

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 50
lr = 0.01
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD
# Set device
optimkwargs = {'lr':0.01}
model_log=None
scheduler=None
schedulerkwargs={'gamma':0.1}
epochs = 5

# wandb config
config = wandb.config
config.batch_size = batch_size
config.device = device
config.criterion = criterion
config.optimizer = optimizer
config.lr = optimkwargs['lr']
config.epochs = epochs
config.scheduler = scheduler
config.gamma = schedulerkwargs['gamma']


# Load dataset
#dataset = Dataset([(1, 2), (3, 6), (2, 4)])
train_data = torchvision.datasets.MNIST("../data", train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.MNIST("../data", train=False, download=True, transform=ToTensor())
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.5), len(test_data) - int(len(test_data)*0.5)])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Create model
experiment_name = "testing_model"
experiment_path = path = str(os.getcwd()) + "\\" + "Experiment_logs" +"\\" + experiment_name + "\\"
if not os.path.isdir(experiment_path):
    os.mkdir(experiment_path)
cnn = CNN()
model = Model(cnn, criterion, optimizer, device, experiment_path, model_log=model_log, optimkwargs=optimkwargs,
              scheduler=scheduler, schedulerkwargs=schedulerkwargs)
model.train(epochs, test_loader, val_loader)
model.validate(val_loader)



