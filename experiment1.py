from Architectures.CNN import *
from Model import *
from Dataset_generator.DatasetGenerator import *
from torch.utils.data.dataloader import DataLoader
import torchvision
from  torchvision.transforms  import ToTensor
import os

batch_size = 50
lr = 0.01
epochs = 20

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset
#dataset = Dataset([(1, 2), (3, 6), (2, 4)])
train_data = torchvision.datasets.MNIST("../data", train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.MNIST("../data", train=False, download=True, transform=ToTensor())
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.5), len(test_data) - int(len(test_data)*0.5)])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

# Create model
experiment_name = "testing_model"
experiment_path = path = str(os.getcwd()) + "\\" + "Experiment_logs" +"\\" + experiment_name + "\\"
if not os.path.isdir(experiment_path):
    os.mkdir(experiment_path)
cnn = CNN()
model = Model(cnn, nn.CrossEntropyLoss, torch.optim.SGD, device, experiment_path, optimkwargs={'lr':lr})
model.train(5, test_loader, val_loader)
model.validate(val_loader)



