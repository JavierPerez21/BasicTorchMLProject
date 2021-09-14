import copy
import torch
import datetime
from prettytable import PrettyTable
import os
import copy
import wanbd

class Model:
    def __init__(self, architecure, criterion, optim, device, model_dir, model_log=None, optimkwargs={'lr':0.1}, scheduler=None, schedulerkwargs={'gamma':0.1}):
        """
            Adapt epoch(), train(), validation_epoch() and validate() for required experiment.

            Params:
                model: Should be a class object that inherits the nn.Module with a properly defined forward function.
                      Always try to use model as described in other benchmark.
                criterion: any function whose gradient can be tracked will work here. nn.Loss objects work better
                optimizer: nn.optim.optimizer object
                scheduler: nn.optim.scheduler object
                device: 'cpu' or 'cuda'
                optimkwargs: dictionary of optimizer kwargs
                schedulerkwargs: dictionary of scheduler kwargs
        """
        self.device = device
        self.model = architecure.to(self.device)
        self.criterion = criterion
        self.optimizer = optim(self.model.parameters(), **optimkwargs)
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **schedulerkwargs)
        else:
            self.scheduler = None
        self.model_dir = model_dir
        if model_log:
            self.log_path = self.model_dir + model_log + ".txt"
            self.model_path = self.model_dir + model_log + ".pth"
            self.model.load_state_dict(torch.load(self.model_path))
            with open(self.log_path, 'a') as fp:
                fp.write(f"\n")
                fp.write(f"\n")
                fp.write(f"\n")
                new_log = "New log: " + str(datetime.datetime.now())
                fp.write(f"{new_log}\n")
        else:
            model_log = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:19]
            self.log_path = self.model_dir + model_log + ".txt"
            self.model_path = self.model_dir + model_log + ".pth"
            with open(self.log_path, 'a') as fp:
                new_log = "New log: " + str(datetime.datetime.now())
                fp.write(f"{new_log}\n")
                pass
        self.best_model = copy.deepcopy(self.model)

    def logit_transformation(self, yp):
        return yp.max(dim=1)[1]

    def epoch(self, loader, train=False):
        total_loss, total_err = 0., 0.
        for x,y in loader:
            x,y = x.to(self.device), y.to(self.device)
            yp = self.model(x)
            try:
                #loss = self.criterion(yp, y)
                loss = torch.nn.CrossEntropyLoss()(yp, y)
            except RuntimeError:
                print(yp.shape, y.shape)
                loss = torch.nn.CrossEntropyLoss()(yp, y)
                #loss = self.criterion(yp, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            yp = self.logit_transformation(yp)
            total_err += (yp != y).sum().item()
            total_loss += loss.item() * x.shape[0]
            torch.cuda.empty_cache()
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def train(self, epochs, train_loader, test_loader):
        acc = 0
        train_info = ['Training_with']
        train_info.append(str(self.model))
        train_info.append("Epochs: {}".format(epochs))
        train_info.append("Loss: {}".format(self.criterion))
        train_info.append(self.optimizer)
        train_info.append(self.scheduler)
        train_info.append("Extra info:")
        with open(self.log_path, 'a') as fp:
            for item in train_info:
                print(item)
                fp.write(f"{item}\n")
            pass
        for epoch in range(0, epochs):
          now = datetime.datetime.now().timestamp()
          train_err, train_loss = self.epoch(train_loader, train=True)
          test_err, test_loss = self.epoch(test_loader)
          wandb.log({
              "Train error": train_err,
              "Train loss": train_loss,
              "Test error": test_err,
              "Test loss": test_loss
          })
          if self.scheduler:
              self.scheduler.step()
          torch.cuda.empty_cache()
          after = datetime.datetime.now().timestamp()
          new_acc = 1 - test_err
          if new_acc > acc:
              torch.save(self.model.state_dict(), self.model_path)
              acc = new_acc
          epoch_info = ["{}  Train_accuracy: {:.6f} Train_loss:  {:.6f}  Test_accuracy: {:.6f}  Test_loss: {:.6f}  Time: {:.2f} s".format(epoch, 1-train_err, train_loss, 1-test_err, test_loss, after-now)]
          epoch_info.append("   Model saved at: {}".format(self.model_path))
          with open(self.log_path, 'a') as fp:
              for item in [epoch_info]:
                  print(item)
                  fp.write(f"{item}\n")
              pass


    def validation_epoch(self, loader):
        total_err = 0.
        for x, y in loader:
            x,y = x.to(self.device), y.to(self.device)
            yp = self.model(x)
            yp = self.logit_transformation(yp)
            total_err += (yp != y).sum().item()
        return total_err / len(loader.dataset)

    def validate(self, val_loader):
        now = datetime.datetime.now().timestamp()
        test_err = self.validation_epoch(val_loader)
        after = datetime.datetime.now().timestamp()
        val_epoch_info = "    Acc: {} in {} s".format(round((1-test_err), 4), after-now)
        val_info = ['\n']
        val_info.append('Validating_with: ')
        val_info.append(str(self.model))
        val_info.append("Model loaded from: {}".format(self.model_path))
        val_info.append(val_epoch_info)
        val_info.append("Extra info: ")
        with open(self.log_path, 'a') as fp:
            for item in val_info:
                print(item)
                fp.write(f"{item}\n")
            pass

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params