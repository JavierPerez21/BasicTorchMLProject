import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def generate_samples(self):
        # Populate sam
        samples = []
        for x, y in self.inputs:
            x = self.x_transform(x)
            y = self.y_transform(y)
            samples.append((x, y))
        return samples

    def x_transform(self,x):
        # Do transformation
        x = x*2-1
        return x

    def y_transform(self,y):
        # Do transformation
        y = y*2-1
        return y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]