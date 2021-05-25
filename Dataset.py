import torch
from torch.utils.data import DataLoader, Dataset

class DatasetRandom(Dataset):

    def __init__(self, size = 10000, shape = (20,20) ):
        """This generates n copies of 2D array in shape specified

        Args:
            size (tuple) : number of copies, in even number
            shape (tuple): shape of each copy of 2D array

        Returns:
            None
        """
 
        shape = (size//2,)+ shape 
        x1 = torch.normal(0, 2, size = shape) # y = 0
        x2 = torch.normal(0.2, 2, size = shape) # y = 1
        self.X = torch.cat([x1, x2], 0)

        y1 = torch.zeros(size//2)
        y2 = torch.ones(size//2)
        self.Y = torch.cat([y1, y2])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x, y