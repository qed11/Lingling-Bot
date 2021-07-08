import torch

class auto_dataset(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.train = path

