from torch.utils.data import Dataset
from typing import Optional, Callable

class BaseDataset(Dataset):
    def __init__(self, root: str, split: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, **kwargs):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        raise NotImplementedError