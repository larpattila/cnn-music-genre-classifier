
from typing import Tuple, Optional, Union, Dict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from utils import get_audio_files_from_dir

import numpy as np
import pickle

class MyDataset(Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 data_parent_dir: Optional[str] = 'Data/',
                 key_features: Optional[str] = 'features',
                 key_class: Optional[str] = 'labels',
                 load_into_memory: Optional[bool] = True) \
            -> None:

        super().__init__()
        data_path = Path(data_parent_dir, data_dir)
        self.files = get_audio_files_from_dir(data_path)
        self.load_into_memory = load_into_memory
        self.key_features = key_features
        self.key_class = key_class

        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)

    @staticmethod
    def _load_file(file_path: Path) -> Tuple[int, np.ndarray]:
        with file_path.open('rb') as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> Tuple[int, np.ndarray]:
        if self.load_into_memory:
            the_item = self.files[item]
        else:
            the_item = self._load_file(self.files[item])
        return the_item[1], the_item[0]


def get_dataset(data_dir: Union[str, Path],
                data_parent_dir: Optional[str] = '',
                key_features: Optional[str] = 'features',
                key_class: Optional[str] = 'labels',
                load_into_memory: Optional[bool] = True) \
        -> MyDataset:
    return MyDataset(data_dir=data_dir,
                     data_parent_dir=data_parent_dir,
                     key_features=key_features,
                     key_class=key_class,
                     load_into_memory=load_into_memory)


def get_data_loader(dataset: MyDataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool) \
        -> DataLoader:

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)