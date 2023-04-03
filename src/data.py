
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from oml.samplers.balance import BalanceSampler
from oml.transforms.images.albumentations.transforms import get_augs_albu, get_normalisation_resize_albu
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels



class DataModule(LightningDataModule):
    def __init__(self, dataset_root, n_labels=2, n_instances=10, batch_size=128, num_workers=2):
        super().__init__()
        dataset_root = Path(dataset_root)

        self.batch_size = batch_size
        self.num_workers = num_workers

        df = pd.read_csv(dataset_root / 'df_with_bboxes.csv')
        df_train = df.loc[df['split'].eq('train')]
        df_val = df.loc[df['split'].eq('validation')]
    
        self.train_dataset = DatasetWithLabels(
            df_train, dataset_root=dataset_root, transform=get_augs_albu(im_size=224))
        self.val_dataset = DatasetQueryGallery(
            df_val, dataset_root=dataset_root, transform=get_normalisation_resize_albu(im_size=224))
        self.batch_sampler = BalanceSampler(
            self.train_dataset.get_labels(), n_labels=n_labels, n_instances=n_instances)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
