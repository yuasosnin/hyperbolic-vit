from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from oml.samplers.balance import BalanceSampler
from oml.transforms.images.albumentations.transforms import get_augs_albu, get_normalisation_resize_albu
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels

from .datasets_converters import build_cars196_df, build_cub_df
from .datasets_downloaders import download_cars, download_cub


class MetricLearningDataModule(LightningDataModule, ABC):
    def __init__(
            self, dataset_root, num_classes, n_labels=2, n_instances=10, batch_size=128, num_workers=2):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        df = self.build_data_frame()
        df_train, df_val, df_test = self._split_train_test(df, method="by_class")

        self.train_dataset = DatasetWithLabels(
            df_train, dataset_root=dataset_root, transform=get_augs_albu(im_size=224))
        self.val_dataset = DatasetQueryGallery(
            df_val, dataset_root=dataset_root, transform=get_normalisation_resize_albu(im_size=224))
        self.test_dataset = DatasetQueryGallery(
            df_test, dataset_root=dataset_root, transform=get_normalisation_resize_albu(im_size=224))
        self.batch_sampler = BalanceSampler(
            self.train_dataset.get_labels(), n_labels=n_labels, n_instances=n_instances)
    
    @abstractmethod
    def build_data_frame(self):
        raise NotImplementedError()
        
    def _split_train_test(self, df, method="default"):
        if method == "default":
            df_train = df.loc[df['split'] == 'train']
            df_test = df.loc[df['split'] == 'validation']
            df_train, df_val = train_test_split(df_train, test_size=0.1)
            return df_train, df_val, df_test
        elif method == "by_class":
            train_val_idx = self.num_classes // 2
            train_idx = train_val_idx - round(train_val_idx * 0.1)
            df_train = df.loc[df["label"] <= train_idx]
            df_val = df.loc[(train_idx < df["label"]) & (df["label"] <= train_val_idx)]
            df_test = df.loc[df["label"] > train_val_idx]
            return df_train, df_val, df_test
        else:
            raise ValueError()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
    

class CARS196DataModule(MetricLearningDataModule):
    def __init__(
        self, datsaet_root, n_labels=2, n_instances=10, batch_size=128, num_workers=2
        ):
        super().__init__(
            datsaet_root, 
            num_classes=196, 
            n_labels=n_labels, 
            n_instances=n_instances, 
            batch_size=batch_size, 
            num_workers=num_workers)

    def build_data_frame(self):
        if not (self.dataset_root / "devkit").exists():
            print("Downloading dataset...")
            download_cars(self.dataset_root.parents[0])
            print("Datset downloaded")
        return build_cars196_df(self.dataset_root)


class CUB200DataModule(LightningDataModule):
    def __init__(
        self, datsaet_root, n_labels=2, n_instances=10, batch_size=128, num_workers=2
        ):
        super().__init__(
            datsaet_root, 
            num_classes=200, 
            n_labels=n_labels, 
            n_instances=n_instances, 
            batch_size=batch_size, 
            num_workers=num_workers)

    def build_data_frame(self):
        if not (self.dataset_root / "images").exists():
            print("Downloading dataset...")
            download_cub(self.dataset_root.parents[0])
            print("Datset downloaded")
        return build_cub_df(self.dataset_root)
