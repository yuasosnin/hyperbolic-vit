from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.retrieval import RetrievalModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.transforms.images.albumentations.transforms import get_augs_albu, get_normalisation_resize_albu

from src.hyptorch.nn import ToPoincare
from oml.distances import EucledianDistance
from src.distances import HyperbolicDistance, SphericalDistance
from src.layers import Normalize

seed_everything(1)
logger = WandbLogger(project='metric-learning')

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class DataModule(LightningDataModule):
    def __init__(self, dataset_root):
        super().__init__()
        dataset_root = Path(dataset_root)
        df = pd.read_csv(dataset_root / 'df_with_bboxes.csv')
        df_train = df.loc[df['split'].eq('train')]
        df_val = df.loc[df['split'].eq('validation')]
    
        self.train_dataset = DatasetWithLabels(
            df_train, dataset_root=dataset_root, transform=get_augs_albu(im_size=224))
        self.val_dataset = DatasetQueryGallery(
            df_val, dataset_root=dataset_root, transform=get_normalisation_resize_albu(im_size=224))
        self.batch_sampler = BalanceSampler(self.train_dataset.get_labels(), n_labels=2, n_instances=10)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_sampler=self.batch_sampler, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=2)


dataset_root = Path('/content/term-paper/data/stanford_cars/')
pl_data = DataModule(dataset_root)

distance = HyperbolicDistance(c=1.0, train_c=False)
model = nn.Sequential(
    ViTExtractor(arch='vits8', weights='vits8_dino'),
    # Normalize(),
    ToPoincare(distance.c, train_c=False)
)
model: nn.Module = torch.compile(model)

criterion = TripletLossWithMiner(distance=distance, margin=0.1, miner=HardTripletsMiner())
metric_callback = MetricValCallback(
    metric=EmbeddingMetrics(
        cmc_top_k=(1,5), 
        precision_top_k=(1,5),
        map_top_k=(1,5),
        distance=distance
    )
)
optimizer = Adam(model.parameters(), lr=1e-5)

pl_model = RetrievalModule(model, criterion, optimizer)
trainer = Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[metric_callback],
    num_sanity_val_steps=0,
    accumulate_grad_batches=10,
    accelerator='auto', 
    precision=16
)

if __name__ == '__main__':
    trainer.fit(pl_model, pl_data)
