from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from oml.lightning.modules.retrieval import RetrievalModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.distances import EucledianDistance

from src.data import DataModule
from src.hyptorch.nn import ToPoincare
from src.distances import HyperbolicDistance, SphericalDistance
from src.layers import Normalize
from src.geoopt_plugin import ManifoldDistance, ManifoldProjection
from geoopt import PoincareBall, Lorentz

seed_everything(1)
logger = WandbLogger(project='metric-learning')

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


dataset_root = Path('/content/term-paper/data/stanford_cars/')
pl_data = DataModule(dataset_root)

manifold = PoincareBall(c=1.0, learnable=False)
# distance = HyperbolicDistance(c=1.0, train_c=False)
distance = ManifoldDistance(manifold)
model = nn.Sequential(
    ViTExtractor(arch='vits8', weights='vits8_dino'),
    # Normalize(),
    # ToPoincare(distance.c, train_c=False)
    ManifoldProjection(manifold)
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
