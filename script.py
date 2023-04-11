from pathlib import Path

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

from src.data import CARS196DataModule
from src.distances import PoincareBallDistance, LorentzDistance
from src.layers import Normalize, PoincareBallProjection, LorentzProjection

seed_everything(1)
logger = WandbLogger(project="metric-learning")

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


data_folder = "/content/term-paper/data"
pl_data = CARS196DataModule(data_folder)

distance = PoincareBallDistance(c=1.0, train_c=False)
model = nn.Sequential(
    ViTExtractor(arch="vits8", weights="vits8_dino"),
    PoincareBallProjection(distance.c, clip_r=1.0)
)
model: nn.Module = torch.compile(model)

criterion = TripletLossWithMiner(distance=distance, margin=0.1, miner=AllTripletsMiner())
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
    accelerator="auto", 
    precision=16,
    inference_mode=False,
)

if __name__ == "__main__":
    trainer.fit(pl_model, pl_data)
    trainer.validate(pl_model, pl_data.test_dataloader())
