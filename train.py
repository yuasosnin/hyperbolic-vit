import platform

import torch
import torch.nn as nn
from torch.optim import AdamW
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

from geoopt.optim import RiemannianAdam

from src.data import CARS196DataModule
from src.hyptorch import ExponentialMap, PoincareBall
from src.oml.distances import PoincareBallDistance
from src.oml.extractor import HViTExtractor

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def get_trainer(epochs, distance):
    logger = WandbLogger(project="metric-learning")
    metric_callback = MetricValCallback(
        metric=EmbeddingMetrics(
            cmc_top_k=(1,5),
            precision_top_k=(1,5),
            map_top_k=(1,5),
            distance=distance))
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[metric_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=10,
        accelerator="auto",
        precision=16,
        inference_mode=False)
    return trainer

def get_data(n_labels, n_instances, num_workers=0, batch_size=16):
    data_folder = "./data"
    return CARS196DataModule(
        data_folder,
        n_labels=n_labels,
        n_instances=n_instances,
        batch_size=batch_size,
        num_workers=num_workers)

def get_model():
    manifold = PoincareBall(c=1.0, clip_factor=None)
    distance = PoincareBallDistance(manifold)
    # model = nn.Sequential(
    #     ViTExtractor(arch="vits16", weights="vits16_dino"),
    #     nn.Linear(384, 64, bias=False),
    #     ExponentialMap(manifold)
    # )
    model = HViTExtractor(arch="hvits16", weights="vits16_dino", strict_load=False)
    if platform.system() == "Linux":
        model: nn.Module = torch.compile(model)
    criterion = TripletLossWithMiner(distance=distance, margin=0.1, miner=AllTripletsMiner())
    optimizer = RiemannianAdam(model.parameters(), lr=1e-5)

    return RetrievalModule(model, criterion, optimizer), distance

if __name__ == "__main__":
    seed_everything(1)

    pl_data = get_data(n_labels=5, n_instances=20, num_workers=0, batch_size=256)
    pl_model, distance = get_model()
    trainer = get_trainer(epochs=100, distance=distance)

    trainer.fit(pl_model, pl_data)
    trainer.validate(pl_model, pl_data.test_dataloader())
