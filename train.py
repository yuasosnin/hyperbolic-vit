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
from oml.distances import EuclideanDistance

from src.hyptorch.optim import RiemannianAdam, RiemannianAdamW
from src.data import CARS196DataModule, CUB200DataModule
from src.hyptorch import ExponentialMap, PoincareBall
from src.oml.distances import PoincareBallDistance, DotProductDistance
from src.oml.extractor import HViTExtractor
from src.hyptorch.layers import Normalize

import timm

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def behead(model):
    model.head = nn.Identity()
    return model


def get_trainer(distance, epochs=300, precision=64):
    logger = WandbLogger(project="metric-learning")
    metric_callback = MetricValCallback(
        metric=EmbeddingMetrics(cmc_top_k=(1,2,4,8), distance=distance))
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[metric_callback],
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
        accelerator="auto",
        precision=precision,
        inference_mode=False)
    return trainer


def get_data(n_labels, n_instances, dataset="cars", num_workers=0, batch_size=512):
    data_folder = "./data"
    if dataset == "cars":
        dataset_type = CARS196DataModule
    elif dataset == "cub":
        dataset_type = CUB200DataModule
    else:
        raise ValueError()
    return dataset_type(
        data_folder,
        n_labels=n_labels,
        n_instances=n_instances,
        batch_size=batch_size,
        num_workers=num_workers)


class ModelInitializer:
    def __init__(
            self,
            model_class="euclidean",
            dim=128,
            weights="vit_small_patch16_224.dino",
            margin=0.1,
            lr=1e-5,
            wd=1e-2,
            opt_name="adamw",
            miner_type="all",
            c=0.1,
            clip_factor=2.3
        ):
        self.model_class = model_class
        self.dim = dim
        self.weights = weights
        self.margin = margin
        self.lr = lr
        self.wd = wd
        self.opt_name = opt_name
        self.miner_type = miner_type
        self.c = c
        self.clip_factor = clip_factor

    @staticmethod
    def compile_if_linux(model: nn.Module) -> nn.Module:
        if platform.system() == "Linux":
            return torch.compile(model)
        return model

    @staticmethod
    def get_optimizer_type(name):
        if name == "adamw":
            return AdamW
        elif name == "radamw":
            return RiemannianAdamW
        elif name == "radam":
            return RiemannianAdam
        else:
            raise ValueError("Unknown name")

    @staticmethod
    def get_miner(name):
        if name == "all":
            return AllTripletsMiner()
        elif name == "hard":
            return HardTripletsMiner()
        else:
            raise ValueError("Unknown name")

    def init_euclidean_model(self):
        distance = DotProductDistance()
        model = nn.Sequential(
            # ViTExtractor(arch="vits16", weights=self.weights),
            behead(timm.create_model(self.weights, pretrained=True)),
            Normalize(p=2),
        )
        model = self.compile_if_linux(model)
        miner = self.get_miner(self.miner_type)
        criterion = TripletLossWithMiner(distance=distance, margin=self.margin, miner=miner)
        optimizer = self.get_optimizer_type(self.opt_name)(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return RetrievalModule(model, criterion, optimizer), distance

    def init_fully_hyperbolic_model(self):
        manifold = PoincareBall(c=self.c, clip_factor=self.clip_factor)
        distance = PoincareBallDistance(manifold)
        model = HViTExtractor(arch="hvits16", weights=self.weights, strict_load=False)
        model = self.compile_if_linux(model)
        miner = self.get_miner(self.miner_type)
        criterion = TripletLossWithMiner(distance=distance, margin=self.margin, miner=miner)
        optimizer = self.get_optimizer_type(self.opt_name)(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return RetrievalModule(model, criterion, optimizer), distance

    def init_hyperbolic_proj_model(self):
        manifold = PoincareBall(c=self.c, clip_factor=self.clip_factor)
        distance = PoincareBallDistance(manifold)
        model = nn.Sequential(
            # ViTExtractor(arch="vits16", weights=self.weights),
            behead(timm.create_model(self.weights, pretrained=True)),
            nn.Linear(384, self.dim, bias=False),
            ExponentialMap(manifold)
        )
        model = self.compile_if_linux(model)
        miner = self.get_miner(self.miner_type)
        criterion = TripletLossWithMiner(distance=distance, margin=self.margin, miner=miner)
        optimizer = self.get_optimizer_type(self.opt_name)(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return RetrievalModule(model, criterion, optimizer), distance

    def init_model(self):
        if self.model_class == "euclidean":
            return self.init_euclidean_model()
        elif self.model_class == "projection":
            return self.init_hyperbolic_proj_model()
        elif self.model_class == "full":
            return self.init_fully_hyperbolic_model()
        else:
            raise ValueError("Invalid model class")


def get_parsed_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_class', type=str, default="euclidean")
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--weights', type=str, default='vit_small_patch16_224.dino')

    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--lr', type=float, default=1e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)

    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--opt_name', type=str, default="adamw")
    parser.add_argument('--miner_type', type=str, default="all")
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--clip_factor', type=float, default=2.3)

    parser.add_argument('--n_labels', type=int, default=5)
    parser.add_argument('--n_instances', type=int, default=20)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--dataset', type=str, default="cub")

    args = parser.parse_args()
    return vars(args)


def main(
        epochs,
        batch_size,
        precision,
        model_class,
        dim,
        weights,
        margin,
        lr,
        weight_decay,
        opt_name,
        miner_type,
        c,
        clip_factor,
        n_labels,
        n_instances,
        dataset
    ):
    initializer = ModelInitializer(
        model_class=model_class,
        dim=dim,
        weights=weights,
        margin=margin,
        lr=lr,
        wd=weight_decay,
        opt_name=opt_name,
        miner_type=miner_type,
        c=c,
        clip_factor=clip_factor)
    pl_model, distance = initializer.init_model()
    pl_data = get_data(
        n_labels=n_labels, n_instances=n_instances, dataset=dataset, num_workers=0, batch_size=batch_size)
    logger = WandbLogger(project="metric-learning")
    metric_callback = MetricValCallback(
        metric=EmbeddingMetrics(cmc_top_k=(1,2,4,8), distance=distance))
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[metric_callback],
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
        accelerator="auto",
        precision=precision,
        inference_mode=False)
    logger.watch(pl_model)

    trainer.fit(
        pl_model,
        train_dataloaders=pl_data.train_dataloader(),
        val_dataloaders=pl_data.test_dataloader()
    )
    return logger


if __name__ == "__main__":
    args = get_parsed_args()
    seed_everything(1)
    logger = main(**args)
    logger.experiment.config.update(args)
