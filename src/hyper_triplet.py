from typing import Dict, Optional, Union, Dict, List

import warnings

import torch
import torch.nn as nn
from torch import Tensor

from oml.losses.triplet import TripletLoss, TripletLossWithMiner
from oml.functional.losses import get_reduced
from oml.interfaces.criterions import ITripletLossWithMiner
from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_all_tri import AllTripletsMiner

from src.hyptorch.nn import ToPoincare


TLogs = Dict[str, float]


class HypTripletLossWithMiner(ITripletLossWithMiner):
    """
    This class combines `Miner` and `TripletLoss`.

    """

    criterion_name = "triplet"  # for better logging

    def __init__(
        self,
        distance,
        margin: Optional[float],
        miner: ITripletsMiner = AllTripletsMiner(),
        reduction: str = "mean",
        need_logs: bool = False,
    ):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            miner: A miner that implements the logic of picking triplets to pass them to the triplet loss.
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` if you want to store logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super().__init__()
        self.distance = distance
        self.projector = ToPoincare(self.distance.c)
        self.tri_loss = TripletLoss(distance=self.distance, margin=margin, reduction=reduction, need_logs=need_logs)
        self.miner = miner
        self._patch_miners_distance()
        self.reduction = reduction
        self.need_logs = need_logs

        self.last_logs: Dict[str, float] = {}

    def _patch_miners_distance(self):
        # this just reduces verbosity of providing distance to both loss and its miner
        if self.miner.distance != self.distance:
            if self.miner._distance_provided:
                warnings.warn(f"Miner was provided with distance ({type(self.miner.distance)}) which is not equal to distance for the loss ({type(self.distance)}). Are you sure?")
            else:
                self.miner._set_distance(self.distance)

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: Features with the shape ``[batch_size, feat]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        labels_list = labels2list(labels)
        features = self.projector(features)

        # if miner can produce triplets using samples outside of the batch,
        # it has to return the corresponding indicator names <is_original_tri>
        if isinstance(self.miner, TripletMinerWithMemory):
            anchor, positive, negative, is_orig_tri = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

            if self.need_logs:

                def avg_d(x1: Tensor, x2: Tensor) -> Tensor:
                    return self.distance.elementwise(x1.clone().detach(), x2.clone().detach()).mean()

                is_bank_tri = ~is_orig_tri
                active = (loss.clone().detach() > 0).float()
                self.last_logs.update(
                    {
                        "orig_active_tri": active[is_orig_tri].sum() / is_orig_tri.sum(),
                        "bank_active_tri": active[is_bank_tri].sum() / is_bank_tri.sum(),
                        "pos_dist_orig": avg_d(anchor[is_orig_tri], positive[is_orig_tri]),
                        "neg_dist_orig": avg_d(anchor[is_orig_tri], negative[is_orig_tri]),
                        "pos_dist_bank": avg_d(anchor[is_bank_tri], positive[is_bank_tri]),
                        "neg_dist_bank": avg_d(anchor[is_bank_tri], negative[is_bank_tri]),
                    }
                )

        else:
            anchor, positive, negative = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

        self.last_logs.update(self.tri_loss.last_logs)
        self.last_logs.update(getattr(self.miner, "last_logs", {}))

        return get_reduced(loss, self.reduction)
