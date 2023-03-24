from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from oml.interfaces.miners import ITripletsMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.losses.triplet import TripletLoss, TripletLossWithMiner

from src.hyptorch.nn import ToPoincare


TLogs = Dict[str, float]


class HypTripletLossWithMiner(TripletLossWithMiner):
    """
    This class combines `Miner` and `TripletLoss`.

    """

    criterion_name = "hyper_triplet"  # for better logging

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
        self.tri_loss = TripletLoss(self.distance, margin=margin, reduction=reduction, need_logs=need_logs)
        self.miner = miner
        self._patch_miners_distance()
        self.reduction = reduction
        self.need_logs = need_logs

        self.last_logs: Dict[str, float] = {}
