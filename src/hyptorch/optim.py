import torch
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter, ManifoldTensor


class RiemannianAdamW(RiemannianAdam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    # grad.add_(point, alpha=weight_decay)  # Instead of adding it here, we add it in the end
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # Add decayed weights as per AdamW
                    update = -learning_rate * (direction + weight_decay * point)
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, update, exp_avg
                    )
                    # use copy only for user facing point
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                    if (
                        group["stabilize"] is not None
                        and state["step"] % group["stabilize"] == 0
                    ):
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
        return loss
