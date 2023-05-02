import torch
from torch import Tensor
torch.set_printoptions(sci_mode=False)


@torch.jit.script
def _mobius_add(x: Tensor, y: Tensor, k: Tensor) -> Tensor:
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    alpha = 1 - 2 * k * xy - k * y2
    beta = 1 + k * x2
    denom = (1 - 2 * k * xy + k**2 * x2 * y2).clamp_min(1e-15)

    return (alpha * x + beta * y) / denom


@torch.jit.script
def outer(x, y):
    return x.unsqueeze(-1) @ y.unsqueeze(-2)


@torch.jit.script
def _mobius_add_grad(x: Tensor, y: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    d = x.shape[-1]
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    alpha = 1 - 2 * k * xy - k * y2
    beta = 1 + k * x2
    denom = (1 - 2 * k * xy + k**2 * x2 * y2).clamp_min(1e-15)
    d_prime_k = -2 * xy + 2 * k * x2 * y2
    x_plus_y = (alpha * x + beta * y) / denom

    dfdx = (
        - (2 * k / denom).unsqueeze(-1) * (outer(x, y) - outer(y, x))
        + (2 * k / denom).unsqueeze(-1) * outer(x_plus_y, y)
        - (2 * k**2 * y2 / denom).unsqueeze(-1) * outer(x_plus_y, x)
        + (alpha / denom).unsqueeze(-1) * torch.eye(d, device=x.device)
    )
    dfdy = (
        - (2 * k / denom).unsqueeze(-1) * (outer(x, x) + outer(x, y))
        + (2 * k / denom).unsqueeze(-1) * outer(x_plus_y, x)
        - (2 * k**2 * x2 / denom).unsqueeze(-1) * outer(x_plus_y, y)
        + (beta / denom).unsqueeze(-1) * torch.eye(d, device=x.device)
    )
    dfdk = (-2 * xy * x - y2 * x + x2 * y - d_prime_k * x_plus_y) / denom

    return dfdx, dfdy, dfdk


class MobiusAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k):
        ctx.save_for_backward(x, y, k)
        return _mobius_add(x, y, k)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, k = ctx.saved_tensors
        dfdx, dfdy, dfdk = _mobius_add_grad(x, y, k)
        return (
            (dfdx.transpose(-2, -1) @ grad_output.unsqueeze(-1)).squeeze(-1),
            (dfdy.transpose(-2, -1) @ grad_output.unsqueeze(-1)).squeeze(-1),
            (dfdk.unsqueeze(-2) @ grad_output.unsqueeze(-1)).squeeze(-1)
        )


def mobius_add(x: Tensor, y: Tensor, k: Tensor) -> Tensor:
    return MobiusAdd.apply(x, y, k)


@torch.jit.script
def _mobius_add_norm(x: Tensor, y: Tensor, k: Tensor, keepdim: bool = False) -> Tensor:
    x2 = x.pow(2).sum(dim=-1, keepdim=keepdim)
    y2 = y.pow(2).sum(dim=-1, keepdim=keepdim)
    xy = (x * y).sum(dim=-1, keepdim=keepdim)

    alpha = 1 - 2 * k * xy - k * y2
    beta = 1 + k * x2
    denom = (1 - 2 * k * xy + k**2 * x2 * y2).clamp_min(1e-15)

    return torch.sqrt(alpha**2 * x2 + 2 * alpha * beta * xy + beta**2 * y2) / denom.abs()


@torch.jit.script
def _mobius_add_norm_grad(x, y, k) -> tuple[Tensor, Tensor, Tensor]:
    # we don't differentiate forward but rather use chain rule for norm
    # which is probably what causes CUDA out of memory
    add = _mobius_add(x, y, k)
    norm = _mobius_add_norm(x, y, k).unsqueeze(-1)
    d_add_d_norm = (add / norm).unsqueeze(-2)
    d_add_dx, d_add_dy, d_add_dk = _mobius_add_grad(x, y, k)
    return (
        (d_add_d_norm @ d_add_dx).squeeze(-2),
        (d_add_d_norm @ d_add_dy).squeeze(-2),
        (d_add_d_norm @ d_add_dk.unsqueeze(-1)).squeeze(-1)
    )


class MobiusAddNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k):
        ctx.save_for_backward(x, y, k)
        return _mobius_add_norm(x, y, k)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, k = ctx.saved_tensors
        grad_output = grad_output.unsqueeze(-1)
        dfdx, dfdy, dfdk = _mobius_add_norm_grad(x, y, k)
        return dfdx * grad_output, dfdy * grad_output, dfdk * grad_output


def mobius_add_norm(x, y, k):
    return MobiusAddNorm.apply(x, y, k)
