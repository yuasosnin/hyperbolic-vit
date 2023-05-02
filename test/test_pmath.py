import unittest

import torch
import geoopt.manifolds.stereographic.math as pmath
from src.hyptorch.pmath.mobius_add import mobius_add, mobius_add_norm


class TestPmathFunctions(unittest.TestCase):
    def test_mobius_add_grad(self):
        x = torch.randn(4,4,4,16, dtype=torch.double)
        y = torch.randn(4,4,4,16, dtype=torch.double)
        k = torch.tensor(-1.0, dtype=torch.double)

        x = pmath.expmap0(x, k=k)
        x.requires_grad = True
        y = pmath.expmap0(y, k=k)
        y.requires_grad = True
        k.requires_grad = True

        check = torch.autograd.gradcheck(mobius_add, (x, y, k), eps=1e-6, atol=1e-4)
        self.assertTrue(check)

    def test_mobius_add_norm_grad(self):
        x = torch.randn(1,10, dtype=torch.double)
        y = torch.randn(1,10, dtype=torch.double)
        k = torch.tensor(-1.0, dtype=torch.double)

        x = pmath.expmap0(x, k=k)
        x.requires_grad = True
        y = pmath.expmap0(y, k=k)
        y.requires_grad = True
        k.requires_grad = True

        check = torch.autograd.gradcheck(mobius_add_norm, (x, y, k), eps=1e-6, atol=1e-4)
        self.assertTrue(check)


if __name__ == "__main__":
    unittest.main()
