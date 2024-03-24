from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from .nf_utils import Flow


class Affine(Flow):
    """Affine transformation y = e^a * x + b.

    Args:
        dim (int): dimension of input/output data. int
    """

    def __init__(self, dim: int = 2):
        """Create and init an affine transformation."""
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(self.dim))  # a
        self.shift = nn.Parameter(torch.zeros(self.dim))  # b

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward transformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        
        import torch
        a=self.log_scale
        b=self.shift
        y = torch.exp(a) * x + b
        log_det_jac = torch.sum(a).repeat(B) #(sum on columns)

        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        B, D = y.shape

        ##########################################################
        import torch
        a=self.log_scale
        b=self.shift
        x = (y - b) * torch.exp(-a)
        inv_log_det_jac = -torch.sum(self.log_scale).repeat(B)

        ##########################################################

        assert x.shape == (B, D)
        assert inv_log_det_jac.shape == (B,)

        return x, inv_log_det_jac
