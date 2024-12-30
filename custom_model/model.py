'''
In this script, we customize the Neural Spline Model of the zuko package to add Dropout.
We do so by introducing the relevant classes of the zuko package and modifying the necessary lines. 
For that we import all other necessary classes and functions from the zuko package. The classes that can not be imported,
are copied from the github repository to the helper.py file.

!! Disclaimer !!: I don't claim any of the content in this or the helper.py file as my work. It has been 
copied from the zuko package [https://github.com/probabilists/zuko/tree/f0db6310c8cd5f63d3552d04bda7ea2008da6dcc] 
and only slightly modified by myself.

All credits fo to François Rozet.
'''

## Import the necessary libraries and classes from zuko
import torch
import torch.nn as nn

import math

from torch import BoolTensor, Tensor, LongTensor, Size
from typing import Callable, Iterable, Sequence, Union, Tuple
from functools import partial
from math import ceil, prod
from torch.distributions import Transform, constraints

from zuko.transforms import MonotonicRQSTransform, AutoregressiveTransform, DependentTransform, MonotonicAffineTransform
## since these classes can not be imported from zuko, they are provided in the helper file
from .helper import LazyTransform, UnconditionalDistribution, Flow, Residual
from zuko.distributions import DiagNormal
from zuko.utils import broadcast, unpack
from zuko.nn import MaskedLinear, MLP
import torch.nn.functional as F

class ElementWiseTransform(LazyTransform):
    r"""Creates a lazy element-wise transformation.

    Arguments:
        features: The number of features.
        context: The number of context features.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = ElementWiseTransform(3, 4)
        >>> t
        ElementWiseTransform(
          (base): MonotonicAffineTransform()
          (hyper): MLP(
            (0): Linear(in_features=4, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=6, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 0.0303,  0.3644, -1.1831])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 0.0303,  0.3644, -1.1831], grad_fn=<DivBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super().__init__()

        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, features * self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(features, *s) for s in shapes)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        return "\n".join([
            f"(base): {base}",
        ])

    def forward(self, c: Tensor = None) -> Transform:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = phi.unflatten(-1, (-1, self.total))
            phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

class MaskedMLP(nn.Sequential):
    r"""Creates a masked multi-layer perceptron (MaskedMLP).

    The resulting MLP is a transformation :math:`y = f(x)` whose Jacobian entries
    :math:`\frac{\partial y_i}{\partial x_j}` are null if :math:`A_{ij} = 0`.

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        hidden_features: The numbers of hidden features.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.
        residual: Whether to use residual blocks or not.

    Example:
        >>> adjacency = torch.randn(4, 3) < 0
        >>> adjacency
        tensor([[False,  True,  True],
                [False,  True,  True],
                [False, False,  True],
                [ True,  True, False]])
        >>> net = MaskedMLP(adjacency, [16, 32], activation=nn.ELU)
        >>> net
        MaskedMLP(
          (0): MaskedLinear(in_features=3, out_features=16, bias=True)
          (1): ELU(alpha=1.0)
          (2): MaskedLinear(in_features=16, out_features=32, bias=True)
          (3): ELU(alpha=1.0)
          (4): MaskedLinear(in_features=32, out_features=4, bias=True)
        )
        >>> x = torch.randn(3)
        >>> torch.autograd.functional.jacobian(net, x)
        tensor([[ 0.0000, -0.0065,  0.1158],
                [ 0.0000, -0.0089,  0.0072],
                [ 0.0000,  0.0000,  0.0089],
                [-0.0146, -0.0128,  0.0000]])
    """

    def __init__(
        self,
        adjacency: BoolTensor,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = None,
        residual: bool = False,
        p_dropout: float = 0.0
    ):
        out_features, in_features = adjacency.shape

        if activation is None:
            activation = nn.ReLU

        # Merge outputs with the same dependencies
        adjacency, inverse = torch.unique(adjacency, dim=0, return_inverse=True)

        # P_ij = 1 if A_ik = 1 for all k such that A_jk = 1
        precedence = adjacency.int() @ adjacency.int().t() == adjacency.sum(dim=-1)

        # Layers
        layers = []

        for i, features in enumerate((*hidden_features, out_features)):
            if i > 0:
                mask = precedence[:, indices]  # noqa: F821
            else:
                mask = adjacency

            if (~mask).all():
                raise ValueError("The adjacency matrix leads to a null Jacobian.")

            if i < len(hidden_features):
                reachable = mask.sum(dim=-1).nonzero().squeeze(dim=-1)
                indices = reachable[torch.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[inverse]

            layers.append(MaskedLinear(adjacency=mask))
            layers.append(nn.Dropout(p=p_dropout))  ### add dropout here.

            if residual:
                if 0 < i < len(hidden_features) and mask.shape[0] == mask.shape[1]:
                    layers.pop()

                mask = precedence[indices, :][:, indices]

                layers.append(
                    Residual(
                        MaskedLinear(adjacency=mask),
                        activation(),
                        nn.Dropout(p=0.05),
                        MaskedLinear(adjacency=mask),
                    )
                )
            else:
                layers.append(activation())

        layers.pop()

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features

class MaskedAutoregressiveTransform(LazyTransform):
    r"""Creates a lazy masked autoregressive transformation.

    See also:
        :class:`zuko.transforms.AutoregressiveTransform`

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        passes: The number of sequential passes for the inverse transformation. If
            :py:`None`, use the number of features instead, making the transformation
            fully autoregressive. Coupling corresponds to :py:`passes=2`.
        order: A feature ordering. If :py:`None`, use :py:`range(features)` instead.
        adjacency: An adjacency matrix describing the transformation graph. If
            `adjacency` is provided, `order` is ignored and `passes` is replaced by the
            diameter of the graph.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MaskedMLP`.

    Example:
        >>> t = MaskedAutoregressiveTransform(3, 4)
        >>> t
        MaskedAutoregressiveTransform(
          (base): MonotonicAffineTransform()
          (order): [0, 1, 2]
          (hyper): MaskedMLP(
            (0): MaskedLinear(in_features=7, out_features=64, bias=True)
            (1): ReLU()
            (2): MaskedLinear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): MaskedLinear(in_features=64, out_features=6, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 1.7428, -1.6483, -0.9920])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 1.7428, -1.6483, -0.9920], grad_fn=<DivBackward0>)
    """

    def __new__(
        cls,
        features: int = None,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        adjacency: BoolTensor = None,
        *args,
        **kwargs,
    ) -> LazyTransform:
        if features is None or features > 1:
            return super().__new__(cls)
        else:
            return ElementWiseTransform(features, context, *args, **kwargs)

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        adjacency: BoolTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super().__init__()

        # Univariate transformation
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        # Adjacency
        self.register_buffer("order", None)

        if adjacency is None:
            if passes is None:
                passes = features

            if order is None:
                order = torch.arange(features)
            else:
                order = torch.as_tensor(order, dtype=int)

            assert order.ndim == 1, "'order' should be a vector."
            assert order.shape[0] == features, f"'order' should have {features} elements."

            self.passes = min(max(passes, 1), features)
            self.order = torch.div(order, ceil(features / self.passes), rounding_mode="floor")

            adjacency = self.order[:, None] > self.order
        else:
            adjacency = torch.as_tensor(adjacency, dtype=bool)

            assert adjacency.ndim == 2, "'adjacency' should be a matrix."
            assert adjacency.shape[0] == features, f"'adjacency' should have {features} rows."
            assert adjacency.shape[1] == features, f"'adjacency' should have {features} columns."
            assert adjacency.diag().all(), "'adjacency' should have ones on the diagonal."

            adjacency = adjacency * ~torch.eye(features, dtype=bool)

            self.passes = self._dag_diameter(adjacency)

        if context > 0:
            adjacency = torch.cat(
                (adjacency, torch.ones((features, context), dtype=bool)),
                dim=1,
            )

        adjacency = torch.repeat_interleave(adjacency, repeats=self.total, dim=0)

        # Hyper network
        self.hyper = MaskedMLP(adjacency, **kwargs)

    @staticmethod
    def _dag_diameter(adjacency: BoolTensor) -> int:
        r"""Returns the diameter of a directed acyclic graph.

        If the graph contains cycles, this function raises an error.

        Credits:
            This code is adapted from :func:`networkx.topological_generations`.

        Arguments:
            adjacency: An adjacency matrix representing a directed graph.

        Returns:
            The diameter of the graph.
        """

        all_generations = []
        indegree = adjacency.sum(dim=1).tolist()
        zero_indegree = [n for n, d in enumerate(indegree) if d == 0]
        while zero_indegree:
            this_generation, zero_indegree = zero_indegree, []
            for node in this_generation:
                for child in adjacency[:, node].nonzero():
                    child = child.item()
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        zero_indegree.append(child)
            all_generations.append(this_generation)

        assert all(d == 0 for d in indegree), "The graph contains cycles."

        return len(all_generations)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        if self.order is None:
            return "\n".join([
                f"(base): {base}",
                f"(passes): {self.passes}",
            ])
        else:
            order = self.order.tolist()

            if len(order) > 10:
                order = order[:5] + [...] + order[-5:]
                order = str(order).replace("Ellipsis", "...")

            return "\n".join([
                f"(base): {base}",
                f"(order): {order}",
            ])

    def meta(self, c: Tensor, x: Tensor) -> Transform:
        if c is not None:
            x = torch.cat(broadcast(x, c, ignore=1), dim=-1)

        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

    def forward(self, c: Tensor = None) -> Transform:
        return AutoregressiveTransform(partial(self.meta, c), self.passes)

class MAF(Flow):
    r"""Creates a masked autoregressive flow (MAF).

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.

    Example:
        >>> flow = MAF(3, 4, transforms=3)
        >>> flow
        MAF(
          (transform): LazyComposedTransform(
            (0): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [0, 1, 2]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
            (1): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [2, 1, 0]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
            (2): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [0, 1, 2]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
          )
          (base): UnconditionalDistribution(DiagNormal(loc: torch.Size([3]), scale: torch.Size([3])))
        )
        >>> c = torch.randn(4)
        >>> x = flow(c).sample()
        >>> x
        tensor([-0.5005, -1.6303,  0.3805])
        >>> flow(c).log_prob(x)
        tensor(-3.7514, grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)

class MonotonicRQSTransform(Transform):
    r"""Creates a monotonic rational-quadratic spline (RQS) transformation.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        widths: The unconstrained bin widths, with shape :math:`(*, K)`.
        heights: The unconstrained bin heights, with shape :math:`(*, K)`.
        derivatives: The unconstrained knot derivatives, with shape :math:`(*, K - 1)`.
        bound: The spline's (co)domain bound :math:`B`.
        slope: The minimum slope of the transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: float = 14.0,
        slope: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        widths = widths / (1 + abs(2 * widths / math.log(slope)))
        heights = heights / (1 + abs(2 * heights / math.log(slope)))
        derivatives = derivatives / (1 + abs(derivatives / math.log(slope)))

        widths = F.pad(F.softmax(widths, dim=-1), (1, 0), value=0)
        heights = F.pad(F.softmax(heights, dim=-1), (1, 0), value=0)
        derivatives = F.pad(derivatives, (1, 1), value=0)

        self.horizontal = bound * (2 * torch.cumsum(widths, dim=-1) - 1)
        self.vertical = bound * (2 * torch.cumsum(heights, dim=-1) - 1)
        self.derivatives = torch.exp(derivatives)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bins={self.bins})"

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: LongTensor) -> Tuple[Tensor, ...]:
        mask = torch.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = torch.stack((k, k + 1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = hs.gather(-1, k0_k1).squeeze(dim=-1)
        y0, y1 = vs.gather(-1, k0_k1).squeeze(dim=-1)
        d0, d1 = ds.gather(-1, k0_k1).squeeze(dim=-1)

        s = (y1 - y0) / (x1 - x0)

        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Tensor, value: Tensor) -> LongTensor:
        seq, value = broadcast(seq, value.unsqueeze(dim=-1), ignore=1)
        seq = seq.contiguous()

        return torch.searchsorted(seq, value).squeeze(dim=-1)

    def _call(self, x: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (s + (d0 + d1 - 2 * s) * z * (1 - z))

        return torch.where(mask, y, x)

    def _inverse(self, y: Tensor) -> Tensor:
        k = self.searchsorted(self.vertical, y) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)

        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - (b**2 - 4 * a * c).sqrt())

        x = x0 + z * (x1 - x0)

        return torch.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (s + (d0 + d1 - 2 * s) * z * (1 - z))

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2
        )

        return torch.where(mask, y, x), mask * jacobian.log()

class NSF(MAF):
    r"""Creates a neural spline flow (NSF) with monotonic rational-quadratic spline
    transformations.

    By default, transformations are fully autoregressive. Coupling transformations
    can be obtained by setting :py:`passes=2`.

    Warning:
        Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.

    See also:
        :class:`zuko.transforms.MonotonicRQSTransform`

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


# example flow

# flow = NSF(144, 2, bins = 8, transforms = 8, randperm = True, hidden_features = [256,256])
# print(flow)