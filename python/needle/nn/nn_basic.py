"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform(in_features, out_features)
        weight = Parameter(weight)
        if bias:
            bias = init.kaiming_uniform(out_features, 1).reshape((1, -1))
            bias = Parameter(bias)
        else:
            bias = None
        self.weight = weight
        self.bias = bias
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        hidden = ops.matmul(X, self.weight)
        if self.bias:
            bias_broadcast = ops.broadcast_to(self.bias, hidden.shape)
            hidden = hidden + bias_broadcast
        return hidden
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        new_shape = reduce(lambda x, y: x * y, X.shape[1:])
        return ops.reshape(X, (X.shape[0], new_shape))
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        onehot = init.one_hot(logits.shape[1], y)
        logsumexp = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation(logits * onehot, axes=(1,))
        loss = logsumexp - z_y
        loss = ops.summation(loss, axes=(0,)) / logits.shape[0]
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim)
        weight = Parameter(weight)
        bias = init.zeros(dim)
        bias = Parameter(bias)
        self.running_mean = init.zeros(dim, dtype=dtype)
        # running_mean = Parameter(running_mean)
        self.running_var = init.ones(dim, dtype=dtype)
        # running_var = Parameter(running_var)
        self.weight = weight
        self.bias = bias
        self.dtype = dtype
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        # print(f"x: {x.dtype}")
        if not self.training:
            mean = self.running_mean
            variance = self.running_var
            mean_broadcast = self._broadcast(mean, x.shape)
            variance_broadcast = self._broadcast(variance, x.shape)
        else:
            # compute mean and variance of the batch dimension
            mean = (ops.summation(x, axes=(0,)) / batch_size)
            # print(f"mean: {mean.dtype}")
            self.running_mean = self.momentum * mean.data + (1 - self.momentum) * self.running_mean
            # print(self.running_mean)
            mean_broadcast = self._broadcast(mean, x.shape)
            variance = (ops.summation((x - mean_broadcast) ** 2, axes=(0,)) / batch_size)
            # print(f"variance: {variance.dtype}")
            self.running_var = self.momentum * variance.data  + (1 - self.momentum) * self.running_var
            variance_broadcast = self._broadcast(variance, x.shape)
        
        # print(f"mean_broadcast: {mean_broadcast.dtype}")
        x_normalized = (x - mean_broadcast) / ops.sqrt(variance_broadcast + self.eps)
        # print(f"x_normalized: {x_normalized.dtype}")
        y = ops.broadcast_to(self.weight, x.shape) * x_normalized + ops.broadcast_to(self.bias, x.shape)
        # print(f"y: {y.dtype}")
        return y
        raise NotImplementedError()
        ### END YOUR SOLUTION
        
    def _broadcast(self, stats, shape):
        """ A helper function to broadcast mean and varience to the batch shape.
        """
        # stats_reshaped = ops.reshape(stats, (1, stats.shape[0]))
        return ops.broadcast_to(stats, shape)

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim)
        weight = Parameter(weight)
        bias = init.zeros(dim)
        bias = Parameter(bias)
        self.weight = weight
        self.bias = bias
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # compute mean and variance
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean_reshaped = ops.reshape(mean, (mean.shape[0], 1))
        mean_broadcast = ops.broadcast_to(mean_reshaped, x.shape)
        variance = ops.summation((x - mean_broadcast) ** 2, axes=(1,)) / self.dim
        # normalize
        variance_broadcast = ops.reshape(variance, (variance.shape[0], 1))
        variance_broadcast = ops.broadcast_to(variance_broadcast, x.shape)
        x_normalized = (x - mean_broadcast) / ops.sqrt(variance_broadcast + self.eps)
        y = ops.broadcast_to(self.weight, x.shape) * x_normalized + ops.broadcast_to(self.bias, x.shape)
        return y
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        else:
            mask = init.randb(*x.shape, p=self.p)
            mask = mask / (1 - self.p)
            return x * mask
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        raise NotImplementedError()
        ### END YOUR SOLUTION
