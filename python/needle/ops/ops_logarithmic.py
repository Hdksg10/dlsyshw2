from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        log_sum_exp = logsumexp(Tensor(Z))
        log_sum_exp = broadcast_to(log_sum_exp, Z.shape)
        result = Z - log_sum_exp
        # print(result)
        return result
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_sub = Z - Z_max
        Z_exp = array_api.exp(Z_sub)
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        Z_log = array_api.log(Z_sum)
        logsumexp = Z_max + Z_log
        logsumexp = array_api.squeeze(logsumexp, axis=self.axes)
        return logsumexp
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        # print(Z.shape)
        # max_indices = array_api.argmax(Z, axis=self.axes, keepdims=True)
        # z_max = array_api.take(Z, max_indices, axis=self.axes)
        z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        shape_list = list(Z.shape)
        axes = self.axes if self.axes is not None else tuple(range(len(Z.shape)))
        for axe in axes:
            shape_list[axe] = 1
        reshaped = reshape(out_grad, shape=shape_list)
        sumexp = array_api.sum(array_api.exp(Z - z_max), axis=self.axes, keepdims=True)
        grad = array_api.exp(Z - z_max) / sumexp
        # grad = (Tensor(Z) - broadcast_to(Tensor(Z_max), Z.shape)) * broadcast_to(reshaped, Z.shape)
        return reshaped * grad
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

