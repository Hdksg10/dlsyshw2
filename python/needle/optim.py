"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for parm in self.params:
            if parm not in self.u:
                self.u[parm] = ndl.zeros(*parm.shape)
            # print(parm.dtype)
            self.u[parm] = self.momentum * self.u[parm] + (1 - self.momentum) * (parm.grad.data + self.weight_decay * parm.data)
            self.u[parm] = ndl.Tensor(self.u[parm].data, device=parm.device, dtype="float32", requires_grad=parm.requires_grad)
            
            parm.data = parm.data - self.lr * self.u[parm]

        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        for parm in self.params:
            grad_norm = np.linalg.norm(parm.grad.data)
            if grad_norm > max_norm:
                parm.grad = parm.grad / grad_norm * max_norm
            
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            # if param.requires_grad == False:
            #     continue
            if param not in self.m:
                self.m[param] = ndl.zeros(*param.shape)
            if param not in self.v:
                self.v[param] = ndl.zeros(*param.shape)
            grad = self.weight_decay * param.data + param.grad.data
            # print(grad)
            self.m[param] = self.beta1 * self.m[param].data + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param].data + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            # m_hat = self.m[param]
            # v_hat = self.v[param]
            m_hat = ndl.Tensor(m_hat.data, device=param.device, dtype="float32", requires_grad=param.requires_grad)
            v_hat = ndl.Tensor(v_hat.data, device=param.device, dtype="float32", requires_grad=param.requires_grad)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)    
        
        # raise NotImplementedError()
        ### END YOUR SOLUTION
