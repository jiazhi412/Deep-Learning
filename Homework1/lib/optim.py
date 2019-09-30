from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                dv *= -self.lr
                if n in self.velocity:
                    dv += self.velocity[n] * self.momentum
                layer.params[n] += dv
                self.velocity[n] = dv
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for layer in self.net.layers:
            for n,dv in layer.grads.items():
                EG_T = (1 - self.decay) * np.square(dv)
                if n in self.cache:
                    EG_T += self.decay * self.cache[n]
                layer.params[n] -= self.lr * dv / np.sqrt(EG_T + self.eps)
                self.cache[n] = EG_T
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        for layer in self.net.layers:
            for n,dv in layer.grads.items():
                self.t += 1
                mt = (1 - self.beta1) * dv
                vt = (1 - self.beta2) * np.square(dv)
                if n in self.mt:
                    mt += self.beta1 * self.mt[n]
                if n in self.vt:
                    vt += self.beta2 * self.vt[n]
                m_t = mt / (1 - self.beta1 ** self.t)
                v_t = vt / (1 - self.beta2 ** self.t)
                layer.params[n] -= (self.lr * m_t) / (np.sqrt(v_t) + self.eps)
                self.mt[n] = mt
                self.vt[n] = vt
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
