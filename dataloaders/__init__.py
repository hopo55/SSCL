from __future__ import absolute_import

from .dataloader import iCIFAR100, iCIFAR10, SSCLDataLoader

from .loader import CIFAR10, CIFAR100

__all__ = ('iCIFAR100','iCIFAR10', 'SSCLDataLoader', 'CIFAR10', 'CIFAR100')