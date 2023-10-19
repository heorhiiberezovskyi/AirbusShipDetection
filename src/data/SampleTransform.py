from abc import ABCMeta, abstractmethod
from typing import Tuple

from numpy import ndarray


class SampleTransform:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        raise NotImplemented('')
