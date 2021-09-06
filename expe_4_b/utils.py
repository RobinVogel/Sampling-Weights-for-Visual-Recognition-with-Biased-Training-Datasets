from typing import Any
import atexit
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch import Tensor
from typing import List, Iterable


from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Normalize,
    Compose,
    RandomCrop,
    CenterCrop,
    Resize,
    ToPILImage,
)


class Logger:
    """
    Logger that can log to tensorboard and tqdm at the same time.
    """

    def __init__(
        self, tensorboard_writer: SummaryWriter = None, tqdm_bar: tqdm = None
    ) -> None:
        """

        :param tensorboard_writer: An initialized tensorboard SummaryWriter to log to.
        :param tqdm_bar: A tqdm progress bar to log to.
        """
        self._tb_writer = tensorboard_writer
        self._tqdm_bar = tqdm_bar
        if self._tqdm_bar is not None:
            self._tqdm_data = {}

        def save():
            if self._tb_writer is not None:
                self._tb_writer.flush()
                self._tb_writer.close()

        # Don't have to rely on user to call flush and close on the tensorboard logger
        atexit.register(save)

    def log(self, key: str, val: Any, iteration: int = None) -> None:
        """
        Logs given key and value.
        :param key: Value name.
        :param val: Value to log.
        :param iteration: (Required if logging to tensorboard) Iteration the value is logged in.
        """
        assert key is not None and val is not None, "Please set key and val"

        if self._tb_writer is not None:
            assert (
                iteration is not None
            ), "Must specify iteration when logging to tensorboard"
            self._tb_writer.add_scalar(key, val, iteration)
        if self._tqdm_bar is not None:
            # update tqdm bar
            self._tqdm_data[key] = val
            self._tqdm_bar.set_postfix(self._tqdm_data, refresh=True)

    def set_tqdm_logger(self, tqdm_bar: tqdm) -> None:
        self._tqdm_bar = tqdm_bar
        self._tqdm_data = {}


def cifar_augmentation():
    std = (0.2023, 0.1994, 0.2010)
    mean = (0.4914, 0.4822, 0.4465)
    preprocessing = Compose(
        [
            ToTensor(),
            RandomCrop((40, 40), pad_if_needed=True),
            RandomCrop((32, 32)),
            RandomHorizontalFlip(),
            Normalize(mean, std),
        ]
    )
    return preprocessing


def cifar_normalization():
    std = (0.2023, 0.1994, 0.2010)
    mean = (0.4914, 0.4822, 0.4465)
    preprocessing = Compose(
        [
            ToTensor(),
            Normalize(mean, std),
        ]
    )
    return preprocessing


def accuracy(output: Tensor, target: Tensor, topk: Iterable = (1,)) -> List[Tensor]:
    """
    Computes top-k accuracy for all values of k
    :param output: output of classifier
    :param target: ground truth labels for all outputs
    :param topk: Tuples of all accuracies to compute
    :return: List of all computed accuracies
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res
