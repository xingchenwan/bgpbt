from abc import abstractmethod
from ConfigSpace import Configuration
from typing import List


class SearchSpace:

    def __init__(self):
        self.config_space = None

    @abstractmethod
    def train_single(self, config: Configuration, exp_idx: int, **kwargs):
        pass

    @abstractmethod
    def train_batch(self, configs: List[Configuration], exp_idx_start: int, **kwargs):
        pass

