from abc import ABC, abstractmethod


class HyperparameterOptimizer(ABC):

    def __init__(self, env, max_iters: int = 100, batch_size: int = 1, n_repetitions: int = 4, anneal_lr: bool = False):
        self.env = env
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.n_repetitions = n_repetitions
        self.X, self.y = [], []

    @abstractmethod
    def run(self):
        pass
