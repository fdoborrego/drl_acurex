import torch
import numpy as np


class AutoClipper:
    """
    Autoclip: Adaptative gradient clipping.

    Este AutoClipper permite saturar el valor de un gradiente al valor de un percentil "clip_percentile".
    """
    def __init__(self, model, clip_percentile=10, history_size=1000000):
        self.model = model
        self.clip_percentile = clip_percentile
        self.grad_history = []
        self.history_size = history_size
        self.ptr = 0

    def clear_history(self):
        self.grad_history = []

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def clip_gradient(self):
        obs_grad_norm = self._get_grad_norm()
        if len(self.grad_history) == self.history_size:
            self.grad_history.pop(0)
        self.grad_history.append(obs_grad_norm)
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
