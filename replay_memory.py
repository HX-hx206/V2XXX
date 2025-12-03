import os
import random
import numpy as np

class ReplayMemory:

    def __init__(self, entry_size, memory_size=50000, batch_size=64, alpha=0.6, beta=0.4):
        self.entry_size = int(entry_size)
        self.memory_size = int(memory_size)
        self.batch_size = int(batch_size)

        # PER hyperparams
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = 1e-6


        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)


        self.priorities = np.zeros(self.memory_size, dtype=np.float32)
        self.count = 0
        self.current = 0

    def add(self, prestate, poststate, reward, action, priority=None):

        idx = self.current
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.prestate[idx] = prestate
        self.poststate[idx] = poststate

        if priority is None:
            max_p = self.priorities.max() if self.count > 0 else 1.0
            self.priorities[idx] = max_p
        else:
            self.priorities[idx] = max(float(priority), self.eps)

        self.count = max(self.count, idx + 1)
        self.current = (idx + 1) % self.memory_size

    def _probabilities(self):

        n = self.count
        if n == 0:
            raise RuntimeError("ReplayMemory.sample() called with empty buffer")
        scaled = np.power(self.priorities[:n] + self.eps, self.alpha)
        denom = scaled.sum()
        if not np.isfinite(denom) or denom <= 0.0:
            # Fallback to uniform if priorities are degenerately small
            return np.ones(n, dtype=np.float32) / float(n)
        return scaled / denom

    def sample(self, beta=None):
        """Sample a batch according to priorities. Returns extra (idxs, is_weights) for PER."""
        if beta is None:
            beta = self.beta
        n = self.count
        if n == 0:
            raise RuntimeError("ReplayMemory.sample() called with empty buffer")

        probs = self._probabilities()

        size = min(self.batch_size, n)
        # choice without replacement, weighted by probs
        idxs = np.random.choice(n, size=size, replace=False, p=probs)

        # Importance Sampling weights
        weights = (n * probs[idxs]) ** (-beta)
        weights /= (weights.max() + 1e-12)
        weights = weights.astype(np.float32)

        return (self.prestate[idxs],
                self.poststate[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                idxs,
                weights)

    def update_priorities(self, idxs, new_priorities):
        """Update priorities for sampled indices; usually use abs(TD)+eps."""
        idxs = np.asarray(idxs, dtype=np.int32)
        ps = np.asarray(new_priorities, dtype=np.float32)
        for i, p in zip(idxs, ps):
            self.priorities[int(i)] = max(float(p), self.eps)
