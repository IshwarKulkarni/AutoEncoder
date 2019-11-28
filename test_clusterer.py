
import random
import torch

class TestCluster:
    """A known test cluster"""

    def __init__(self, label, num_tests=20):
        self.num_tests = num_tests
        self._batches = []
        self.label = label
        self.batch_mu = None

    def add_item(self, item):
        """Add an image item, known to be in this cluster"""
        self._batches.append(item)

    def rebatch(self, batch_size):
        assert len(self._batches) > batch_size, "Need atleast {} items to batch".format(batch_size)
        batches = []
        for b in range(0, len(self._batches), batch_size):
            batch = self._batches[b:b+batch_size]
            batch = torch.squeeze(torch.stack(batch), 1)
            batches.append(batch)
        self._batches = batches
        self.metadata = [self.label] * batch_size

    def intra_cluster_std(self, model):
        r = random.randint(0, len(self._batches)-2)
        batch_mu = model(self._batches[r])
        self.batch_mu = batch_mu
        self.center = batch_mu.mean(dim=0)
        return self.batch_mu.std(dim = 0).mean()

    def intra_cluster_sim(self, model, num_samples =10):
        r = random.randint(0, len(self._batches)-2)
        batch_mu = model(self._batches[r])
        self.batch_mu = batch_mu
        self.center = batch_mu.mean(dim=0)
        n = batch_mu.shape[0] - 1
        mean_sim = 0
        for _ in range(0, num_samples):
            r1 = random.randint(0, n)
            r2 = random.randint(0, n)
            while r1 == r2:
                r2 = random.randint(0, n)
            mean_sim += F.cosine_similarity(batch_mu[r1], batch_mu[r2], dim=0)

        return mean_sim / num_samples
