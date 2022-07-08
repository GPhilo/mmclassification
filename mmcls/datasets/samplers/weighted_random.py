from mmcls.datasets import SAMPLERS
from .distributed_sampler import DistributedSampler

import torch

@SAMPLERS.register_module()
class DistributedWeightedSampler(DistributedSampler):
    # TODO: Add class weights? Could then be used in calculate_weights to scale the weights of each individual sample according to
    # some logic
    # > Or, dynamically calculate weights only if class weights were not explicitly given
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True, round_up=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, round_up=round_up, seed=seed)
        self.replacement = replacement

    def calculate_weights(self, targets):
        targets = torch.tensor(targets)
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            # deterministically shuffle based on epoch
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # > This is the only part different from DistributedSampler
        targets = self.dataset.get_gt_labels()
        weights = self.calculate_weights(targets)
        subsample_balanced_indices = torch.multinomial(weights, self.total_size, self.replacement, generator=g)
        subsample_balanced_indices = subsample_balanced_indices[indices]

        return iter(subsample_balanced_indices.tolist())
