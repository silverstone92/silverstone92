import numpy as np
import torch
from torch.utils.data import Dataset

class InfoBatch(Dataset):
    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875):
        '''
        Args:
        - dataset: Dataset used for training.
        - num_epochs (int): The number of epochs to train the model.
        - prune_ratio (float): The proportion of samples being pruned during training.
        - delta (float): Epoch ratio to perform the pruning process.
        '''
        
        # Initialize the dataset, keep_ratio, num_epochs, delta, scores, weights, num_pruned_samples, and cur_batch_index
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        # Initialize the scores and weights
        self.scores = torch.ones(len(self.dataset)) * 3
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        # Set the current batch index
        # For tracking the current batch index to update the weights and scores
        self.cur_batch_index = cur_batch_indices

    def update(self, values):
        # Update the weights and scores
        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)
        loss_val = values.detach().clone()
        self.cur_batch_index = []
        # Update the scores, use cpu() to avoid memory leak
        self.scores[indices.cpu().long()] = loss_val.cpu()
        values.mul_(weights)
        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]

    def prune(self):
        # Prune the well-learned samples
        well_learned_mask = (self.scores < self.scores.mean()).numpy()
        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        selected_indices = np.random.choice(well_learned_indices, int(
            self.keep_ratio * len(well_learned_indices)), replace=False)
        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        return remained_indices

    @property
    def sampler(self):
        # Return the sampler
        sampler = IBSampler(self)
        return sampler

    def no_prune(self):
        # Do not prune the samples
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        # Return the mean score used for pruning
        return self.scores.mean()

    def get_weights(self, indexes):
        # Return the weights of the samples with the given indexes
        return self.weights[indexes]

    def get_pruned_count(self):
        # Return the number of pruned samples
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        # Return the epoch to stop pruning
        return self.num_epochs * self.delta

    def reset_weights(self):
        # Reset the weights
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        # Reset the sampler
        # If the number of iterations is greater than the epoch to stop pruning, reset the weights
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj)
        
    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self
