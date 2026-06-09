"""Hidden state pipeline for compensation training."""

from torch.utils.data import Dataset


class HiddenStateDataset(Dataset):
    """Wraps a tensor of hidden states for DataLoader consumption.
    
    Each item returns (index, state_vector).
    """
    def __init__(self, states, device):
        self.states = states
        self.device = device

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return idx, self.states[idx].to(self.device)
