import torch
from torch.utils.data import DataLoader, Dataset
import pickle

class TrajectoryDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.trajectory = pickle.load(f)

    def __len__(self):
        return len(self.trajectory['states'])

    def __getitem__(self, idx):
        state = torch.tensor(self.trajectory['states'][idx])
        action = torch.tensor(self.trajectory['actions'][idx])
        reward = torch.tensor(self.trajectory['rewards'][idx])
        next_state = torch.tensor(self.trajectory['next_states'][idx])
        done = torch.tensor(self.trajectory['dones'][idx])
        return state, action, reward, next_state, done

class TrajectoryDataLoader(DataLoader):
    def __init__(self, filename, batch_size=32, shuffle=True):
        dataset = TrajectoryDataset(filename)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage:
filename = 'trajectory.pickle'
batch_size = 32
shuffle = True

trajectory_loader = TrajectoryDataLoader(filename, batch_size=batch_size, shuffle=shuffle)
