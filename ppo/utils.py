import torch
class RolloutBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def get_batch(self):
        n_batches = len(self.states) // self.batch_size
        for i in range(n_batches):
            yield (
                torch.stack(self.states[i*self.batch_size:(i+1)*self.batch_size]).to(self.device),
                torch.stack(self.actions[i*self.batch_size:(i+1)*self.batch_size]).to(self.device),
                torch.stack(self.logprobs[i*self.batch_size:(i+1)*self.batch_size]).to(self.device),
                torch.tensor(self.rewards[i*self.batch_size:(i+1)*self.batch_size], dtype=torch.float32).to(self.device),
                torch.tensor(self.is_terminals[i*self.batch_size:(i+1)*self.batch_size], dtype=torch.bool).to(self.device)
            )
