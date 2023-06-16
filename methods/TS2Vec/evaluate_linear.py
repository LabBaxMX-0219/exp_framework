import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, linear_in, linear_out):
        super(Linear, self).__init__()

        self.fc = nn.Linear(linear_in, linear_out)
    
    def forward(self, x):
        out = self.fc(x)
        return out
