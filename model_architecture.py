import torch
import torch.nn as nn
import numpy as np


class Networks(nn.Module):

    """
    This script contains three neural structures: 
        - CNN (Convolutional Neural Network)
        - FNN (Multilayer Perceptron Network - Feedforward Network)
        - Shallow (Two-layer Neural Network)

    These parametric network models are designed to capture the energy function. The `num_f` parameter determines the size of the model, influencing the number of hidden units in each layer. Each network structure is defined within the `Networks` class, allowing for flexible model architecture choices.

    To choose a specific mode, set the `mode` parameter to 'cnn', 'ffn', or 'shallow'. The energy function is modeled using these structures, and the resulting model size is determined by the number of trainable parameters.

    Usage:
    ```python
    # Example usage for a CNN with num_f = 64
    cnn_model = Networks(num_f=64, mode='cnn')
    # Example usage for an FNN with num_f = 64
    ffn_model = Networks(num_f=64, mode='ffn')
    # Example usage for a Shallow network with num_f = 64
    shallow_model = Networks(num_f=64, mode='shallow')
    all these three models have about same number of parameters ~1m. 
    """

    
    def __init__(self, num_f, mode, **kwargs):
        super().__init__()

        c_hid1 = num_f // 4
        c_hid2 = num_f
        c_hid3 = num_f * 2
        c_hid4 = num_f * 4
        c_hid5 = num_f * 2

        self.logits = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=3, stride=1, padding=1),  # Layer 1 cnn
            nn.ReLU(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=4, stride=2, padding=1),  # Layer 2 cnn
            nn.ReLU(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=4, stride=2, padding=1),  # Layer 3 cnn
            nn.ReLU(),
            nn.Conv2d(c_hid3, c_hid4, kernel_size=4, stride=2, padding=1),  # Layer 4 cnn
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c_hid4 * 3 * 3, c_hid5),  # Layer 5 dense
            nn.ReLU(),
            nn.Linear(c_hid5, 10)  # Final output
        )

        self.model_size = sum(p.numel() for p in self.logits.parameters() if p.requires_grad)

        if mode == 'cnn':
            self.logits = self.logits

        elif mode == 'ffn':

            L = (np.sqrt((15 + (28 * 28))**2 - 16 * (10 - self.model_size)) - (15 + (28 * 28))) / 8
            L = int(L)

            self.logits = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, L),  # Layer 1
                nn.ReLU(),
                nn.Linear(L, L),  # Layer 2
                nn.ReLU(),
                nn.Linear(L, L),  # Layer 3
                nn.ReLU(),
                nn.Linear(L, L),  # Layer 4
                nn.ReLU(),
                nn.Linear(L, L),  # Layer 5
                nn.ReLU(),
                nn.Linear(L, 10)  # Final output
            )

            self.model_size = sum(p.numel() for p in self.logits.parameters() if p.requires_grad)
            self.model_size = sum(p.numel() for p in self.logits.parameters() if p.requires_grad)

        elif mode == 'shallow':

            L = (self.model_size - 10) / (28 * 28 + 11)
            L = int(L)
            self.logits = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, L),  # Layer 1
                nn.ReLU(),
                nn.Linear(L, 10)  # Final output
            )

            self.model_size = sum(p.numel() for p in self.logits.parameters() if p.requires_grad)

    def forward(self, x):
        neg_joint_en = self.logits(x).squeeze(dim=-1)
        neg_marginal_en = torch.log(torch.exp(neg_joint_en).sum(-1))
        return neg_marginal_en

