import math
import torch
import torch.nn as nn

from src.ff_model import TDistributionActivation, ReLU_full_grad


class HybridModel(nn.Module):
    """Hybrid Model: FF layers for feature extraction, BP layers for classification."""

    def __init__(self, opt):
        super(HybridModel, self).__init__()
        self.opt = opt
        self.ff_num_channels = [self.opt.model.hidden_dim] * self.opt.model.ff_num_layers
        self.bp_num_channels = [self.opt.model.hidden_dim] * self.opt.model.bp_num_layers

        # FF-specific activation function
        if self.opt.model.ff_activation == "t_distribution":
            self.ff_act_fn = TDistributionActivation(nu=5)
        elif self.opt.model.ff_activation == "relu_full_grad":
            self.ff_act_fn = ReLU_full_grad()
        else:
            raise ValueError(f"Unknown FF activation: {self.opt.model.ff_activation}")

        # BP-specific activation function
        if self.opt.model.bp_activation == "relu":
            self.bp_act_fn = nn.ReLU()
        elif self.opt.model.bp_activation == "leaky_relu":
            self.bp_act_fn = nn.LeakyReLU(negative_slope=0.01)
        elif self.opt.model.bp_activation == "sigmoid":
            self.bp_act_fn = nn.Sigmoid()
        elif self.opt.model.bp_activation == "tanh":
            self.bp_act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown BP activation: {self.opt.model.bp_activation}")

        # FF layers
        self.ff_layers = nn.ModuleList([nn.Linear(3072, self.ff_num_channels[0])])
        for i in range(1, len(self.ff_num_channels)):
            self.ff_layers.append(nn.Linear(self.ff_num_channels[i - 1], self.ff_num_channels[i]))

        # BP layers
        self.bp_layers = nn.ModuleList([nn.Linear(self.ff_num_channels[-1], self.bp_num_channels[0])])
        for i in range(1, len(self.bp_num_channels)):
            self.bp_layers.append(nn.Linear(self.bp_num_channels[i - 1], self.bp_num_channels[i]))
        self.final_classifier = nn.Linear(self.bp_num_channels[-1], 10)

        # Loss functions
        self.ff_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.classification_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward_ff(self, x):
        """Forward pass through FF layers."""
        for layer in self.ff_layers:
            x = layer(x)
            if isinstance(self.ff_act_fn, torch.autograd.Function):  # Custom activation
                x = self.ff_act_fn.apply(x)
            else:
                x = self.ff_act_fn(x)
        return x

    def forward_bp(self, x):
        """Forward pass through BP layers."""
        for layer in self.bp_layers:
            x = layer(x)
            x = self.bp_act_fn(x)
        return self.final_classifier(x)

    def forward(self, inputs, labels):
        """Full forward pass."""
        scalar_outputs = {"Loss": torch.zeros(1, device=self.opt.device)}

        # FF forward pass
        x = inputs["neutral_sample"].reshape(inputs["neutral_sample"].shape[0], -1)
        ff_output = self.forward_ff(x)
        x = self._layer_norm(ff_output)
        # BP forward pass
        logits = self.forward_bp(x)

        # Classification loss and accuracy
        classification_loss = self.classification_loss(logits, labels["class_labels"])
        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss

        # Accuracy calculation
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels["class_labels"]).float().mean().item()
        scalar_outputs["classification_accuracy"] = accuracy

        return scalar_outputs
