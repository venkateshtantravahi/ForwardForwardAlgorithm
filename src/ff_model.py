import math

import torch
import torch.nn as nn

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers

        # Dynamically select activation function
        if self.opt.model.activation == "relu":
            self.act_fn = nn.ReLU()
        elif self.opt.model.activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=0.01)
        elif self.opt.model.activation == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif self.opt.model.activation == "tanh":
            self.act_fn = nn.Tanh()
        elif self.opt.model.activation == "relu_full_grad":
            self.act_fn = ReLU_full_grad()
        elif self.opt.model.activation == "t_distribution":
            self.act_fn = TDistributionActivation(nu=5)
        else:
            raise ValueError(f"Unknown activation function: {self.opt.model.activation}")

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(3072, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()
        # self.ff_loss = nn.CrossEntropyLoss(reduction='mean')

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False),
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                if self.opt.model.activation == "t_distribution":
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    torch.nn.init.normal_(
                        m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                    )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)
        # Adjust FF loss calculations if T-distribution is used
        if isinstance(self.act_fn, TDistributionActivation):
            sum_of_squares = sum_of_squares / z.shape[1]  # Normalize for T-distribution

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)

            if isinstance(self.act_fn, torch.autograd.Function):  # Custom activation
                z = self.act_fn.apply(z)
            else:  # Standard PyTorch activation
                z = self.act_fn(z)
            # z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)

                if isinstance(self.act_fn, torch.autograd.Function):  # Custom activation
                    z = self.act_fn.apply(z)
                else:  # Standard PyTorch activation
                    z = self.act_fn(z)
                # z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        scalar_outputs["logits"] = output
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class TDistributionActivation(torch.autograd.Function):
    """Custom activation function based on the negative log of the density under a t-distribution."""

    @staticmethod
    def forward(ctx, input, nu=10):
        """
        Forward pass of the t-distribution-based activation function.
        Args:
            input: Input tensor.
            nu: Degrees of freedom for the t-distribution (default: 10).
        Returns:
            Tensor after applying the activation function.
        """
        ctx.nu = nu
        # precompute constants
        constant_term = (
            math.log(math.sqrt(nu * math.pi)) +
            torch.lgamma(torch.tensor(nu / 2.0)) -
            torch.lgamma(torch.tensor(nu + 1.0) / 2.0)
        )
        # compute activation
        activation = constant_term + ((nu + 1) / 2.0) * torch.log(1 + (input ** 2) / nu)
        ctx.save_for_backward(input)
        return activation

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the activation function.
        Args:
            grad_output: Gradient of the loss with respect to the output.
        Returns:
            Gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        nu = ctx.nu

        # Compute the gradient
        grad_input = grad_output * ((nu + 1) * input) / (nu + input ** 2)
        return grad_input, None  # Return None for `nu` since it's not trainable

