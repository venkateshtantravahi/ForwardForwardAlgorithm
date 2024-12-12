import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from src import utils

matplotlib.use('Agg')


class FF_CIFAR(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.cifar = utils.get_CIFAR_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.cifar)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _visualize_samples(dataset, index, num_classes=10):
        """Visualizes positive, negative, and neutral samples for a given index."""

        # Mean and std for unnormalizing CIFAR-10
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])

        # Generate samples
        pos_sample, neg_sample, neutral_sample, class_label = dataset._generate_sample(index)

        # Unnormalize images
        pos_img = utils.unnormalize_image(pos_sample, mean, std)
        neg_img = utils.unnormalize_image(neg_sample, mean, std)
        neutral_img = utils.unnormalize_image(neutral_sample, mean, std)

        # Plot the samples
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(pos_img)
        axes[0].set_title(f"Positive Sample (Label: {class_label})")
        axes[0].axis("off")

        axes[1].imshow(neg_img)
        axes[1].set_title("Negative Sample")
        axes[1].axis("off")

        axes[2].imshow(neutral_img)
        axes[2].set_title("Neutral Sample")
        axes[2].axis("off")

        # Save the plot instead of showing it
        plt.savefig("visualized_sample.png", dpi=300, bbox_inches="tight")
        print("Visualization saved to visualized_sample.png")
        plt.close()


    def _generate_sample(self, index):
        # Get CIFAR sample.
        sample, class_label = self.cifar[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label