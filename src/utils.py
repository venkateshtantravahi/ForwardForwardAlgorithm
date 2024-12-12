import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice
from torch.utils.data import default_collate
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src import ff_cifar, ff_model, ff_bp_model

NUM_CLASSES = 10

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer

cutmix = CutMix(num_classes=NUM_CLASSES)
mixup = MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

def get_data(opt, partition, visualize=False, index=10):
    dataset = ff_cifar.FF_CIFAR(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    if visualize:
      ff_cifar.FF_CIFAR._visualize_samples(dataset, index)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
        # collate_fn=collate_fn,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_CIFAR_partition(opt, partition):
    # Define the normalization parameters for CIFAR-10 (mean and std for RGB channels)
    normalize_transform = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    # Define the transformation pipeline (convert to tensor and normalize)
    transforms_val_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert PIL Image or numpy array to tensor
        normalize_transform,    # Normalize RGB channels with CIFAR-10-specific values
    ])

    transforms = torchvision.transforms.Compose([
      torchvision.transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
      torchvision.transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
      # torchvision.transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
      #                        contrast = 0.1,
      #                        saturation = 0.1),
      torchvision.transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1), # Randomly adjust sharpness
      torchvision.transforms.ToTensor(),   # Converting image to tensor
      normalize_transform, # Normalizing with standard mean and standard deviation
      # torchvision.transforms.v2.CutMix(alpha=1.0),
      # torchvision.transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False),
    ])

    if partition in ["train", "val", "train_val"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transforms,
        )
    elif partition in ["test"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transforms_val_test,
        )
    else:
        raise NotImplementedError

    if partition == "train":
        cifar = torch.utils.data.Subset(cifar, range(40000))
    elif partition == "val":
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transforms_val_test,
        )
        cifar = torch.utils.data.Subset(cifar, range(40000, 50000))

    return cifar


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict

def unnormalize_image(img, mean, std):
    """Reverses normalization for visualization."""
    img = img.permute(1, 2, 0).numpy()  # Convert CHW to HWC and to a NumPy array
    img = img * std + mean  # Broadcast mean and std across the channels
    img = np.clip(img, 0, 1)  # Clip pixel values to [0, 1] for display
    return img


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if key != "logits":
            if isinstance(value, float):
                result_dict[key] += value / num_steps
            else:
                result_dict[key] += value.item() / num_steps
    return result_dict

def save_and_plot_confusion_matrix(opt, preds, labels, partition, epoch):
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    classes = list(range(cm.shape[0]))

    # Save confusion matrix to a file
    cm_file_path = os.path.join(opt.run.dir, f"{partition}_confusion_matrix_epoch_{epoch}.npy")
    np.save(cm_file_path, cm)
    print(f"Confusion matrix saved at {cm_file_path}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({partition}) - Epoch {epoch}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_plot_path = os.path.join(opt.run.dir, f"{partition}_confusion_matrix_epoch_{epoch}.png")
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved at {cm_plot_path}")
    plt.close()

def unnormalized_image(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])
    img = img.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
    img = img * std + mean  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip values to [0, 1]
    return img

def plot_sample_predictions(opt, correct_images, incorrect_images, partition, epoch, num_samples=5):
    # Select samples
    correct_samples = correct_images[:num_samples]
    incorrect_samples = incorrect_images[:num_samples]

    # Plot correct predictions
    plt.figure(figsize=(15, 5))
    for i, (img, true_label, pred_label) in enumerate(correct_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(unnormalized_image(img))
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis("off")
    plt.suptitle(f"Correct Predictions ({partition}) - Epoch {epoch}")

    # Save correct predictions plot
    correct_plot_path = os.path.join(
        opt.run.dir, f"{partition}_correct_predictions_epoch_{epoch}.png"
    )
    plt.savefig(correct_plot_path)
    print(f"Correct predictions plot saved at {correct_plot_path}")
    plt.close()

    # Plot incorrect predictions
    plt.figure(figsize=(15, 5))
    for i, (img, true_label, pred_label) in enumerate(incorrect_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(unnormalized_image(img))
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis("off")
    plt.suptitle(f"Incorrect Predictions ({partition}) - Epoch {epoch}")

    # Save incorrect predictions plot
    incorrect_plot_path = os.path.join(
        opt.run.dir, f"{partition}_incorrect_predictions_epoch_{epoch}.png"
    )
    plt.savefig(incorrect_plot_path)
    print(f"Incorrect predictions plot saved at {incorrect_plot_path}")
    plt.close()

def get_optimizer(opt, parameters):
    return torch.optim.SGD(parameters, lr=opt.training.learning_rate, momentum=opt.training.momentum)

def get_scheduler(opt, optimizer):
    if opt.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.training.epochs)
    elif opt.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        return None

def initialize_hybrid_model(opt):
    return ff_bp_model.HybridModel(opt)
