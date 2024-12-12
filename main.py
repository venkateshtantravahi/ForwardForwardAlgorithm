import time
from collections import defaultdict
import os

import hydra
import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig
from torchviz import make_dot

from src import utils


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train", visualize=True, index=10)
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)
        # Initialize the CosineAnnealingLR scheduler
        # scheduler = CosineAnnealingLR(optimizer, T_max=opt.training.epochs)
        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )
        # Update the learning rate
        # scheduler.step()
        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    # Create logs directory if not exists
    os.makedirs(opt.run.dir, exist_ok=True)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    all_preds, all_labels, correct_images, incorrect_images = [], [], [], []

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

            # Extract predictions and true labels
            logits = scalar_outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels['class_labels'].cpu().numpy()

            # Save predictions and labels for confusion matrix
            all_preds.extend(preds)
            all_labels.extend(true_labels)

            # Save some correct and incorrect predictions
            for i in range(len(preds)):
                if preds[i] == true_labels[i]:
                    correct_images.append((inputs['neutral_sample'][i], true_labels[i], preds[i]))
                else:
                    incorrect_images.append((inputs['neutral_sample'][i], true_labels[i], preds[i]))

    # Log and save confusion matrix
    utils.save_and_plot_confusion_matrix(opt, all_preds, all_labels, partition, epoch)

    # Plot sample correct and incorrect predictions
    utils.plot_sample_predictions(
        opt, correct_images, incorrect_images, partition, epoch, num_samples=5
    )
    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()

def visualize_model(opt, model):
    # Initialize the model
    # model = FF_model(opt)

    # Create dummy input
    dummy_input = {
        "pos_images": torch.randn(opt.input.batch_size, 3, 32, 32),
        "neg_images": torch.randn(opt.input.batch_size, 3, 32, 32),
        "neutral_sample": torch.randn(opt.input.batch_size, 3, 32, 32)
    }
    dummy_labels = {"class_labels": torch.randint(0, 10, (opt.input.batch_size,))}

    # Perform a forward pass to create the computational graph
    outputs = model(dummy_input, dummy_labels)

    # Generate the visualization
    graph = make_dot(outputs["logits"], params=dict(model.named_parameters()))
    graph.render("FF_Model_Visualization", format="png")
    print("Model visualization saved as FF_Model_Visualization.png")


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    visualize_model(opt, model)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    my_main()