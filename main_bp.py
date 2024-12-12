import time
from collections import defaultdict
import hydra
import torch
from omegaconf import DictConfig
from src import utils


def train_hybrid_model(opt, model):
    """
    Train the Hybrid Model with FF for feature extraction and BP for classification.
    """
    # Create shared data loaders
    data_loader = utils.get_data(opt, "train")
    val_loader = utils.get_data(opt, "val")

    # Optimizer and scheduler for BP layers
    optimizer = utils.get_optimizer(opt, model.bp_layers.parameters())
    scheduler = utils.get_scheduler(opt, optimizer)

    start_time = time.time()

    for epoch in range(opt.training.epochs):
        print(f"\nEpoch {epoch + 1}/{opt.training.epochs}")

        # Training loop
        model.train()
        epoch_results = defaultdict(float)

        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            # Forward pass through FF layers
            ff_output = model.forward_ff(inputs["neutral_sample"].view(inputs["neutral_sample"].shape[0], -1))

            # Forward pass through BP layers using FF output
            logits = model.forward_bp(ff_output)

            # Compute BP loss
            classification_loss = model.classification_loss(logits, labels["class_labels"])

            # Backpropagation and optimization
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            # Log results
            epoch_results["classification_loss"] += classification_loss.item()
            epoch_results["classification_accuracy"] += utils.get_accuracy(
                opt, logits.data, labels["class_labels"]
            )

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Print results for the epoch
        print(f"Loss: {epoch_results['classification_loss']:.4f}, "
              f"Accuracy: {epoch_results['classification_accuracy'] / len(data_loader):.4f}")

        # Validation phase
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(model, val_loader, opt, partition="val", epoch=epoch)

    print("\nTraining complete!")
    return model


def validate_or_test(model, data_loader, opt, partition="val", epoch=None):
    """
    Validate or test the hybrid model.
    """
    print(f"\n{partition.capitalize()} phase:")
    start_time = time.time()
    test_results = defaultdict(float)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            # Forward pass through FF and BP layers
            ff_output = model.forward_ff(inputs["neutral_sample"].view(inputs["neutral_sample"].shape[0], -1))
            logits = model.forward_bp(ff_output)

            # Compute loss and accuracy
            classification_loss = model.classification_loss(logits, labels["class_labels"])
            accuracy = utils.get_accuracy(opt, logits.data, labels["class_labels"])

            test_results["classification_loss"] += classification_loss.item()
            test_results["classification_accuracy"] += accuracy

            # Collect predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels["class_labels"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    # Log and visualize results
    utils.save_and_plot_confusion_matrix(opt, all_preds, all_labels, partition, epoch)
    utils.print_results(partition, time.time() - start_time, test_results, epoch=epoch)


@hydra.main(config_path=".", config_name="config_bp", version_base=None)
def my_main(opt: DictConfig):
    opt = utils.parse_args(opt)
    model = utils.initialize_hybrid_model(opt)  # Initialize the FF + BP hybrid model
    model = train_hybrid_model(opt, model)

    if opt.training.final_test:
        test_loader = utils.get_data(opt, "test")
        validate_or_test(model, test_loader, opt, partition="test")


if __name__ == "__main__":
    my_main()