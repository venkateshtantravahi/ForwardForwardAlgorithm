# **Forward-Forward Algorithm: Experimentation on CIFAR-10**

## **Project Overview**
This project investigates the **Forward-Forward (FF) Algorithm**, a novel alternative to Backpropagation (BP) for training neural networks. The key objectives are:
- Extending the FF algorithm to handle complex datasets like CIFAR-10.
- Experimenting with various activation functions, including custom activations (`ReLU_Full_Grad`, `T-Distribution`).
- Comparing FF against BP to evaluate performance, stability, and computational efficiency.

### **Key Features**
- **Goodness-based Layer Training**: FF trains layers independently by maximizing goodness scores for positive samples and minimizing them for negative samples.
- **Custom Activations**: Incorporates unique activations tailored for FF.
- **Dataset**: CIFAR-10, with custom positive, negative, and neutral samples for FF.
- **Performance Comparison**: Comprehensive analysis of FF and BP algorithms using multiple metrics.

---

## **Directory Structure**
```plaintext
FF/
├── Charts&Results/
│   ├── graphs/                   # Plots for results visualization
│   ├── Results/                  # Tabular summaries of experiments
├── datasets/
│   ├── cifar-10-batches-py/      # Raw CIFAR-10 data
│   ├── FashionMNIST/             # Fashion MNIST data (optional)
│   ├── cifar-10-python.tar.gz    # Compressed CIFAR-10 dataset
├── logs/                         # Logs for training/validation/testing
│   ├── experiments/              # Configurations and run-specific outputs
├── outputs/                      # Model checkpoints and predictions
├── src/                          # Main source code
│   ├── ff_model.py               # FF model definition
│   ├── ff_cifar.py               # Dataset handling and training loops
│   ├── ff_bp_model.py            # Hybrid FF-BP implementation
│   ├── utils.py                  # Helper utilities
│   ├── main.py                   # Main execution script for FF
│   ├── main_bp.py                # BP-only execution script
├── notebooks/
│   ├── CIFAR_BP.ipynb            # Experimentation with BP
│   ├── parse_and_visualize.ipynb # Visualization and metrics parsing
├── config.yaml                   # Configurations for FF experiments
├── config_bp.yaml                # Configurations for BP experiments
├── environment.yml               # Conda environment file
├── README.md                     # Project overview and instructions
```
# **Installation**

## 1. Clone the Repository
```plaintext
    git clone [<repository-url>](https://github.com/venkateshtantravahi/ForwardForwardAlgorithm.git)
    cd FF
```

## 2. Set Up the Environment

Use the `environment.yml` file to create a Conda environment:
```plaintext
conda env create -f environment.yml
conda activate FF
```

## 3. Dataset Setup
The `CIFAR-10` dataset will be automatically downloaded when running the scripts for the first time.

# **Execution Instructions**

## 1. Running the FF Algorithm
To train the model using the Forward-Forward algorithm:
```
    python src/main.py
```
Hydra configuration will automatically handle the settings from `config.yaml`. 
Logs and results will be stored in the `logs/` directory.

## 2. Running the BP Algorithm
To compare FF with Backpropagation:
```
python src/main_bp.py
```

## 3. Visualizing Results
To generate and view visualizations from the experimental results:
```
jupyter notebook notebooks/parse_and_visualize.ipynb
```

# **Experimental Features**

## Forward-Forward Algorithm
- **Goodness-Based Training:** FF layers are trained by maximizing the goodness scores for positive samples and minimizing them for negative samples.
- **Custom Activations:** Includes novel activations like `ReLU_Full_Grad` and `T-Distribution` for FF optimization.

## Backpropagation Algorithm
- Standard BP implementation using gradient descent.
- Benchmarked against FF for identical tasks and datasets.

# **Results**

- **Performance Trends:** FF shows competitive results but lacks BP's overall stability in handling complex datasets like CIFAR-10.
- **Activation Comparisons:** Detailed analysis of different activation functions across FF and BP.
For detailed results, refer to the `Charts&Results/Results` directory.
