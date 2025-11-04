# Vision Transformer (ViT) Implementation for CIFAR-10

This project is a PyTorch implementation of a Vision Transformer (ViT) model for image classification on the CIFAR-10 dataset. It also includes a baseline Convolutional Neural Network (CNN) model for performance comparison.

The notebook trains both models from scratch, evaluates their performance, and generates visualizations for model interpretability, including attention maps for the ViT and Grad-CAM for the CNN.

## 📋 Project Overview

The primary goal is to build a Vision Transformer from its core components (patch embedding, multi-head self-attention, MLP blocks) and compare its performance against a traditional CNN on a small dataset like CIFAR-10.

* **ViT Model:** A from-scratch implementation of the Vision Transformer architecture.
* **CNN Model:** A simple 4-layer CNN to serve as a baseline.
* **Dataset:** CIFAR-10, with standard augmentations (RandomCrop, RandomHorizontalFlip).
* **Evaluation:** Compares training/test accuracy, loss, and classification metrics for both models.
* **Visualization:**
    * Generates **Attention Maps** from the ViT's CLS token to show what the model focuses on.
    * Generates **Grad-CAM** heatmaps for the CNN to visualize class-discriminative regions.

## ⚙️ Model Architectures & Parameters

The models are trained using parameters derived from the student roll number `2205412`, which is also used as the random seed (`SEED = 2205412`) for reproducibility.

### Vision Transformer (ViT)

* **Patch Size:** 8x8
* **Number of Patches:** (32 / 8) * (32 / 8) = 16
* **Hidden Dimension (`embed_dim`):** 192
* **Transformer Layers:** 6
* **Attention Heads:** 6
* **MLP Dimension:** 768 (192 * 4)
* **Dropout:** 0.1

### Simple CNN (Baseline)

* **Conv 1:** 3 -> 32 filters, 3x3 kernel
* **Conv 2:** 32 -> 64 filters, 3x3 kernel
* **Conv 3:** 64 -> 128 filters, 3x3 kernel
* **Conv 4:** 128 -> 256 filters, 3x3 kernel
* **Pooling:** MaxPool2d (2, 2) after each of the first 3 conv layers.
* **Fully Connected:** 2 layers (256 * 2 * 2 -> 512, then 512 -> 10)
* **Regularization:** BatchNorm2d after each conv layer, Dropout (0.3) before the final layer.

## 📈 Results & Analysis

Both models were trained for **12 epochs** with an `AdamW` optimizer and a `CosineAnnealingLR` scheduler.

### Performance Comparison

| Model | Best Test Accuracy | Final Train Accuracy | Final Test Accuracy | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Vision Transformer** | **58.92%** | 59.15% | 58.79% | 0.59 |
| **Simple CNN** | **79.92%** | 80.50% | 79.92% | 0.80 |

### Key Takeaway

The `SimpleCNN` **significantly outperformed** the ViT model when trained from scratch on CIFAR-10 for 12 epochs. This result is expected, as Vision Transformers are notoriously data-hungry and typically require large-scale pre-training (like on ImageNet-21k) to build effective representations. The CNN's inherent inductive bias (translation invariance, locality) makes it much more data-efficient on smaller datasets.

## 🖼️ Generated Visualizations

The script automatically generates and saves the following analysis plots:

1.  **`model_comparison.png`**: Side-by-side line charts comparing the Training/Test Loss and Training/Test Accuracy of the ViT and CNN over 12 epochs.
2.  **`confusion_matrix_vision_transformer.png`**: A heatmap showing the classification performance of the ViT model.
3.  **`confusion_matrix_cnn.png`**: A heatmap showing the classification performance of the CNN model.
4.  **`attention_visualization.png`**: Visualizes the ViT's CLS token attention from different layers, showing how the model's focus evolves.
5.  **`gradcam_visualization.png`**: Shows the Grad-CAM (Gradient-weighted Class Activation Mapping) output for the CNN, highlighting the pixels most important for its prediction.

## 🚀 How to Run

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python tqdm
    ```

3.  **Run the Notebook:**
    Launch Jupyter Notebook and open `experiment.ipynb`.
    ```bash
    jupyter notebook experiments.ipynb
    ```

4.  **View Results:**
    Run all cells in the notebook. The training process will begin, and all 8 output files (model weights, plots, and results) will be saved to the root directory.


5.  **DEMO Video:**
    [![Click to Watch Video Demo](https://ik.imagekit.io/0ms2qhnkm/demo.mp4?updatedAt=1762232541120)](https://ik.imagekit.io/0ms2qhnkm/demo.mp4?updatedAt=1762232541120)
## 📂 Files Generated

Upon successful execution, the following files will be created:

1.  **`best_vision_transformer.pth`**: Best ViT model weights.
2.  **`best_simple_cnn.pth`**: Best CNN model weights.
3.  **`model_comparison.png`**: ViT vs CNN performance charts.
4.  **`confusion_matrix_vision_transformer.png`**: ViT confusion matrix.
5.  **`confusion_matrix_cnn.png`**: CNN confusion matrix.
6.  **`attention_visualization.png`**: ViT attention maps.
7.  **`gradcam_visualization.png`**: CNN Grad-CAM heatmaps.
8.  **`final_results.pth`**: A PyTorch file containing a dictionary with complete training/test histories and model parameters.
