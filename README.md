
# Uncertainty-Rebalanced Multimodal Fusion (URMF) for Spatial Multi-Omics Learning

## 📌 Introduction

This project implements a **VAE + GCN + EURO + UDR**-based multimodal learning framework for **spatial multi-omics data fusion**. Inspired by **URMF (Uncertainty-Rebalanced Multimodal Fusion)**, the model dynamically balances **epistemic uncertainty** across different omics modalities (e.g., RNA and Protein). The fusion strategy is powered by **Gating Functions** and **Uncertainty-Driven Regularization (UDR)**.

## 🚀 Features

- **Graph Convolutional Networks (GCN)** for capturing spatial dependencies
- **Variational Autoencoders (VAE)** for learning latent representations
- **Uncertainty-Driven Regularization (UDR)** for robustness
- **Epistemic Uncertainty-Rebalanced Optimization (EURO)** for adaptive training
- **Channel Attention Mechanism** for multimodal fusion
- **Task-Specific Adaptation (TSA)** for improved representation learning

## 📁 Project Structure

```
|-- spatialglue/
    |-- preprocess.py  # Data processing and graph construction
    |-- model.py       # VAE + GCN + EURO + UDR implementation
    |-- train.py       # Training pipeline
    |-- utils.py       # Utility functions
    |-- attention.py   # Channel attention and gating functions
    |-- readme.md      # Project documentation
```

## 🛠 Installation

This project requires **Python 3.8+** and the following dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas scanpy anndata tqdm
pip install scikit-learn seaborn matplotlib
```

If using **CUDA**, install the appropriate **PyTorch version** from [PyTorch.org](https://pytorch.org/get-started/locally/).

## 🏗 Model Architecture

The model consists of **two encoders** and **two decoders**, connected via **uncertainty-aware multimodal fusion**:

### **Encoder (VAE + GCN)**

- **Graph Convolutional Networks (GCN)** for learning spatial relations
- **VAE-style encoders** for each modality (`omics1`, `omics2`)
- **Epistemic uncertainty modeling** via variance estimation

### **Multimodal Fusion (EURO + UDR)**

- **Gating Function** for dynamic uncertainty reweighting
- **UDR (Uncertainty-Driven Regularization)** for robust learning
- **Channel Attention (ChannelGate)** for feature integration

### **Decoder (GCN Reconstruction)**

- **Modality-specific decoders** for reconstructing input features
- **L_recon + KL loss** for VAE training

## 📊 Training Pipeline

To train the model on spatial multi-omics data:

```python
from spatialglue.train import Train_SpatialGlue

trainer = Train_SpatialGlue(data=your_adata_dict, device='cuda', epochs=600)
results = trainer.train()
```

### **Loss Functions**

- **Reconstruction Loss (L_recon)**: MSE loss for feature reconstruction
- **KL Divergence Loss**: VAE regularization
- **Adaptive Gradient Reweighting (EURO)**: Uncertainty-based weight adjustment

## 📌 Example Usage

```python
from spatialglue.model import Encoder_overall

model = Encoder_overall(dim_input1=3000, dim_output1=64, dim_input2=3000, dim_output2=64)
output = model(features_omics1, features_omics2, adj_spatial, adj_feature_omics1, adj_feature_omics2)
```

## 📝 Citation

If you use this code for your research, please cite the original **URMF** paper:

```bibtex
@article{urmf2023,
  author    = {Author Name},
  title     = {Uncertainty-Rebalanced Multimodal Fusion for Spatial Multi-Omics Learning},
  journal   = {NeurIPS},
  year      = {2023}
}
```

## 📬 Contact

For questions and collaborations, please contact **1781748187@qq.com** or create an issue in this repository.
