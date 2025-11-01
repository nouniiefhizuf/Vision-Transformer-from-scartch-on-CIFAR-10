# Vision Transformer (ViT) - CIFAR-10 Classification

A complete PyTorch implementation of Vision Transformer from scratch for image classification on the CIFAR-10 dataset.

## 📌 Project Overview

This project implements a Vision Transformer (ViT) model that applies transformer architecture—traditionally used in NLP—to computer vision tasks. Rather than processing images as grids of pixels like CNNs, ViT divides images into fixed-size patches and treats them as a sequence, enabling the model to capture global dependencies through self-attention mechanisms[web:21][web:22].

The implementation includes all core ViT components: patch embedding, positional encoding, multi-head self-attention, and a classification head with CLS token. The model is trained on CIFAR-10 with optional data augmentation for improved generalization[web:25].

## 🎯 Key Features

- **From-Scratch Implementation**: Pure PyTorch architecture without pre-trained weights
- **Modular Design**: Reusable components (PatchEmbedding, MultiHeadAttention, TransformerEncoderLayer)
- **Data Augmentation**: Fine-tuning with RandomHorizontalFlip and RandomCrop strategies
- **Reproducible**: Fixed random seeds and comprehensive documentation
- **Production-Ready Code**: Device-agnostic (GPU/CPU), proper error handling, and clean Jupyter cells

## 🏗️ Model Architecture

| Component | Configuration |
|-----------|----------------|
| Image Size | 32×32 pixels |
| Patch Size | 4×4 pixels |
| Total Patches | 64 patches/image |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| MLP Hidden Dimension | 512 |
| Total Parameters | ~2.7M |

**Architecture Flow**:
