#  Multi-Modal Deep Learning Architectures for Visual Servoing

> MSc Data Science Dissertation — University of Surrey (2024–2025)  
> Supervised by Dr Amir Esfahani

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-orange)]()

---

## 📌 Overview

This project presents a **comparative study of two multi-modal deep learning architectures** designed to map visual inputs (encoded images) and trajectory information to robot control weights — a task known as **visual servoing**.

Traditional visual servoing techniques (IBVS, PBVS) require careful hand-engineering of features and struggle in unstructured environments. This research explores whether deep learning can provide a more flexible and generalisable alternative.

**Two architectures are compared:**
| Model | Architecture | Key Strength |
|---|---|---|
| Model 1 | CNN-MLP Hybrid | Spatial feature extraction + trajectory fusion |
| Model 2 | Vision Transformer (ViT) | Self-attention over image patches, stable training |

---

## 🏗️ Repository Structure

```
├── notebooks/
│   ├── 01_MLP_and_CNN.ipynb              # Standalone MLP and CNN baselines
│   ├── 02_Hybrid_Neural_Network.ipynb    # CNN-MLP Hybrid architecture
│   ├── 03_Vision_Transformer.ipynb       # ViT-based model
│   ├── 04_LLM_Transformer.ipynb          # LLM transformer exploration
│   ├── 05_Euclidean_Loss.ipynb           # Custom Euclidean loss function
│   └── 06_models_main.ipynb              # Full training pipeline with real data
│
├── models/
│   ├── mlp.py                            # Fully Connected MLP
│   ├── cnn.py                            # Convolutional Neural Network
│   ├── hybrid.py                         # CNN-MLP Hybrid model
│   └── vision_transformer.py             # Vision Transformer (ViT)
│
├── utils/
│   ├── dataset.py                        # Custom PyTorch Dataset & DataLoader
│   ├── losses.py                         # Euclidean loss implementation
│   └── train.py                          # Training loop & evaluation utilities
│
├── docs/
│   └── dissertation.pdf                  # Full MSc dissertation
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### Model 1 — CNN-MLP Hybrid
Combines a **Convolutional Neural Network** for processing encoded images with a deep **Fully Connected Network** handling trajectory data. Outputs are concatenated and passed through a final prediction head.

```
Image (128×128×3) → CNN → 64-dim visual features
Trajectory data   → MLP → 64-dim trajectory features
                         ↓
               Concatenate → FC → Predicted weights (70)
```

### Model 2 — Vision Transformer (ViT)
Divides input images into **2×2 patches**, projects them into a 256-dim embedding space, and processes them through **6 transformer layers with 8 attention heads**.

```
Image → Patch Embedding (2×2) → Positional Encoding
      → Transformer Encoder (6 layers, 8 heads, dim=256)
      → MLP Head → Predicted weights (70)
```

---

## 📊 Results

Both models were evaluated using MSE, MAE, R² Score, and accuracy within a 10% threshold:

| Metric | CNN-MLP Hybrid | Vision Transformer |
|---|---|---|
| MSE | 1,620,828 | 2,181,668 |
| MAE | 363.16 | 431.59 |
| R² Score | -0.0716 | -0.0479 |
| Accuracy (±10%) | 0.22% | 1.11% |

**Key findings:**
- The CNN-MLP Hybrid achieved lower raw error (MSE/MAE)
- The Vision Transformer showed a marginally better R² and accuracy within threshold
- The ViT demonstrated more stable training (smoother loss curve)
- Both models are significantly constrained by the small dataset (~100 aligned samples)

> The high weight values stem from Gaussian basis function widths used in the weight space representation, not model failure alone.

---

## 🔧 Setup & Usage

### Prerequisites
```bash
git clone https://github.com/christina-kamble/visual-servoing-multimodal.git
cd visual-servoing-multimodal
pip install -r requirements.txt
```

### Running the notebooks
Each notebook in `notebooks/` is self-contained with inline explanations. Start with:
1. `01_MLP_and_CNN.ipynb` — understand baseline models
2. `02_Hybrid_Neural_Network.ipynb` — run the hybrid architecture
3. `03_Vision_Transformer.ipynb` — run the ViT model

### Using your own data
The `CustomDataset` class in `utils/dataset.py` expects four directories:
```
data/
├── img_encoded/     # Encoded images (.png or .npy)
├── weights/         # Target weights (.npy)
├── traj_joint/      # Joint trajectory arrays (.npy)
└── traj_task/       # Task trajectory arrays (.npy)
```

---

## 📦 Data

The dataset used consists of:
- **Encoded images** — autoencoder-compressed visual scenes
- **Weights** — target prediction values (mean: 0.14, std: 1480.91)
- **Joint trajectories** — robot joint-space movements over time
- **Task trajectories** — end-effector movements in 3D space + orientation

Due to size constraints, the dataset is not included in this repository. The notebooks include placeholder synthetic data so all code can be run and verified immediately.

---

## 🗺️ Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

1. **Business Understanding** — define visual servoing as a deep learning regression task
2. **Data Understanding** — analyse encoded images, weights, and trajectory distributions
3. **Data Preparation** — normalisation, alignment, custom collation for variable-length trajectories
4. **Modelling** — implement CNN-MLP Hybrid and Vision Transformer in PyTorch
5. **Evaluation** — MSE, MAE, R², 10%-threshold accuracy
6. **Deployment** — discussion of real-world integration considerations

---

## 🔮 Future Work

- [ ] Expand dataset beyond 100 samples — critical for ViT performance
- [ ] Implement more sophisticated multi-modal fusion strategies (cross-attention)
- [ ] Explore task-specific loss functions beyond Euclidean distance
- [ ] Test on physical robotic hardware
- [ ] Investigate reinforcement learning and few-shot learning approaches
- [ ] Systematic hyperparameter optimisation

---

## 📄 Citation

If you use this work, please cite:
```
Kamble, C.J. (2024). Multi-Modal Deep Learning Architectures for Visual Servoing.
MSc Dissertation, University of Surrey.
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
