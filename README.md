# ML Canvas — Interactive Machine Learning Laboratory

> An interactive laboratory & visualization workbench for exploring 51 machine learning algorithms in real-time right in the browser.

[![Build & Test](https://github.com/mostafizcse007/mlcanvas/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mostafizcse007/mlcanvas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ML Canvas is an educational workbench designed to build geometric and mathematical intuition for classical machine learning, deep learning, sequence models, and reinforcement learning. Every model executes client-side with dedicated Web Workers, responsive heatmaps, live loss curves, timeline scrubbing, and analytical 3-lens inspections (Intuition, Math, and Code snippets).

---

## 🌟 Interactive Stages & Architecture

- **Stage 01: Classical Convex & Linear Models**
  - Linear SVM, Kernel SVM (RBF/Polynomial), Logistic Regression, Lasso (L1), Ridge (L2), ElasticNet, Ordinary Least Squares, Polynomial Regression, Support Vector Regression (SVR), Gaussian Process Regression (GPR), Naive Bayes, Perceptron
- **Stage 02: Ensemble & Tree Partitioning**
  - Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM, Stacking Classifier/Regressor, Voting Classifier
- **Stage 03: Clustering & Manifold Learning**
  - K-Means, GMM (Gaussian Mixture Models), DBSCAN, Hierarchical Clustering, PCA, LDA, ICA, t-SNE, UMAP
- **Stage 04: Dense & Generative Neural Graph Lab**
  - Multi-Layer Perceptron (MLP with backpropagation & live neuron mini-activation heatmaps)
  - Deterministic Bottleneck Autoencoder
  - Variational Autoencoder ($\beta$-VAE with Gaussian uncertainty ellipses & reparameterization)
  - Generative Adversarial Network (GAN Minimax Arena with real-time vector field dynamics)
- **Stage 05: Convolutional Vision Architectures**
  - LeNet-5, AlexNet, VGG-16, ResNet, Vision Transformer (ViT patch projection & multi-head self-attention)
- **Stage 06: Sequence Modeling & Attention**
  - Recurrent Neural Networks (Vanilla RNN, LSTM with cell highway, GRU)
  - Scaled Dot-Product Attention & Multi-Head Self-Attention
  - Encoder-Decoder Transformer
- **Stage 07: Reinforcement Learning**
  - Tabular Q-Learning, Deep Q-Networks (DQN with experience replay & target networks), REINFORCE Policy Gradient on interactive gridworlds

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- `npm` or `pnpm` / `yarn`

### Installation

```bash
# Clone the repository
git clone https://github.com/mostafizcse007/mlcanvas.git
cd mlcanvas

# Install dependencies
npm install
```

### Running Locally

```bash
# Start Vite development server
npm run dev
```

Visit `http://localhost:5173/` in your browser.

### Testing & Verification

```bash
# Run unit and UI tests (Vitest)
npm run test

# Typecheck and build production bundle
npm run build
```

---

## 🛠️ Tech Stack

- **Framework**: React 19, TypeScript
- **Styling**: Tailwind CSS v4, Lucide Icons
- **Formula Rendering**: KaTeX ($LaTeX$)
- **Engine**: Dedicated Web Worker (`engine.worker.ts`) with zero UI thread freezing
- **Build Tooling**: Vite, Vitest, Testing Library

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
