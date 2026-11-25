import { AlgorithmPlugin, StageKind, StageInfo, TaskType } from '../types/algorithm';
import { linearSvmPlugin } from './stage01/linearSvm';
import { lassoL1Plugin } from './stage01/lassoL1';
import { linearRegressionPlugin } from './stage01/linearRegression';
import { polynomialRegressionPlugin } from './stage01/polynomialRegression';
import { ridgeL2Plugin } from './stage01/ridgeL2';
import { elasticNetPlugin } from './stage01/elasticNet';
import { logisticRegressionPlugin } from './stage01/logisticRegression';
import { perceptronPlugin } from './stage01/perceptron';
import { kernelSvmPlugin } from './stage01/kernelSvm';
import { svrPlugin } from './stage01/svr';
import { knnPlugin } from './stage01/knn';
import { naiveBayesPlugin } from './stage01/naiveBayes';
import { gaussianProcessPlugin } from './stage01/gaussianProcess';
import { decisionTreePlugin } from './stage02/decisionTree';
import { randomForestPlugin } from './stage02/randomForest';
import { baggingPlugin } from './stage02/bagging';
import { adaboostPlugin } from './stage02/adaboost';
import { gradientBoostingPlugin } from './stage02/gradientBoosting';
import { xgboostPlugin } from './stage02/xgboost';
import { lightgbmPlugin } from './stage02/lightgbm';
import { catboostPlugin } from './stage02/catboost';
import { votingClassifierPlugin } from './stage02/votingClassifier';
import { stackingPlugin } from './stage02/stacking';
import { kMeansPlugin } from './stage03/kMeans';
import { hierarchicalClusteringPlugin } from './stage03/hierarchicalClustering';
import { dbscanPlugin } from './stage03/dbscan';
import { gmmPlugin } from './stage03/gmm';
import { pcaPlugin } from './stage03/pca';
import { ldaPlugin } from './stage03/lda';
import { icaPlugin } from './stage03/ica';
import { tsnePlugin } from './stage03/tsne';
import { umapPlugin } from './stage03/umap';
import { mlpPlugin } from './stage04/mlp';
import { autoencoderPlugin } from './stage04/autoencoder';
import { vaePlugin } from './stage04/vae';
import { ganPlugin } from './stage04/gan';
import { cnnLenetPlugin } from './stage05/cnnLenet';
import { cnnAlexnetPlugin } from './stage05/cnnAlexnet';
import { cnnVggPlugin } from './stage05/cnnVgg';
import { cnnResnetPlugin } from './stage05/cnnResnet';
import { vitPlugin } from './stage05/vit';
import { rnnPlugin } from './stage06/rnn';
import { lstmPlugin } from './stage06/lstm';
import { gruPlugin } from './stage06/gru';
import { attentionPlugin } from './stage06/attention';
import { transformerEncoderPlugin } from './stage06/transformerEncoder';
import { transformerDecoderPlugin } from './stage06/transformerDecoder';
import { transformerEncDecPlugin } from './stage06/transformerEncDec';
import { qLearningPlugin } from './stage07/qLearning';
import { dqnPlugin } from './stage07/dqn';
import { reinforcePlugin } from './stage07/reinforce';

export const STAGES: StageInfo[] = [
  {
    id: 'continuous-2d',
    index: 1,
    name: 'Stage 01: Classical Convex & Linear',
    shortName: 'Convex & Linear',
    description: 'Separating hyperplanes, margin maximization, subgradient optimization, L1/L2 penalties.',
  },
  {
    id: 'tree-partition',
    index: 2,
    name: 'Stage 02: Partition, Ensemble & Boosting Arena',
    shortName: 'Ensembles & Trees',
    description: 'Axis-aligned orthogonal partitions, bootstrap aggregation, residual boosting.',
  },
  {
    id: 'unsupervised-2d',
    index: 3,
    name: 'Stage 03: Unsupervised & Dimensionality',
    shortName: 'Unsupervised',
    description: 'Expectation-maximization, manifold unfolding, eigenvector projections.',
  },
  {
    id: 'neural-graph',
    index: 4,
    name: 'Stage 04: Neural Foundations & MLPs',
    shortName: 'Neural Foundations',
    description: 'Affine compositions, non-linear activation boundaries, backpropagation flow.',
  },
  {
    id: 'vision-grid',
    index: 5,
    name: 'Stage 05: Deep Architectures & Vision',
    shortName: 'Deep Architectures',
    description: 'Spatial equivariance, convolutional kernels, residual skip connections, generative priors.',
  },
  {
    id: 'sequence-attn',
    index: 6,
    name: 'Stage 06: Attention, Transformers & Modern Generative',
    shortName: 'Attention & Generative',
    description: 'Query-key-value routing, softmax distributions, vector fields, and generative reverse flows.',
  },
  {
    id: 'rl-gridworld',
    index: 7,
    name: 'Stage 07: Reinforcement Learning & Policy Control',
    shortName: 'Reinforcement Learning',
    description: 'Policy gradient, value iteration, Q-learning, and Bellman optimality.',
  },
];

export interface AlgorithmCatalogEntry {
  id: string;
  stageId: StageKind;
  name: string;
  shortCode: string;
  taskType: TaskType;
  description: string;
  isImplemented: boolean;
  plugin?: AlgorithmPlugin<any, any>;
  loader?: () => Promise<AlgorithmPlugin<any, any>>;
}

// Complete 51 Algorithm Catalog across 7 Stages
export const ALGORITHM_CATALOG: AlgorithmCatalogEntry[] = [
  // Stage 01: Classical Convex & Linear & Kernel (13 Fully Implemented Algorithms)
  {
    id: 'linear-regression',
    stageId: 'continuous-2d',
    name: 'Linear Regression (OLS / SGD)',
    shortCode: 'LinReg (OLS)',
    taskType: 'regression',
    description: 'Ordinary Least Squares & Stochastic Gradient Descent with residual error bars and Huber loss.',
    isImplemented: true,
    plugin: linearRegressionPlugin,
    loader: () => import('./stage01/linearRegression').then((m) => m.linearRegressionPlugin),
  },
  {
    id: 'polynomial-regression',
    stageId: 'continuous-2d',
    name: 'Polynomial Feature Regression',
    shortCode: 'PolyReg',
    taskType: 'regression',
    description: 'Vandermonde basis expansion with Runge oscillation detection and Tikhonov stabilizer.',
    isImplemented: true,
    plugin: polynomialRegressionPlugin,
    loader: () => import('./stage01/polynomialRegression').then((m) => m.polynomialRegressionPlugin),
  },
  {
    id: 'ridge-l2',
    stageId: 'continuous-2d',
    name: 'Ridge L2 Regularized Regression',
    shortCode: 'Ridge L2',
    taskType: 'regression',
    description: 'Tikhonov L2 quadratic penalty with analytical Gram stabilization and circular weight-space inset.',
    isImplemented: true,
    plugin: ridgeL2Plugin,
    loader: () => import('./stage01/ridgeL2').then((m) => m.ridgeL2Plugin),
  },
  {
    id: 'lasso-l1',
    stageId: 'continuous-2d',
    name: 'Lasso L1 Sparse Regression',
    shortCode: 'Lasso L1',
    taskType: 'regression',
    description: 'L1-regularized sparse regression via coordinate descent with soft-thresholding and weight-space inset.',
    isImplemented: true,
    plugin: lassoL1Plugin,
    loader: () => import('./stage01/lassoL1').then((m) => m.lassoL1Plugin),
  },
  {
    id: 'elastic-net',
    stageId: 'continuous-2d',
    name: 'Elastic Net Regularization',
    shortCode: 'ElasticNet',
    taskType: 'regression',
    description: 'Combined L1 and L2 penalties balancing sparsity and grouped collinear feature selection.',
    isImplemented: true,
    plugin: elasticNetPlugin,
    loader: () => import('./stage01/elasticNet').then((m) => m.elasticNetPlugin),
  },
  {
    id: 'logistic-regression',
    stageId: 'continuous-2d',
    name: 'Logistic Regression (Sigmoid & Softmax)',
    shortCode: 'LogReg',
    taskType: 'classification',
    description: 'Bernoulli likelihood maximization with calibrated probabilities and live iso-probability contour shifts.',
    isImplemented: true,
    plugin: logisticRegressionPlugin,
    loader: () => import('./stage01/logisticRegression').then((m) => m.logisticRegressionPlugin),
  },
  {
    id: 'perceptron',
    stageId: 'continuous-2d',
    name: 'Rosenblatt Perceptron',
    shortCode: 'Perceptron',
    taskType: 'classification',
    description: 'Original 1958 error-correcting learning algorithm for linearly separable half-spaces.',
    isImplemented: true,
    plugin: perceptronPlugin,
    loader: () => import('./stage01/perceptron').then((m) => m.perceptronPlugin),
  },
  {
    id: 'linear-svm',
    stageId: 'continuous-2d',
    name: 'Linear Support Vector Machine',
    shortCode: 'Linear SVM',
    taskType: 'classification',
    description: 'Primal subgradient Pegasos solver with exact margin width computation and support vector detection.',
    isImplemented: true,
    plugin: linearSvmPlugin,
    loader: () => import('./stage01/linearSvm').then((m) => m.linearSvmPlugin),
  },
  {
    id: 'kernel-svm',
    stageId: 'continuous-2d',
    name: 'Support Vector Machine (Kernel RBF & Poly)',
    shortCode: 'Kernel SVM',
    taskType: 'classification',
    description: 'Dual representation in Reproducing Kernel Hilbert Space (RKHS) with active support vector isolation.',
    isImplemented: true,
    plugin: kernelSvmPlugin,
    loader: () => import('./stage01/kernelSvm').then((m) => m.kernelSvmPlugin),
  },
  {
    id: 'svr',
    stageId: 'continuous-2d',
    name: 'Support Vector Regression (ε-Insensitive)',
    shortCode: 'SVR',
    taskType: 'regression',
    description: 'Vapnik ε-insensitive loss corridor with zero penalty inside the tube and sparse support vectors.',
    isImplemented: true,
    plugin: svrPlugin,
    loader: () => import('./stage01/svr').then((m) => m.svrPlugin),
  },
  {
    id: 'knn',
    stageId: 'continuous-2d',
    name: 'k-Nearest Neighbors (Lazy Instance Learner)',
    shortCode: 'k-NN',
    taskType: 'classification',
    description: 'Non-parametric local metric voting with interactive probe rays, distance weighting, and metric distortion.',
    isImplemented: true,
    plugin: knnPlugin,
    loader: () => import('./stage01/knn').then((m) => m.knnPlugin),
  },
  {
    id: 'naive-bayes',
    stageId: 'continuous-2d',
    name: 'Naive Bayes Classifier (Gaussian & Discrete)',
    shortCode: 'NaiveBayes',
    taskType: 'classification',
    description: 'Conditional independence generative model with marginal axis-aligned bell curves and discrete grid quantization.',
    isImplemented: true,
    plugin: naiveBayesPlugin,
    loader: () => import('./stage01/naiveBayes').then((m) => m.naiveBayesPlugin),
  },
  {
    id: 'gaussian-process',
    stageId: 'continuous-2d',
    name: 'Gaussian Process Regression (Kriging & GPR)',
    shortCode: 'GaussProcess',
    taskType: 'regression',
    description: 'Bayesian non-parametric inference with exact covariance inversion and epistemic ±2σ uncertainty envelope.',
    isImplemented: true,
    plugin: gaussianProcessPlugin,
    loader: () => import('./stage01/gaussianProcess').then((m) => m.gaussianProcessPlugin),
  },

  // Stage 02: Partition, Ensemble & Boosting Arena (10 Fully Implemented Algorithms)
  {
    id: 'decision-tree',
    stageId: 'tree-partition',
    name: 'Decision Tree (CART Classifier & Regressor)',
    shortCode: 'CART Tree',
    taskType: 'classification',
    description: 'Recursive greedy binary feature split with exact Gini reduction and cost-complexity pruning.',
    isImplemented: true,
    plugin: decisionTreePlugin,
    loader: () => import('./stage02/decisionTree').then((m) => m.decisionTreePlugin),
  },
  {
    id: 'random-forest',
    stageId: 'tree-partition',
    name: 'Random Forest Ensemble',
    shortCode: 'RandForest',
    taskType: 'classification',
    description: 'Bagged decision trees with randomized subspace feature selection and OOB error estimation.',
    isImplemented: true,
    plugin: randomForestPlugin,
    loader: () => import('./stage02/randomForest').then((m) => m.randomForestPlugin),
  },
  {
    id: 'bagging',
    stageId: 'tree-partition',
    name: 'Bagging (Bootstrap Aggregation)',
    shortCode: 'Bagging',
    taskType: 'classification',
    description: 'Bootstrap Aggregating with sample multiplicity badges and Out-of-Bag (OOB) validation.',
    isImplemented: true,
    plugin: baggingPlugin,
    loader: () => import('./stage02/bagging').then((m) => m.baggingPlugin),
  },
  {
    id: 'adaboost',
    stageId: 'tree-partition',
    name: 'AdaBoost (Adaptive Boosting — SAMME)',
    shortCode: 'AdaBoost',
    taskType: 'classification',
    description: 'Sequential sample reweighting with dynamic point radius scaling focusing on hard instances.',
    isImplemented: true,
    plugin: adaboostPlugin,
    loader: () => import('./stage02/adaboost').then((m) => m.adaboostPlugin),
  },
  {
    id: 'gradient-boosting',
    stageId: 'tree-partition',
    name: 'Gradient Boosted Decision Trees (Vanilla GBM)',
    shortCode: 'GBDT',
    taskType: 'classification',
    description: 'Functional gradient descent fitting shallow regression trees to pseudo-residuals with shrinkage.',
    isImplemented: true,
    plugin: gradientBoostingPlugin,
    loader: () => import('./stage02/gradientBoosting').then((m) => m.gradientBoostingPlugin),
  },
  {
    id: 'xgboost',
    stageId: 'tree-partition',
    name: 'XGBoost (Extreme Gradient Boosting)',
    shortCode: 'XGBoost',
    taskType: 'classification',
    description: 'Second-order Taylor expansion boosting with Newton-Raphson similarity and level-wise growth.',
    isImplemented: true,
    plugin: xgboostPlugin,
    loader: () => import('./stage02/xgboost').then((m) => m.xgboostPlugin),
  },
  {
    id: 'lightgbm',
    stageId: 'tree-partition',
    name: 'LightGBM (Leaf-Wise Best-First Boosting)',
    shortCode: 'LightGBM',
    taskType: 'classification',
    description: 'Fast histogram-binned gradient boosting with leaf-wise best-first asymmetric tree expansion.',
    isImplemented: true,
    plugin: lightgbmPlugin,
    loader: () => import('./stage02/lightgbm').then((m) => m.lightgbmPlugin),
  },
  {
    id: 'catboost',
    stageId: 'tree-partition',
    name: 'CatBoost (Symmetric Oblivious Trees)',
    shortCode: 'CatBoost',
    taskType: 'classification',
    description: 'Symmetric oblivious decision trees sharing split criteria per depth, forming a full-span plaid grid.',
    isImplemented: true,
    plugin: catboostPlugin,
    loader: () => import('./stage02/catboost').then((m) => m.catboostPlugin),
  },
  {
    id: 'voting-classifier',
    stageId: 'tree-partition',
    name: 'Voting Classifier (Hard vs. Soft Ensemble)',
    shortCode: 'Voting',
    taskType: 'classification',
    description: 'Combines heterogeneous base classifiers (Linear, Tree, KNN) via weighted majority or probability consensus.',
    isImplemented: true,
    plugin: votingClassifierPlugin,
    loader: () => import('./stage02/votingClassifier').then((m) => m.votingClassifierPlugin),
  },
  {
    id: 'stacking',
    stageId: 'tree-partition',
    name: 'Stacked Generalization (Stacking)',
    shortCode: 'Stacking',
    taskType: 'classification',
    description: 'Level-0 base model prediction projection into meta-probability space fitted with a Level-1 meta-classifier.',
    isImplemented: true,
    plugin: stackingPlugin,
    loader: () => import('./stage02/stacking').then((m) => m.stackingPlugin),
  },

  // Stage 03: Unsupervised Clustering & Manifold Projection Lab (9 Fully Implemented Algorithms)
  {
    id: 'k-means',
    stageId: 'unsupervised-2d',
    name: "k-Means (Lloyd's Algorithm & K-Means++)",
    shortCode: 'k-Means',
    taskType: 'clustering',
    description: 'Alternating Voronoi partition assignment and center-of-mass updates with draggable centroids.',
    isImplemented: true,
    plugin: kMeansPlugin,
    loader: () => import('./stage03/kMeans').then((m) => m.kMeansPlugin),
  },
  {
    id: 'hierarchical-clustering',
    stageId: 'unsupervised-2d',
    name: 'Agglomerative Hierarchical Clustering',
    shortCode: 'Hierarchical',
    taskType: 'clustering',
    description: 'Bottom-up cluster merging with live interactive dendrogram and horizontal cut-height slider.',
    isImplemented: true,
    plugin: hierarchicalClusteringPlugin,
    loader: () => import('./stage03/hierarchicalClustering').then((m) => m.hierarchicalClusteringPlugin),
  },
  {
    id: 'dbscan',
    stageId: 'unsupervised-2d',
    name: 'DBSCAN Density-Based Clustering',
    shortCode: 'DBSCAN',
    taskType: 'clustering',
    description: 'Epsilon-neighborhood density connected components with automatic outlier noise rejection.',
    isImplemented: true,
    plugin: dbscanPlugin,
    loader: () => import('./stage03/dbscan').then((m) => m.dbscanPlugin),
  },
  {
    id: 'gmm',
    stageId: 'unsupervised-2d',
    name: 'Gaussian Mixture Models (EM Algorithm)',
    shortCode: 'GMM (EM)',
    taskType: 'density-estimation',
    description: 'Soft cluster responsibilities with full covariance ellipsoid estimation and BIC evaluation.',
    isImplemented: true,
    plugin: gmmPlugin,
    loader: () => import('./stage03/gmm').then((m) => m.gmmPlugin),
  },
  {
    id: 'pca',
    stageId: 'unsupervised-2d',
    name: 'Principal Component Analysis (PCA)',
    shortCode: 'PCA',
    taskType: 'dimensionality-reduction',
    description: 'Orthogonal variance-maximizing eigenvectors with orthogonal drop-lines and scree plot.',
    isImplemented: true,
    plugin: pcaPlugin,
    loader: () => import('./stage03/pca').then((m) => m.pcaPlugin),
  },
  {
    id: 'lda',
    stageId: 'unsupervised-2d',
    name: "Linear Discriminant Analysis (Fisher's Criterion)",
    shortCode: 'LDA',
    taskType: 'dimensionality-reduction',
    description: 'Rayleigh quotient maximization of between-class scatter over within-class scatter with PCA ghost comparison.',
    isImplemented: true,
    plugin: ldaPlugin,
    loader: () => import('./stage03/lda').then((m) => m.ldaPlugin),
  },
  {
    id: 'ica',
    stageId: 'unsupervised-2d',
    name: 'Independent Component Analysis (FastICA)',
    shortCode: 'FastICA',
    taskType: 'dimensionality-reduction',
    description: 'Fixed-point negentropy maximization resolving non-orthogonal source distribution edges.',
    isImplemented: true,
    plugin: icaPlugin,
    loader: () => import('./stage03/ica').then((m) => m.icaPlugin),
  },
  {
    id: 't-sne',
    stageId: 'unsupervised-2d',
    name: 't-SNE Manifold Projection',
    shortCode: 't-SNE',
    taskType: 'dimensionality-reduction',
    description: 'Student-t affinity matching in low dimensions with spring force vectors and KL divergence telemetry.',
    isImplemented: true,
    plugin: tsnePlugin,
    loader: () => import('./stage03/tsne').then((m) => m.tsnePlugin),
  },
  {
    id: 'umap',
    stageId: 'unsupervised-2d',
    name: 'UMAP (Uniform Manifold Approximation)',
    shortCode: 'UMAP',
    taskType: 'dimensionality-reduction',
    description: 'Fuzzy simplicial set graph optimization balancing local and global manifold topology.',
    isImplemented: true,
    plugin: umapPlugin,
    loader: () => import('./stage03/umap').then((m) => m.umapPlugin),
  },

  // Stage 04: Neural Foundations & Dense / Generative Graph Lab (4 Fully Implemented Algorithms)
  {
    id: 'mlp',
    stageId: 'neural-graph',
    name: 'Multi-Layer Perceptron (Deep Feedforward)',
    shortCode: 'MLP',
    taskType: 'classification',
    description: 'Exact backpropagation chain rule with live per-neuron 24x24 activation heatmaps and adaptive optimizers.',
    isImplemented: true,
    plugin: mlpPlugin,
    loader: () => import('./stage04/mlp').then((m) => m.mlpPlugin),
  },
  {
    id: 'autoencoder',
    stageId: 'neural-graph',
    name: 'Deterministic Bottleneck Autoencoder',
    shortCode: 'Autoencoder',
    taskType: 'dimensionality-reduction',
    description: 'Non-linear low-dimensional bottleneck compression with live latent cursor probe and PCA equivalence validation.',
    isImplemented: true,
    plugin: autoencoderPlugin,
    loader: () => import('./stage04/autoencoder').then((m) => m.autoencoderPlugin),
  },
  {
    id: 'vae',
    stageId: 'neural-graph',
    name: 'Variational Autoencoder (β-VAE)',
    shortCode: 'VAE',
    taskType: 'dimensionality-reduction',
    description: 'Probabilistic Gaussian latent space with live 1σ uncertainty ellipses and continuous ELBO balance.',
    isImplemented: true,
    plugin: vaePlugin,
    loader: () => import('./stage04/vae').then((m) => m.vaePlugin),
  },
  {
    id: 'gan',
    stageId: 'neural-graph',
    name: 'Generative Adversarial Network (Minimax Arena)',
    shortCode: 'GAN',
    taskType: 'density-estimation',
    description: 'Minimax zero-sum game with discriminator probability contours, generator pull vectors, and latent noise warp.',
    isImplemented: true,
    plugin: ganPlugin,
    loader: () => import('./stage04/gan').then((m) => m.ganPlugin),
  },

  // Stage 05: Deep Architectures & Vision (5 Implemented Algorithms)
  {
    id: 'cnn-lenet',
    stageId: 'vision-grid',
    name: 'LeNet-5 (Classical Conv & Subsampling)',
    shortCode: 'LeNet-5',
    taskType: 'classification',
    description: 'Classical 5x5 convolutions with subsampling pooling demonstrating translation invariance.',
    isImplemented: true,
    plugin: cnnLenetPlugin,
    loader: () => import('./stage05/cnnLenet').then((m) => m.cnnLenetPlugin),
  },
  {
    id: 'cnn-alexnet',
    stageId: 'vision-grid',
    name: 'AlexNet (Large Filters, ReLU & Dropout)',
    shortCode: 'AlexNet',
    taskType: 'classification',
    description: 'Multi-scale large 1st-layer filters, non-saturating ReLU activations, and Dropout regularization.',
    isImplemented: true,
    plugin: cnnAlexnetPlugin,
    loader: () => import('./stage05/cnnAlexnet').then((m) => m.cnnAlexnetPlugin),
  },
  {
    id: 'cnn-vgg',
    stageId: 'vision-grid',
    name: 'VGG (Deep Stacked 3x3 Receptive Fields)',
    shortCode: 'VGG',
    taskType: 'classification',
    description: 'Deep stacks of 3x3 convolutions expanding effective receptive fields with fewer parameters.',
    isImplemented: true,
    plugin: cnnVggPlugin,
    loader: () => import('./stage05/cnnVgg').then((m) => m.cnnVggPlugin),
  },
  {
    id: 'cnn-resnet',
    stageId: 'vision-grid',
    name: 'ResNet (Residual Highway H(x) = F(x) + x)',
    shortCode: 'ResNet',
    taskType: 'classification',
    description: 'Identity shortcut connections preserving backward gradient flow across deep residual stacks.',
    isImplemented: true,
    plugin: cnnResnetPlugin,
    loader: () => import('./stage05/cnnResnet').then((m) => m.cnnResnetPlugin),
  },
  {
    id: 'vit',
    stageId: 'vision-grid',
    name: 'ViT (Vision Transformer & Global Self-Attention)',
    shortCode: 'ViT',
    taskType: 'classification',
    description: 'Patch tokenization, 2D positional embeddings, and global patch-to-patch self-attention routing.',
    isImplemented: true,
    plugin: vitPlugin,
    loader: () => import('./stage05/vit').then((m) => m.vitPlugin),
  },

  // Stage 06: Sequence Gates & Attention Arena (7 Fully Implemented Algorithms)
  {
    id: 'rnn',
    stageId: 'sequence-attn',
    name: 'Vanilla Elman Recurrent Neural Network',
    shortCode: 'Vanilla RNN',
    taskType: 'classification',
    description: 'Unrolled recurrent chain h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) illustrating BPTT vanishing & exploding gradient bottlenecks.',
    isImplemented: true,
    plugin: rnnPlugin,
    loader: () => import('./stage06/rnn').then((m) => m.rnnPlugin),
  },
  {
    id: 'lstm',
    stageId: 'sequence-attn',
    name: 'Long Short-Term Memory Network',
    shortCode: 'LSTM',
    taskType: 'classification',
    description: 'Additive cell state conveyor belt C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t preserving backward gradient flow across long sequence horizons.',
    isImplemented: true,
    plugin: lstmPlugin,
    loader: () => import('./stage06/lstm').then((m) => m.lstmPlugin),
  },
  {
    id: 'gru',
    stageId: 'sequence-attn',
    name: 'Gated Recurrent Unit',
    shortCode: 'GRU',
    taskType: 'classification',
    description: 'Coupled update/reset gates balancing interpolation h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t without a separate cell state vector.',
    isImplemented: true,
    plugin: gruPlugin,
    loader: () => import('./stage06/gru').then((m) => m.gruPlugin),
  },
  {
    id: 'attention',
    stageId: 'sequence-attn',
    name: 'Scaled Dot-Product Attention',
    shortCode: 'SDP Attention',
    taskType: 'classification',
    description: 'Softmax(Q K^T / √d_k) V token routing mechanism proving direct O(1) path length and 1/√d_k gradient stabilization.',
    isImplemented: true,
    plugin: attentionPlugin,
    loader: () => import('./stage06/attention').then((m) => m.attentionPlugin),
  },
  {
    id: 'transformer-encoder',
    stageId: 'sequence-attn',
    name: 'Transformer Encoder Layer',
    shortCode: 'Encoder (BERT)',
    taskType: 'classification',
    description: 'Bidirectional Multi-Head Self-Attention + Residual LayerNorm + Sinusoidal Positional Encoding.',
    isImplemented: true,
    plugin: transformerEncoderPlugin,
    loader: () => import('./stage06/transformerEncoder').then((m) => m.transformerEncoderPlugin),
  },
  {
    id: 'transformer-decoder',
    stageId: 'sequence-attn',
    name: 'Autoregressive Causal Transformer Decoder',
    shortCode: 'Decoder (GPT)',
    taskType: 'classification',
    description: 'Autoregressive generation with strictly lower-triangular causal self-attention mask preventing future token look-ahead leaks.',
    isImplemented: true,
    plugin: transformerDecoderPlugin,
    loader: () => import('./stage06/transformerDecoder').then((m) => m.transformerDecoderPlugin),
  },
  {
    id: 'transformer-enc-dec',
    stageId: 'sequence-attn',
    name: 'Sequence-to-Sequence Encoder-Decoder',
    shortCode: 'Enc-Dec (T5)',
    taskType: 'classification',
    description: 'Full dual-tower Seq2Seq model with bidirectional source encoder, causal target decoder, and cross-attention alignment.',
    isImplemented: true,
    plugin: transformerEncDecPlugin,
    loader: () => import('./stage06/transformerEncDec').then((m) => m.transformerEncDecPlugin),
  },

  // Stage 07: Reinforcement Learning & Policy Control (3 Algorithms)
  {
    id: 'q-learning',
    stageId: 'rl-gridworld',
    name: 'Tabular Q-Learning (Off-Policy TD Control)',
    shortCode: 'Q-Learning',
    taskType: 'reinforcement-learning',
    description: 'Tabular off-policy Bellman optimality backups updating Q(s, a) directly toward Q* without environment dynamics.',
    isImplemented: true,
    plugin: qLearningPlugin,
    loader: () => import('./stage07/qLearning').then((m) => m.qLearningPlugin),
  },
  {
    id: 'dqn',
    stageId: 'rl-gridworld',
    name: 'Deep Q-Network (Function Approx & Replay)',
    shortCode: 'DQN',
    taskType: 'reinforcement-learning',
    description: 'Neural function approximation with circular Experience Replay ring buffer and frozen Target Network θ⁻.',
    isImplemented: true,
    plugin: dqnPlugin,
    loader: () => import('./stage07/dqn').then((m) => m.dqnPlugin),
  },
  {
    id: 'policy-gradient',
    stageId: 'rl-gridworld',
    name: 'REINFORCE (Monte-Carlo Policy Gradient)',
    shortCode: 'REINFORCE',
    taskType: 'reinforcement-learning',
    description: 'Likelihood-ratio gradient ascent on policy probabilities π_θ(a|s) with baseline subtraction and entropy regularization.',
    isImplemented: true,
    plugin: reinforcePlugin,
    loader: () => import('./stage07/reinforce').then((m) => m.reinforcePlugin),
  },
];

export function getPlugin(id: string): AlgorithmPlugin<any, any> | null {
  if (id === 'svm-rbf' || id === 'svm-poly') return kernelSvmPlugin;
  if (id === 'kmeans') return kMeansPlugin;
  if (id === 'gmm-em') return gmmPlugin;
  if (id === 'tsne') return tsnePlugin;
  if (id === 'mlp-dense' || id === 'ann') return mlpPlugin;
  if (id === 'auto-encoder') return autoencoderPlugin;
  if (id === 'scaled-dot-product-attention' || id === 'attn') return attentionPlugin;
  if (id === 'multi-head-attention') return transformerEncoderPlugin;
  if (id === 'reinforce') return reinforcePlugin;
  if (id === 'deep-q-network') return dqnPlugin;
  if (id === 'qlearning') return qLearningPlugin;
  const entry = ALGORITHM_CATALOG.find((a) => a.id === id);
  return entry && entry.plugin ? entry.plugin : null;
}

export async function loadPlugin(id: string): Promise<AlgorithmPlugin<any, any> | null> {
  const plugin = getPlugin(id);
  if (plugin) return plugin;
  const entry = ALGORITHM_CATALOG.find((a) => a.id === id);
  if (entry && entry.loader) {
    const loaded = await entry.loader();
    entry.plugin = loaded;
    return loaded;
  }
  return null;
}
