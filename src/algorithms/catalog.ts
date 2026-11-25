import { AlgorithmPlugin, ExecutionMode, StageInfo, StageKind, TaskType } from '../types/algorithm';

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

export interface AlgorithmCatalogMetadata {
  id: string;
  stageId: StageKind;
  name: string;
  shortCode: string;
  taskType: TaskType;
  description: string;
  isImplemented: boolean;
  executionMode?: ExecutionMode;
  loader: () => Promise<AlgorithmPlugin<any, any>>;
}

const ALIAS_TO_ID: Record<string, string> = {
  'svm-rbf': 'kernel-svm',
  'svm-poly': 'kernel-svm',
  kmeans: 'k-means',
  'gmm-em': 'gmm',
  tsne: 't-sne',
  'mlp-dense': 'mlp',
  ann: 'mlp',
  'auto-encoder': 'autoencoder',
  'scaled-dot-product-attention': 'attention',
  attn: 'attention',
  'multi-head-attention': 'transformer-encoder',
  reinforce: 'policy-gradient',
  'deep-q-network': 'dqn',
  qlearning: 'q-learning',
};

/**
 * Lightweight 51-algorithm catalog with dynamic code-split loaders so initial
 * page entry and worker boot only evaluate the active algorithm module.
 */
export const ALGORITHM_CATALOG_METADATA: AlgorithmCatalogMetadata[] = [
  // Stage 01: Classical Convex & Linear & Kernel (13 Fully Implemented Algorithms)
  {
    id: 'linear-regression',
    stageId: 'continuous-2d',
    name: 'Linear Regression (OLS / SGD)',
    shortCode: 'LinReg (OLS)',
    taskType: 'regression',
    description: 'Ordinary Least Squares & Stochastic Gradient Descent with residual error bars and Huber loss.',
    isImplemented: true,
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
    executionMode: 'phase-walkthrough',
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
    executionMode: 'closed-form',
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
    executionMode: 'closed-form',
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
    loader: () => import('./stage07/reinforce').then((m) => m.reinforcePlugin),
  },
];

const pluginCache = new Map<string, AlgorithmPlugin<any, any>>();
const inFlightLoads = new Map<string, Promise<AlgorithmPlugin<any, any> | null>>();

export function resolveAlgorithmId(id: string): string {
  return ALIAS_TO_ID[id] ?? id;
}

export function getCatalogEntry(id: string): AlgorithmCatalogMetadata | undefined {
  const canonical = resolveAlgorithmId(id);
  return ALGORITHM_CATALOG_METADATA.find((entry) => entry.id === canonical);
}

export function registerPlugin(plugin: AlgorithmPlugin<any, any>): void {
  pluginCache.set(plugin.id, plugin);
}

export function getLoadedPlugin(id: string): AlgorithmPlugin<any, any> | null {
  const canonical = resolveAlgorithmId(id);
  return pluginCache.get(canonical) ?? null;
}

export async function loadPlugin(id: string): Promise<AlgorithmPlugin<any, any> | null> {
  const canonical = resolveAlgorithmId(id);
  const cached = pluginCache.get(canonical);
  if (cached) return cached;

  const existing = inFlightLoads.get(canonical);
  if (existing) return existing;

  const entry = ALGORITHM_CATALOG_METADATA.find((a) => a.id === canonical);
  if (!entry) return null;

  const promise = entry.loader().then((loaded) => {
    pluginCache.set(canonical, loaded);
    inFlightLoads.delete(canonical);
    return loaded;
  });
  inFlightLoads.set(canonical, promise);
  return promise;
}
