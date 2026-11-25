import { AlgorithmPlugin, StageKind, TaskType } from '../types/algorithm';
import {
  STAGES,
  ALGORITHM_CATALOG_METADATA,
  resolveAlgorithmId,
  registerPlugin,
} from './catalog';
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

export { STAGES };

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

const STATIC_PLUGINS: Record<string, AlgorithmPlugin<any, any>> = {
  'linear-regression': linearRegressionPlugin,
  'polynomial-regression': polynomialRegressionPlugin,
  'ridge-l2': ridgeL2Plugin,
  'lasso-l1': lassoL1Plugin,
  'elastic-net': elasticNetPlugin,
  'logistic-regression': logisticRegressionPlugin,
  perceptron: perceptronPlugin,
  'linear-svm': linearSvmPlugin,
  'kernel-svm': kernelSvmPlugin,
  svr: svrPlugin,
  knn: knnPlugin,
  'naive-bayes': naiveBayesPlugin,
  'gaussian-process': gaussianProcessPlugin,
  'decision-tree': decisionTreePlugin,
  'random-forest': randomForestPlugin,
  bagging: baggingPlugin,
  adaboost: adaboostPlugin,
  'gradient-boosting': gradientBoostingPlugin,
  xgboost: xgboostPlugin,
  lightgbm: lightgbmPlugin,
  catboost: catboostPlugin,
  'voting-classifier': votingClassifierPlugin,
  stacking: stackingPlugin,
  'k-means': kMeansPlugin,
  'hierarchical-clustering': hierarchicalClusteringPlugin,
  dbscan: dbscanPlugin,
  gmm: gmmPlugin,
  pca: pcaPlugin,
  lda: ldaPlugin,
  ica: icaPlugin,
  't-sne': tsnePlugin,
  umap: umapPlugin,
  mlp: mlpPlugin,
  autoencoder: autoencoderPlugin,
  vae: vaePlugin,
  gan: ganPlugin,
  'cnn-lenet': cnnLenetPlugin,
  'cnn-alexnet': cnnAlexnetPlugin,
  'cnn-vgg': cnnVggPlugin,
  'cnn-resnet': cnnResnetPlugin,
  vit: vitPlugin,
  rnn: rnnPlugin,
  lstm: lstmPlugin,
  gru: gruPlugin,
  attention: attentionPlugin,
  'transformer-encoder': transformerEncoderPlugin,
  'transformer-decoder': transformerDecoderPlugin,
  'transformer-enc-dec': transformerEncDecPlugin,
  'q-learning': qLearningPlugin,
  dqn: dqnPlugin,
  'policy-gradient': reinforcePlugin,
};

for (const plugin of Object.values(STATIC_PLUGINS)) {
  registerPlugin(plugin);
}

// Complete 51 Algorithm Catalog across 7 Stages (eagerly populated for synchronous unit tests)
export const ALGORITHM_CATALOG: AlgorithmCatalogEntry[] = ALGORITHM_CATALOG_METADATA.map((meta) => ({
  id: meta.id,
  stageId: meta.stageId,
  name: meta.name,
  shortCode: meta.shortCode,
  taskType: meta.taskType,
  description: meta.description,
  isImplemented: meta.isImplemented,
  plugin: STATIC_PLUGINS[meta.id],
  loader: meta.loader,
}));

export function getPlugin(id: string): AlgorithmPlugin<any, any> | null {
  const canonical = resolveAlgorithmId(id);
  const entry = ALGORITHM_CATALOG.find((a) => a.id === canonical);
  return entry && entry.plugin ? entry.plugin : null;
}

export async function loadPlugin(id: string): Promise<AlgorithmPlugin<any, any> | null> {
  const plugin = getPlugin(id);
  if (plugin) return plugin;
  const canonical = resolveAlgorithmId(id);
  const entry = ALGORITHM_CATALOG.find((a) => a.id === canonical);
  if (entry && entry.loader) {
    const loaded = await entry.loader();
    entry.plugin = loaded;
    registerPlugin(loaded);
    return loaded;
  }
  return null;
}
