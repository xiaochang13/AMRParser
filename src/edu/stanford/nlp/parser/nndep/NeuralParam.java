package edu.stanford.nlp.parser.nndep;

import java.util.Random;

public class NeuralParam {
  double[][] E;
  double[][] W1;
  double[] b1;
  double[][] W2;
  int nTokens;

  NeuralParam(int numEmbeddings, int embeddingSize, int numInput, int hiddenSize, int numTransitions, int numTokens) {
    E = new double[numEmbeddings][embeddingSize];
    int inputDim = numInput * embeddingSize;
    W1 = new double[hiddenSize][inputDim];
    b1 = new double[hiddenSize];
    W2 = new double[numTransitions][hiddenSize];
    nTokens = numTokens;
  }

  void randomInitialize(double initRange) {
    Random random = Util.getRandom();
    for (int i = 0; i < W1.length; ++i)
      for (int j = 0; j < W1[i].length; ++j)
        W1[i][j] = random.nextDouble() * 2 * initRange - initRange;

    for (int i = 0; i < b1.length; ++i)
      b1[i] = random.nextDouble() * 2 * initRange - initRange;

    for (int i = 0; i < W2.length; ++i)
      for (int j = 0; j < W2[i].length; ++j)
        W2[i][j] = random.nextDouble() * 2 * initRange - initRange;
  }

  void setEmbedding(int i, int j, double value) {
    E[i][j] = value;
  }
}