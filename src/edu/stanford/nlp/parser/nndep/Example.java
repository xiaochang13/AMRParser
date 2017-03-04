package edu.stanford.nlp.parser.nndep;

import java.util.List;

/**
 * @author Christopher Manning
 */
class Example {

  private final List<Integer> feature;
  private final List<Integer> label;
  private final int wordID;

  public Example(List<Integer> feature, List<Integer> label, int wordID) {
    this.feature = feature;
    this.label = label;
    this.wordID = wordID;
  }

  public List<Integer> getFeature() {
    return feature;
  }

  public List<Integer> getLabel() {
    return label;
  }

  public int getWordID() {return wordID;}
}
