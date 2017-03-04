
/*
* 	@Author:  Danqi Chen
* 	@Email:  danqi@cs.stanford.edu
*	@Created:  2014-09-01
* 	@Last Modified:  2014-09-30
*/

package edu.stanford.nlp.parser.nndep;

import java.util.ArrayList;
import java.util.List;

/**
 * Defines a list of training / testing examples in multi-class classification setting.
 *
 * @author Danqi Chen
 */

public class AMRData {

  int cn;
  int an;
  int pn;
  //final int numFeatures, numLabels;
  Dataset conceptIDExamples;
  Dataset arcConnectExamples;
  Dataset pushPopExamples;

  AMRData() {
    cn = 0;
    an = 0;
    pn = 0;
    //this.numFeatures = numFeatures;
    //this.numLabels = numLabels;
    //conceptIDExamples = new ArrayList<>();
    //arcConnectExamples = new ArrayList<>();
    //pushPopExamples = new ArrayList<>();
  }

  void initExample(int numFeats, int numLabels, int type) {
    if (type == 0)
      conceptIDExamples = new Dataset(numFeats, numLabels);
    else if (type == 1)
      arcConnectExamples = new Dataset(numFeats, numLabels);
    else
      pushPopExamples = new Dataset(numFeats, numLabels);
  }

  public void addExample(List<Integer> feature, List<Integer> label, int type, int wordID) {
    //Example data = new Example(feature, label);
    if (type == 0) { //Concept id example
      cn += 1;
      //if(wordID == -1) {
      //  System.err.println("Shoot!");
      //  System.exit(1);
      //}
      conceptIDExamples.addExample(feature, label, wordID);
    }
    else if (type == 1) { //Connect arc labels
      an += 1;
      arcConnectExamples.addExample(feature, label, wordID);
    }
    else {
      pn += 1;
      pushPopExamples.addExample(feature, label, wordID);
    }
  }
}
