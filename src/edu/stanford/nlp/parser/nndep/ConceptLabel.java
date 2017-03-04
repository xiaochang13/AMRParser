package edu.stanford.nlp.parser.nndep;

import java.util.*;

import edu.stanford.nlp.ling.AnnotationLookup.KeyLookup;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Generics;


/**
 * A CoreLabel represents a single word with ancillary information
 * attached using CoreAnnotations.
 * A CoreLabel also provides convenient methods to access tags,
 * lemmas, etc. (if the proper annotations are set).
 * <p>
 * A CoreLabel is a Map from keys (which are Class objects) to values,
 * whose type is determined by the key.  That is, it is a heterogeneous
 * typesafe Map (see Josh Bloch, Effective Java, 2nd edition).
 * <p>
 * The CoreLabel class in particular bridges the gap between old-style JavaNLP
 * Labels and the new CoreMap infrastructure.  Instances of this class can be
 * used (almost) anywhere that the now-defunct FeatureLabel family could be
 * used.  This data structure is backed by an {@link ArrayCoreMap}.
 *
 * @author dramage
 * @author rafferty
 */
public class ConceptLabel{

  private static final long serialVersionUID = 2L;
  private String value;
  public boolean isVar;
  public boolean aligned;
  public List<Integer> alignments;
  public List<String> rels;
  public Map<Integer, String> relMap;
  public List<Integer> tails;
  public List<String> parRels;
  public List<Integer> parConcepts;
  //public

  public ConceptLabel() {
    value = null;
    alignments = new ArrayList<>();
    rels = new ArrayList<>();
    tails = new ArrayList<>();
    parRels = new ArrayList<>();
    parConcepts = new ArrayList<>();
    aligned = false;
  }

  public ConceptLabel(String label) {
    value = label;
    alignments = new ArrayList<>();
    rels = new ArrayList<>();
    relMap = new HashMap<>();
    tails = new ArrayList<>();
    parRels = new ArrayList<>();
    parConcepts = new ArrayList<>();
    aligned = false;
  }

  public final void setVar(boolean v) {
    isVar = v;
  }

  public String getRel(int index) {
    assert relMap.containsKey(index);
    return relMap.get(index);
  }

  public void buildRelMap() {
    int nRels = rels.size();
    for (int i = 0; i < nRels; i++) {
      int currIndex = tails.get(i);
      relMap.put(currIndex, rels.get(i));
    }
    int nPRels = parRels.size();
    for (int i = 0; i < nPRels; i++) {
      int currIndex = parConcepts.get(i);
      relMap.put(currIndex, parRels.get(i));
    }
  }

  public final void setValue(String value) {
    this.value = value;
  }

  public final String value() {
    return value;
  }

  public final void addWord(int wordIndex) {
    alignments.add(wordIndex);
  }
}
