package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.util.StringUtils;

import java.util.*;

/**
 * Represents a partial or complete dependency parse of a sentence, and
 * provides convenience methods for analyzing the parse.
 *
 * @author Xiaochang Peng
 */
class AMRGraph {

  int n;
  Integer root;
  List<ConceptLabel> concepts;
  String[] toks;
  List<Integer> dists;
  Map<Integer, Set<Integer>> headToTail;
  Map<Integer, Set<Integer>> tailToHead;
  Map<Integer, Integer> wordToConcept;

  //List<String>
  //List<Integer> head;
  //List<String> label;
  private int counter; //Count the current concept index

  public AMRGraph() {
    n = 0;
    concepts = new ArrayList<>();
    counter = 0;
    wordToConcept = new HashMap<>();
    toks = null;
    //head = new ArrayList<Integer>();
    //head.add(Config.NONEXIST);
    //label = new ArrayList<String>();
    //label.add(Config.UNKNOWN);
  }

  public AMRGraph(AMRGraph graph) {
    n = graph.n;
    concepts = new ArrayList<>(graph.concepts);
    counter = 0;
    //head = new ArrayList<Integer>(graph.head);
    //label = new ArrayList<String>(graph.label);
  }

  public void setRoot(int index) {
    root = index;
  }
  public void count() { counter += 1;}
  public boolean nextIsAligned() {
    assert counter < concepts.size();
    return concepts.get(counter).aligned;
  }

  public void setSentence(String[] toks) {
    this.toks = toks;
  }

  public int nextConceptId() {
    return counter;
  }
  //Not sure if this will result in a full cover though
  public void buildWordToIndices() {
    for (int i = 0; i < concepts.size(); i++) {
      ConceptLabel c = concepts.get(i);
      //System.out.println(c);
      //System.out.println(c.value());
      for (int wordID : c.alignments)
        wordToConcept.put(wordID, i);
    }
  }

  public void printGraph() {
    for (int i = 0; i < concepts.size(); i++) {
      ConceptLabel c = concepts.get(i);
      System.out.println("Current concept:" + c.value());
      c.buildRelMap();
      String relRep = "";
      for (int tail: c.tails) {
        relRep += c.relMap.get(tail);
        relRep += (":" + concepts.get(tail).value() + " ");
      }
      System.out.println("Tail concepts:" + relRep);
    }
  }

  public void buildEdgeMap() {
    headToTail = new HashMap<>();
    tailToHead = new HashMap<>();
    for (int i = 0; i < concepts.size(); i++) {
      ConceptLabel c = concepts.get(i);
      headToTail.put(i, new HashSet<>());
      for (int tail: c.tails) {
        headToTail.get(i).add(tail);
        if (!tailToHead.containsKey(tail))
          tailToHead.put(tail, new HashSet<>());
        tailToHead.get(tail).add(i);
      }
    }
  }

  public void add(ConceptLabel c) {
    concepts.add(c);
    n += 1;
  }

  //public void set(int k, int h, String l) {
  //  head.set(k, h);
  //  label.set(k, l);
  //}

  //public int getHead(int k) {
  //  if (k <= 0 || k > n)
  //    return Config.NONEXIST;
  //  else
  //    return head.get(k);
  //}

  //public String getLabel(int k) {
  //  if (k <= 0 || k > n)
  //    return Config.NULL;
  //  else
  //    return label.get(k);
  //}

  /**
   * Get the index of the node which is the root of the parse (i.e.,
   * that node which has the ROOT node as its head).
   */
  //public int getRoot() {
  //  for (int k = 1; k <= n; ++k)
  //    if (getHead(k) == 0)
  //      return k;
  //  return 0;
  //}

  /**
   * Check if this parse has only one root.
   */
  //public boolean isSingleRoot() {
  //  int roots = 0;
  //  for (int k = 1; k <= n; ++k)
  //    if (getHead(k) == 0)
  //      roots = roots + 1;
  //  return (roots == 1);
  //}

  //// check if the tree is legal, O(n)
  //public boolean isTree() {
  //  List<Integer> h = new ArrayList<Integer>();
  //  h.add(-1);
  //  for (int i = 1; i <= n; ++i) {
  //    if (getHead(i) < 0 || getHead(i) > n)
  //      return false;
  //    h.add(-1);
  //  }
  //  for (int i = 1; i <= n; ++i) {
  //    int k = i;
  //    while (k > 0) {
  //      if (h.get(k) >= 0 && h.get(k) < i) break;
  //      if (h.get(k) == i)
  //        return false;
  //      h.set(k, i);
  //      k = getHead(k);
  //    }
  //  }
  //  return true;
  //}

  //// check if the tree is projective, O(n^2)
  //public boolean isProjective() {
  //  if (!isTree())
  //    return false;
  //  counter = -1;
  //  return visitTree(0);
  //}

  //// Inner recursive function for checking projectivity of tree
  //private boolean visitTree(int w) {
  //  for (int i = 1; i < w; ++i)
  //    if (getHead(i) == w && visitTree(i) == false)
  //      return false;
  //  counter = counter + 1;
  //  if (w != counter)
  //    return false;
  //  for (int i = w + 1; i <= n; ++i)
  //    if (getHead(i) == w && visitTree(i) == false)
  //      return false;
  //  return true;
  //}

  //// TODO properly override equals, hashCode?
  //public boolean equal(DependencyTree t) {
  //  if (t.n != n)
  //    return false;
  //  for (int i = 1; i <= n; ++i) {
  //    if (getHead(i) != t.getHead(i))
  //      return false;
  //    if (!getLabel(i).equals(t.getLabel(i)))
  //      return false;
  //  }
  //  return true;
  //}

  //public void print() {
  //  for (int i = 1; i <= n; ++i)
  //    System.out.println(i + " " + getHead(i) + " " + getLabel(i));
  //  System.out.println();
  //}

}