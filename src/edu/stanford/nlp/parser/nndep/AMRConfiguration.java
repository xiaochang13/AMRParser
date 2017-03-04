
package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.util.*;

/**
 * Describe the current configuration of a parser (i.e., parser state).
 *
 * This class uses an indexing scheme where an index of zero refers to
 * the ROOT node and actual word indices begin at one.
 *
 * @author Danqi Chen
 */
public class AMRConfiguration {

  final List<Pair<Integer, Pair<Integer, Integer>>> stack;
  final List<Integer> buffer;
  final List<Pair<Integer, Integer>> cache;
  final int cacheSize;
  int pushIndex;
  String lastAction = null;
  int lastType = -1;
  Pair<Integer, Integer> lastP = new Pair(-1, -1); //(word, concept) pair

  //final DependencyTree tree;
  final AMRGraph graph;
  boolean startAction;
  AMRGraph goldGraph;

  String[] wordSeq;
  String[] posSeq;
  DependencyTree tree;
  //final CoreMap sentence;
  public AMRConfiguration(int cacheS) {
    stack = new ArrayList<>();
    buffer = null;
    cacheSize = cacheS;
    cache = new ArrayList<>();
    pushIndex = -1;
    for (int i = 0; i < cacheSize; i++) {
      cache.add(new Pair(-1, -1));
    }
    wordSeq = null;
    posSeq = null;
    //this.cache = new int[cacheSize];
    //for (int i = 0; i < cacheSize; i++)
    //  cache[i] = -1;
    this.graph = new AMRGraph();
    this.goldGraph = null;
    startAction = true;
  }

  public AMRConfiguration(int cacheS, int length) {
    stack = new ArrayList<>();
    buffer = new ArrayList<>();
    for (int i = 0; i < length; i++)
      buffer.add(i);
    cacheSize = cacheS;
    cache = new ArrayList<>();
    pushIndex = -1;
    for (int i = 0; i < cacheSize; i++)
      cache.add(new Pair(-1, -1));

    this.graph = new AMRGraph();
    this.goldGraph = null;

    wordSeq = null;
    posSeq = null;

    startAction = true;
  }

  public void setGold(AMRGraph g) {
    goldGraph = g;
  }

  public AMRConfiguration(AMRConfiguration config) {
    cacheSize = config.cacheSize;
    stack = new ArrayList<>(config.stack);
    buffer = new ArrayList<Integer>(config.buffer);
    cache = new ArrayList<>(config.cache);
    //for (int i = 0; i < cacheSize; i++)
    //    cache[i] = config.cache[i];
    //tree = new DependencyTree(config.tree);
    graph = new AMRGraph(config.graph);
    pushIndex = -1;
    startAction = true;

    //sentence = new CoreLabel(config.sentence);
  }



  public void popBuffer() {
    buffer.remove(0);
  }

  //Given the AMR graph, choose the vertex in cache that will
  // have the furthest connection
  public int chooseCache() {
    Map<Integer, Set<Integer>> headToTail = goldGraph.headToTail;
    Map<Integer, Set<Integer>> tailToHead = goldGraph.tailToHead;
    Map<Integer, Integer> wordToConcept = goldGraph.wordToConcept;
    int maxDist = -1;
    int maxIndex = -1;
    for (int cachePos = 0; cachePos < cache.size(); cachePos++) {
      Pair<Integer, Integer> p = cache.get(cachePos);
      int cacheVertex = p.second;
      //int cacheVertex = cache.get(cachePos);
      int currDist = 1000;
      Set<Integer> tailSet = headToTail.get(cacheVertex);
      Set<Integer> headSet = null;
      if (tailToHead.containsKey(cacheVertex))
        headSet = tailToHead.get(cacheVertex);
      //Set<Integer> headSet = tailToHead.get(cacheVer)
      for (int bufferPos = 0; bufferPos < buffer.size(); bufferPos++) {
        int wordIndex = buffer.get(bufferPos);
        if (wordToConcept.containsKey(wordIndex)) {
          int buffIndex = wordToConcept.get(wordIndex);
          if (tailSet.contains(buffIndex) || (headSet != null && headSet.contains(buffIndex))) {
            currDist = bufferPos;
            break;
          }
        }
      }
      if (currDist > maxDist) {
        maxIndex = cachePos;
        maxDist = currDist;
      }
    }
    return maxIndex;
  }

  public Pair<Integer, Pair<Integer, Integer>> removeTop() {
    int stackSize = stack.size();
    if (stackSize < 1) {
      return null;
    }
    Pair<Integer, Pair<Integer, Integer>> pair = stack.remove(stackSize-1);
    return pair;
  }

  public boolean needsPop() {
    Map<Integer, Set<Integer>> headToTail = goldGraph.headToTail;
    Map<Integer, Set<Integer>> tailToHead = goldGraph.tailToHead;
    Map<Integer, Integer> wordToConcept = goldGraph.wordToConcept;

    //Check if the rightmost vertex in the cache need pop
    int cacheSize = cache.size();
    Pair<Integer, Integer> p = cache.get(cacheSize-1);
    int vertex = p.second;
    //int vertex = cache.get(cacheSize-1);
    if (vertex == -1) {
      return false; //The rightmost is '$'
    }
    Set<Integer> tailSet = headToTail.get(vertex);
    Set<Integer> headSet = null;
    if (tailToHead.containsKey(vertex))
      headSet = tailToHead.get(vertex);
    //System.out.println()
    //System.out.println(tailToHead);
    //System.out.println(tailSet);
    for (int bufferPos = 0; bufferPos < buffer.size(); bufferPos++) {
      int wordIndex = buffer.get(bufferPos);
      if (wordToConcept.containsKey(wordIndex)) {
        int buffIndex = wordToConcept.get(wordIndex);
        if (tailSet.contains(buffIndex) || (headSet != null && headSet.contains(buffIndex))) {
          return false;
        }
      }
    }
    return true;
  }

  public Pair<Integer, Integer> getCache(int index) {
    if (index < 0 || index >= cacheSize)
      return Config.NONPAIR;
    return cache.get(index);
  }

  public boolean shift() {
    return shiftN(0);
  }

  public boolean shiftN(int n) {
    int k = getBuffer(0);
    if (k == Config.NONEXIST)
      return false;
    buffer.remove(0);
    if (n != 0) {
      pushIndex = n;
    }
    return true;
  }

  public boolean pop() {//pop from stack
    int nStack = getStackSize();
    if (nStack < 1)
      return false;
    Pair<Integer, Pair<Integer, Integer>> p = stack.get(nStack-1);
    int cacheIndex = p.first;
    Pair<Integer, Integer> pa = p.second;
    //int conceptId = p.second;
    cache.add(cacheIndex, pa);
    cache.remove(cacheSize);
    stack.remove(nStack-1);
    return true;
  }

  public void printConfig() {
    System.err.println(buffer);
    System.err.println(cache);
    System.err.println(stack);
  }

  public void connect(int currVertex, int cacheVertex, int direction, String arc) {
    //int cacheConcept = cache.get(cacheIndex);
    //if (cacheIndex == pushIndex) { //connect to replaced index
    //  int nStack = stack.size();
    //  cacheConcept = stack.get(nStack-1).second;
    //}
    //int currConcept = cache.get(pushIndex);
    ConceptLabel c = null;
    ConceptLabel pc = null;
    if (direction == 0) { //L edge, from pushed to cache
      c = graph.concepts.get(currVertex);
      c.tails.add(cacheVertex);

      pc = graph.concepts.get(cacheVertex);
      pc.parConcepts.add(currVertex);

      //ConceptLabel cachedC = graph.concepts.get(cacheConcept);
    }
    else {
      c = graph.concepts.get(cacheVertex);
      c.tails.add(currVertex);

      pc = graph.concepts.get(currVertex);
      pc.parConcepts.add(cacheVertex);
    }
    c.rels.add(arc);
    pc.parRels.add(arc);
  }

  public int getStackSize() {
    return stack.size();
  }

  public int getBufferSize() {
    return buffer.size();
  }

  /**
   * @param k Word index (zero = root node; actual word indexing
   *          begins at 1)
   */
  public int getHead(int k) {
    return tree.getHead(k);
  }

  /**
   * @param k Word index (zero = root node; actual word indexing
   *          begins at 1)
   */
  public String getLabel(int k) {
    return tree.getLabel(k);
  }

  /**
   * Get the sentence index of the kth word on the stack.
   *
   * @return Sentence index or {@link Config#NONEXIST} if stack doesn't
   *         have an element at this index
   */
  public Pair<Integer, Pair<Integer, Integer>> getStack(int k) {
    int nStack = getStackSize();
    return (k >= 0 && k < nStack) ? stack.get(nStack - 1 - k) : Config.NONTRIPLE;
  }

  //public int getStackVertex(int k) {
  //  int nStack = stack.size();
  //  if (k >= 0 && k < nStack) {
  //    Pair<Integer, Integer> p = stack.get(nStack-1-k);
  //    return p.second;
  //  }
  //  return Config.NONEXIST;
  //}

  //public Pair<Integer, Integer> getCache(int k) {
  //  if (k < 0) return
  //}

  public int getCacheVertex(int k) {
    if (k < 0) return Config.NONEXIST;
    return cache.get(cacheSize-1-k).second;
  }

  public int getCacheWord(int k) {
    if (k < 0 || k >= cacheSize) return Config.NONEXIST;
    return cache.get(k).first;
  }

  public int getID(String s, Map<String, Integer> idMap) {
    return idMap.containsKey(s) ? idMap.get(s) : idMap.get(Config.UNKNOWN);
  }

  public void getCacheContexts(int cacheIndex, int k, Map<String, Integer> idMap1, Map<String, Integer> idMap2, List<Integer> feats, boolean isWord) {
    for (int i = cacheIndex-k; i <= cacheIndex+k; i++) {
      if (i < 0 || i >= cacheSize) {
        feats.add(idMap1.get(Config.NULL));
        if (isWord) {
          feats.add(idMap2.get(Config.NULL));
        }
        continue;
      }
      if (isWord) {
        int index = getCacheWord(i);
        String word = getWord(index);
        feats.add(getID(word, idMap1));

        String pos = getPOS(index);
        feats.add(getID(pos, idMap2));
      }
      else {
        int vertex = getCacheVertex(i);
        String concept = getConcept(vertex);
        feats.add(getID(concept, idMap1));
      }
    }
  }

  public void getCacheFeats(int k, Map<String, Integer> idMap, List<Integer> feats, boolean isWord) {
    for (int i = 0; i < k; i++) {
      int vertex = getCacheVertex(i);
      String concept = getConcept(vertex);
      if (isWord) {
        vertex = getCacheWord(cacheSize-1-i); //From the rightmost
        concept = getWord(vertex);
      }
      feats.add(getID(concept, idMap));
    }
  }

  //k is the window size
  public void getBufferFeats(int k, Map<String, Integer> idMap, List<Integer> feats, boolean isWord) {
    if (buffer.size() == 0) {
      for (int i = 0; i <= 2*k; i++) {
        feats.add(getID(Config.NULL, idMap));
      }
      return;
    }
    int currPos = buffer.get(0);
    for (int i = currPos-k; i <= currPos+k; i++) {
      String word = Config.NULL;
      if (i >= 0 && i < wordSeq.length) {
        if (isWord)
          word = wordSeq[i];
        else
          word = posSeq[i];
      }
      feats.add(getID(word, idMap));
    }
  }

  public void pushStack(int cacheIndex) {
    Pair<Integer, Integer> p = cache.get(cacheIndex);
    //int vertex = cache.get(cacheIndex);
    Pair<Integer, Pair<Integer, Integer>> pa = new Pair<>();
    pa.first = cacheIndex;
    pa.second = p;
    cache.remove(cacheIndex);
    stack.add(pa);
  }

  /**
   * Get the sentence index of the kth word on the buffer.
   *
   * @return Sentence index or {@link Config#NONEXIST} if stack doesn't
   *         have an element at this index
   */
  public int getBuffer(int k) {
    return (k >= 0 && k < getBufferSize()) ? buffer.get(k) : Config.NONEXIST;
  }

  public List<CoreLabel> getCoreLabels() {
    return null;
  }

  /**
   * @param k Word index (zero = root node; actual word indexing
   *          begins at 1)
   */
  public String getWord(int k) {
    if (k < 0) return Config.NULL;
    return wordSeq[k];
  }

  public String getConcept(int k) {
    if (k < 0) {
      return Config.NULL;
    }
    return graph.concepts.get(k).value();
  }

  /**
   * @param k Word index (zero = root node; actual word indexing
   *          begins at 1)
   */
  public String getPOS(int k) {
    if (k < 0) return Config.NULL;
    return posSeq[k];
  }

  /**
   * @param h Word index of governor (zero = root node; actual word
   *          indexing begins at 1)
   * @param t Word index of dependent (zero = root node; actual word
   *          indexing begins at 1)
   * @param l Arc label
   */
  public void addArc(int h, int t, String l) {
    //tree.set(t, h, l);
  }

  //public int getStack(int i) {
  //  int stackSize = stack.size();
  //  if (i >= stackSize) {
  //    return Config.NONEXIST;
  //  }
  //  else
  //    return stack.get(stackSize-1-i).second;
  //}

  public int getLeftChild(int k, int cnt) {
    if (k < 0 || k >= tree.n)
      return Config.NONEXIST;

    int c = 0;
    for (int i = 1; i < k; ++i)
      if (tree.getHead(i) == k)
        if ((++c) == cnt)
          return i;
    return Config.NONEXIST;
  }

  public int getLeftChild(int k) {
    return getLeftChild(k, 1);
  }

  public int getRightChild(int k, int cnt) {
    if (k < 0 || k >= tree.n)
      return Config.NONEXIST;

    int c = 0;
    for (int i = tree.n-1; i > k; --i)
      if (tree.getHead(i) == k)
        if ((++c) == cnt)
              return i;
    return Config.NONEXIST;
  }

  public int getRightChild(int k) {
    return getRightChild(k, 1);
  }

  //See if there is an arc from wordIndex to the cacheIndex
  public String getArcLabel(int cacheIndex, int wordIndex, boolean left) {
    Pair<Integer, Integer> p = cache.get(cacheIndex);
    int cacheWordIndex = p.first;
    if (cacheWordIndex < 0) {
      return Config.NULL;
    }

    if (left) {
      if (tree.getHead(cacheWordIndex) == wordIndex)
        return getLabel(cacheWordIndex);
    }
    else {
      if (tree.getHead(wordIndex) == cacheWordIndex)
        return getLabel(wordIndex);
    }
    return Config.NULL;
  }

  public boolean hasOtherChild(int k, DependencyTree goldTree) {
    //for (int i = 1; i <= tree.n; ++i)
    //  if (goldTree.getHead(i) == k && tree.getHead(i) != k) return true;
    //return false;
    return true;
  }

  public int getLeftValency(int k) {
    //if (k < 0 || k > tree.n)
    //  return Config.NONEXIST;
    //int cnt = 0;
    //for (int i = 1; i < k; ++i)
    //  if (tree.getHead(i) == k)
    //    ++cnt;
    //return cnt;
    return 0;
  }

  public int getRightValency(int k) {
    //if (k < 0 || k > tree.n)
    //  return Config.NONEXIST;
    //int cnt = 0;
    //for (int i = k + 1; i <= tree.n; ++i)
    //  if (tree.getHead(i) == k)
    //    ++cnt;
    //return cnt;
    return 0;
  }

  public String getLeftLabelSet(int k) {
    //if (k < 0 || k > tree.n)
    //  return Config.NULL;

    //HashSet<String> labelSet = new HashSet<String>();
    //for (int i = 1; i < k; ++i)
    //  if (tree.getHead(i) == k)
    //    labelSet.add(tree.getLabel(i));

    //List<String> ls = new ArrayList<String>(labelSet);
    //Collections.sort(ls);
    //String s = "";
    //for (int i = 0; i < ls.size(); ++i)
    //  s = s + "/" + ls.get(i);
    //return s;
    return null;
  }

  public String getRightLabelSet(int k) {
    //if (k < 0 || k > tree.n)
    //  return Config.NULL;

    //HashSet<String> labelSet = new HashSet<String>();
    //for (int i = k + 1; i <= tree.n; ++i)
    //  if (tree.getHead(i) == k)
    //    labelSet.add(tree.getLabel(i));

    //List<String> ls = new ArrayList<String>(labelSet);
    //Collections.sort(ls);
    //String s = "";
    //for (int i = 0; i < ls.size(); ++i)
    //  s = s + "/" + ls.get(i);
    //return s;
    return null;
  }

  //returns a string that concatenates all elements on the stack and buffer, and head / label.
  public String getStr() {
    //String s = "[S]";
    //for (int i = 0; i < getStackSize(); ++i) {
    //  if (i > 0) s = s + ",";
    //  s = s + stack.get(i);
    //}
    //s = s + "[B]";
    //for (int i = 0; i < getBufferSize(); ++i) {
    //  if (i > 0) s = s + ",";
    //  s = s + buffer.get(i);
    //}
    //s = s + "[H]";
    //for (int i = 1; i <= tree.n; ++i) {
    //  if (i > 1) s = s + ",";
    //  s = s + getHead(i) + "(" + getLabel(i) + ")";
    //}
    //return s;
    return null;
  }
}