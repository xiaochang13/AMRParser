package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasTag;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.TreeGraphNode;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalStructure;
import edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalRelations;
import edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import sun.management.counter.StringCounter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.*;

import static java.util.stream.Collectors.toList;

/**
 * This class defines a transition-based dependency parser which makes
 * use of a classifier powered by a neural network. The neural network
 * accepts distributed representation inputs: dense, continuous
 * representations of words, their part of speech tags, and the labels
 * which connect words in a partial dependency parse.
 *
 * <p>
 * This is an implementation of the method described in
 *
 * <blockquote>
 *   Danqi Chen and Christopher Manning. A Fast and Accurate Dependency
 *   Parser Using Neural Networks. In EMNLP 2014.
 * </blockquote>
 *
 * <p>
 *
 * <p>
 * This parser can also be used programmatically. The easiest way to
 * prepare the parser with a pre-trained model is to call
 * parser instance in order to get new parses.
 *
 * @author Xiaochang Peng
 */
public class AMRParser {
  public static final String DEFAULT_MODEL = "edu/stanford/nlp/models/parser/nndep/english_UD.gz";

  /**
   * Words, parts of speech, and dependency relation labels which were
   * observed in our corpus / stored in the model
   *
   */
  private List<String> knownWords, knownPos, knownDeps, knownConcepts, knownArcs;
  private List<String> allLabels;

  private Writer featureWriter;
  private Writer devFeatWriter;

  /**
   * Mapping from word / POS / dependency relation label to integer ID
   */
  private Map<String, Integer> wordIDs, posIDs, depIDs, conceptIDs, arcIDs;
  private Map<String, Integer> conceptIDTargetIDs;
  private Map<Integer, Set<Integer>> wordToConcepts;
  private Map<Integer, Map<String, Integer>> wordConceptCounts;
  private Map<Integer, Set<Integer>> conceptIDDict;
  private List<String> unalignedSet;

  private List<Integer> preComputed;

  /**
   * Given a particular parser configuration, this classifier will
   * predict the best transition to make next.
   *
   * The {@link edu.stanford.nlp.parser.nndep.Classifier} class
   * handles both training and inference.
   */
  //private Classifier classifier;
  private Classifier conceptIDClassifier;
  private Classifier arcConnectClassifier;
  private Classifier pushIndexClassifier;
  private CacheTransition system;

  private final Config config;

  /**
   * Language used to generate
   * {@link edu.stanford.nlp.trees.GrammaticalRelation} instances.
   */
  private final Language language;

  AMRParser() {
    this(new Properties());
  }

  public AMRParser(Properties properties) {
    config = new Config(properties);
    wordConceptCounts = new HashMap<>();
    conceptIDDict = new HashMap<>();
    unalignedSet = new ArrayList<>();

    preComputed = null;

    // Convert Languages.Language instance to
    // GrammaticalLanguage.Language
    this.language = config.language;

    if (config.saveFeat) {
      try {
        featureWriter = IOUtils.getPrintWriter("train.feat");
        devFeatWriter = IOUtils.getPrintWriter("dev.feat");
      }
      catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }
  }

  /**
   * Get an integer ID for the given word. This ID can be used to index
   * into the embeddings.
   *
   * @return An ID for the given word, or an ID referring to a generic
   *         "unknown" word if the word is unknown
   */
  public int getWordID(String s) {
    return wordIDs.containsKey(s) ? wordIDs.get(s) : wordIDs.get(Config.UNKNOWN);
  }

  public int getConceptID(String s) {
    return conceptIDs.containsKey(s) ? conceptIDs.get(s) : conceptIDs.get(Config.UNKNOWN);
  }

  public int getPosID(String s) {
      return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(Config.UNKNOWN);
  }

  public int getDepID(String s) {
    return depIDs.containsKey(s) ? depIDs.get(s) : depIDs.get(Config.UNKNOWN);
  }

  public int getArcID(String s) {
    return arcIDs.containsKey(s) ? arcIDs.get(s) : arcIDs.get(Config.UNKNOWN);
  }

  //Define a separate set of features for concept identification.
  //We assume the most important features are buffer context surrounding the next word
  //including the current word
  public List<Integer> conceptIDFeatures(AMRConfiguration c) {
    List<Integer> feature = new ArrayList<>();

    //TOP two "vertices" on the stack
    c.getCacheFeats(2, conceptIDs, feature, false);

    //word and POS tag features from a context window of 2
    c.getBufferFeats(2, wordIDs, feature, true);
    c.getBufferFeats(2, posIDs, feature, false);

    return feature;
  }

  //Define a separate set of features for connecting arcs.
  //On the cache there is a context window of features.

  public List<Integer> arcConnectFeatures(AMRConfiguration c, int cacheIndex, String[] allDecisions) {
    List<Integer> feature = new ArrayList<>();
    Pair<Integer, Integer> conceptP = c.lastP;

    int wordIndex = conceptP.first;
    //int conceptIndex = conceptP.second;
    int cacheWordIndex = c.getCacheWord(cacheIndex);

    // on the cache
    c.getCacheContexts(cacheIndex, 1, wordIDs, posIDs, feature, true);
    c.getCacheContexts(cacheIndex, 1, conceptIDs, null, feature, false);

    //word and POS tag features from a context window of 2
    c.getBufferFeats(1, wordIDs, feature, true);
    c.getBufferFeats(1, posIDs, feature, false);;

    //Dependency features
    //First: general features for predicting labels; Second: cache index features for making specific decisions
    int index = c.getLeftChild(wordIndex);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));   //Here we have assume the input feature order does not matter
    feature.add(getDepID(c.getLabel(index)));

    index = c.getRightChild(wordIndex);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    index = c.getLeftChild(wordIndex, 2);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    index = c.getRightChild(wordIndex, 2);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    index = c.getLeftChild(cacheWordIndex);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));   //Here we have assume the input feature order does not matter
    feature.add(getDepID(c.getLabel(index)));

    index = c.getRightChild(cacheWordIndex);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    index = c.getLeftChild(cacheWordIndex, 2);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    index = c.getRightChild(cacheWordIndex, 2);
    feature.add(getWordID(c.getWord(index)));
    feature.add(getPosID(c.getPOS(index)));
    feature.add(getDepID(c.getLabel(index)));

    //A unary label feature for current cache index and the current word
    feature.add(getDepID(c.getArcLabel(cacheIndex, wordIndex, true)));
    feature.add(getDepID(c.getArcLabel(cacheIndex, wordIndex, false)));

    return feature;
  }

  //Define a separate set of features for pushing index to an index in the cache.
  public List<Integer> pushIndexFeatures(AMRConfiguration c) {
    List<Integer> feature = new ArrayList<>();
    c.getCacheFeats(c.cacheSize, conceptIDs, feature, false);
    c.getCacheFeats(c.cacheSize, wordIDs, feature, true);
    //c.getBufferFeats(c.cacheSize, posIDs, feature, false);
    return feature;
  }

  //List of token features used
  public List<Integer> getFeatures(AMRConfiguration c) {
    // Presize the arrays for very slight speed gain. Hardcoded, but so is the current feature list.
    List<Integer> fWord = new ArrayList<>(3);
    List<Integer> fPos = new ArrayList<>(3);
    List<Integer> fConcept = new ArrayList<>(3);
    //List<Integer> fLabel = new ArrayList<>(12);   //Currently no label features available

    //TOP three "vertices" on the stack
    //At this time,
    for (int j = 2; j >= 0; --j) {
      //int index = c.getStackVertex(j);
      int index = c.getCacheVertex(j);
      //fWord.add(getWordID(c.getWord(index)));
      fConcept.add(getConceptID(c.getConcept(index)));
      //System.err.println(c.getConcept(index));
      //fPos.add(getPosID(c.getPOS(index)));
    }
    //Top three words on the buffer
    for (int j = 0; j <= 2; ++j) {
      int index = c.getBuffer(j);
      String currWord = c.getWord(index);
      //System.err.println(currWord);
      //System.err.println(c.getPOS(index));
      fWord.add(getWordID(currWord));
      fPos.add(getPosID(c.getPOS(index)));
    }

    //Dependency features

    //Features based on the current setting
    //for (int j = 0; j <= 1; ++j) {
    //  //int k = c.getStack(j);
    //  Pair<Integer, Integer> p = c.getStack(j);
    //  int k = p.second;
    //  int index = c.getLeftChild(k);
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));

    //  index = c.getRightChild(k);
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));

    //  index = c.getLeftChild(k, 2);
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));

    //  index = c.getRightChild(k, 2);
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));

    //  index = c.getLeftChild(c.getLeftChild(k));
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));

    //  index = c.getRightChild(c.getRightChild(k));
    //  fWord.add(getWordID(c.getWord(index)));
    //  fPos.add(getPosID(c.getPOS(index)));
    //  fLabel.add(getLabelID(c.getLabel(index)));
    //}

    List<Integer> feature = new ArrayList<>(9);
    feature.addAll(fWord);
    feature.addAll(fPos);
    feature.addAll(fConcept);
    //System.err.println(feature);
    //feature.addAll(fLabel);
    return feature;
  }

  private static final int POS_OFFSET = 18;
  private static final int DEP_OFFSET = 36;
  private static final int STACK_OFFSET = 6;
  private static final int STACK_NUMBER = 6;

  /*
  To implement feature extraction for the current configuration c
   */
  /*
  private int[] getFeatureArray(Configuration c) {
    int[] feature = new int[config.numTokens];  // positions 0-17 hold fWord, 18-35 hold fPos, 36-47 hold fLabel

    for (int j = 2; j >= 0; --j) {
      int index = c.getStack(j);
      feature[2-j] = getWordID(c.getWord(index));
      feature[POS_OFFSET + (2-j)] = getPosID(c.getPOS(index));
    }

    for (int j = 0; j <= 2; ++j) {
      int index = c.getBuffer(j);
      feature[3 + j] = getWordID(c.getWord(index));
      feature[POS_OFFSET + 3 + j] = getPosID(c.getPOS(index));
    }

    for (int j = 0; j <= 1; ++j) {
      int k = c.getStack(j);

      int index = c.getLeftChild(k);
      feature[STACK_OFFSET + j * STACK_NUMBER] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER] = getLabelID(c.getLabel(index));

      index = c.getRightChild(k);
      feature[STACK_OFFSET + j * STACK_NUMBER + 1] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 1] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 1] = getLabelID(c.getLabel(index));

      index = c.getLeftChild(k, 2);
      feature[STACK_OFFSET + j * STACK_NUMBER + 2] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 2] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 2] = getLabelID(c.getLabel(index));

      index = c.getRightChild(k, 2);
      feature[STACK_OFFSET + j * STACK_NUMBER + 3] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 3] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 3] = getLabelID(c.getLabel(index));

      index = c.getLeftChild(c.getLeftChild(k));
      feature[STACK_OFFSET + j * STACK_NUMBER + 4] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 4] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 4] = getLabelID(c.getLabel(index));

      index = c.getRightChild(c.getRightChild(k));
      feature[STACK_OFFSET + j * STACK_NUMBER + 5] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 5] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 5] = getLabelID(c.getLabel(index));
    }

    return feature;
  }
  */

  //Given the current setting of the buffer, see if the rightmost element in the cache needs a pop
  public boolean needsPop(List<Integer>cache, List<Integer>buffer, Map<Integer, Set<Integer>> headToTail,
                          Map<Integer, Set<Integer>> tailToHead, Map<Integer, Integer> wordToConcept) {
    int rightId = cache.get(cache.size()-1);
    if (rightId == -1) {
      return false;
    }
    Set<Integer> tailSet = headToTail.get(rightId);
    Set<Integer> headSet = null;
    if (tailToHead.containsKey(rightId))
      headSet = tailToHead.get(rightId);
    for (int bufId : buffer) {
      if (wordToConcept.containsKey(bufId)) {
        int buffIndex = wordToConcept.get(bufId);
        if (tailSet.contains(buffIndex) || (headSet != null && headSet.contains(buffIndex))) {
          return false;
        }
      }
    }
    return true;
  }

  public int getType(String oracle) {
    int ret;
    if (oracle.equals("POP") || oracle.contains("conID") || oracle.contains("conGen") || oracle.contains("conEMP")) {
      ret = 0;
    }
    else if (oracle.contains("ARC"))
      ret = 1;
    else //The push index transitions
      ret = 2;
    return ret;
  }

  public boolean checkGold(List<Integer> labels) {
    int oneNum = 0;
    for (int l: labels) {
      if (l == 1) {
        oneNum++ ;
      }
    }
    return oneNum == 1;
  }

  //Feature extraction for training examples in the oracle sequence
  public AMRData genTrainExamples(List<String[]> sents, List<String[]> poss, List<DependencyTree> trees, List<AMRGraph> graphs, boolean isTrain) {

    //system.makeTransitions();   //Generate all the possible actions for the three neural nets.
    //int numTrans = system.numTransitions(2);
    //Dataset ret = new Dataset(config.numTokens, numTrans);
    AMRData ret = new AMRData(); //Parameters to be fixed
    //Map<String, Set<String>> conceptMap = new HashMap<>();

    Counter<Integer> tokPosCount = new IntCounter<>();
    System.err.println(Config.SEPARATOR);
    System.err.println("Generate training examples...");

    int numConceptIDTransitions = system.numTransitions(0);
    int numArcConnectTransitions = system.numTransitions(1);
    int numPushIndexTransitions = system.numTransitions(2);

    ret.initExample(Config.conIDTokens, numConceptIDTransitions, 0);
    ret.initExample(Config.arcConnectTokens, numArcConnectTransitions, 1);
    ret.initExample(Config.pushIndexTokens, numPushIndexTransitions, 2);

    //System.err.println(conceptIDs);

    for (int i = 0; i < sents.size(); ++i) {
      //System.out.println("Sentence " + i);
      //if (i > 0)
      //  break;

      if (i != 1272)
        continue;

      if (i > 0) {
        if (i % 1000 == 0)
          System.err.print(i + " ");
        if (i % 10000 == 0 || i == sents.size() - 1)
          System.err.println();
      }

      String[] tokSeq = sents.get(i);
      String[] posSeq = poss.get(i);
      DependencyTree tree = trees.get(i);
      AMRGraph graph = graphs.get(i);
      graph.setSentence(tokSeq);

      int tokIndex = 0;

      AMRConfiguration c = new AMRConfiguration(config.CACHE, tokSeq.length);
      c.wordSeq = tokSeq;
      c.posSeq = posSeq;
      c.tree = tree;
      c.setGold(graph);

      String cacheID = null;
      c.startAction = true;

      //System.err.println(allLabels);
      //graph.printGraph();

      while (!system.isTerminal(c)) {
        String oracle = system.getOracle(c, tokIndex);
        String nextWord = null;
        //System.err.println(oracle);
        if (oracle.contains("conID") || oracle.contains("conEMP")) {
          nextWord = tokSeq[tokIndex];
          //System.out.println(nextWord + " "+ oracle);
          tokIndex += 1;
        }

        int oracleType = getType(oracle);

        int numTrans = system.numTransitions(oracleType);

        if (!oracle.contains("ARC")) {
          List<Integer> feature = null;
          //List<Integer> feature = getFeatures(c);
          if (oracle.contains("PUSH")) {
            feature = pushIndexFeatures(c);
            //System.err.println("PUSH Index, feature size: " + feature.size());
          }
          else {
            feature = conceptIDFeatures(c);
            //System.err.println("Concept Identification, feature size: " + feature.size());
          }

          List<Integer> label = new ArrayList<>();

          int optid = -1;
          for (int j = 0; j < numTrans; ++j) {
            String str = system.getTransition(oracleType, j);

            //System.out.println("Current transition:" + str);
            if (str.equals(oracle)) {
              label.add(1);
              optid = j;
            } else if (system.canApply(c, str)) label.add(0);
            else label.add(-1);
          }

          int wordID = -1;
          if (oracle.contains("conID") || oracle.contains("conEMP")) {
            String currWord = tokSeq[c.getBuffer(0)];
            if (!currWord.equals(nextWord)) {
              System.err.println("Something wrong with buffer!");
              System.exit(1);
            }
            wordID = getWordID(currWord);
            //wordID = wordIDs.get(currWord);
          }

          //if (oracleType == 0 && wordID == -1) {
          //  System.err.println(tokSeq[c.getBuffer(0)]);
          //}
          if (!checkGold(label) && !oracle.contains("ARC")) { //Edges to be processed independently
            if (!isTrain) {
              label.set(1, 1);  //Set the first position 1
            }
            else {
              System.err.println(label);
              System.err.println("Label error!");
              System.err.println(oracle);
              System.err.println(system.transitionList(oracleType));
              System.exit(1);
            }
          }

          ret.addExample(feature, label, oracleType, wordID);
          for (int j = 0; j < feature.size(); ++j) {
            int featIndex = feature.get(j) * feature.size() + j;
            //System.err.println("featureIndex: " + featIndex + "; feature label: " + allLabels.get(feature.get(j)));
            tokPosCount.incrementCount(featIndex);
          }
        }
        else { //For predictions of ARCs, need separate procedure
          //First we need to split the oracle into individual edge-connecting decisions
          String[] parts = oracle.split(":");
          String[] arcDecisions = parts[1].split("#");
          //System.err.println(parts[1]);

          if (arcDecisions.length != c.cacheSize) {
            System.err.println("The number of arc decisions does not match cache size!");
            System.exit(1);
          }

          for (int cacheIndex = 0; cacheIndex < c.cacheSize; cacheIndex++) {
            String currAction = arcDecisions[cacheIndex];
            if (currAction.length() > 2) {
              String l = currAction.substring(2);
              if (!arcIDs.containsKey(l)) {
                currAction = currAction.substring(0, 2) + Config.UNKNOWN;
              }
            }

            List<Integer> feature = arcConnectFeatures(c, cacheIndex, arcDecisions);
            //System.err.println("ARC Connect, feature size: " + feature.size());
            List<Integer> label = new ArrayList<>();

            for (int j = 0; j < numTrans; ++j) {
              String str = system.getTransition(oracleType, j);

              if (str.equals(currAction)) {
                //System.err.println("Current action:" + currAction);
                //System.err.println(j);
                //System.exit(1);
                label.add(1);
              } else if (system.canApply(c, str)) label.add(0);
              else label.add(-1);
            }


            if (!checkGold(label)) { //Edges to be processed independently
              System.err.println("ARC Label error!");
              System.err.println(currAction);
              System.err.println(system.transitionList(oracleType));
              System.exit(1);
            }

            int wordID = -1;
            ret.addExample(feature, label, oracleType, wordID);
            for (int j = 0; j < feature.size(); ++j) {
              int featIndex = feature.get(j) * feature.size() + j;
              //System.err.println("featureIndex: " + featIndex + "; feature label: " + allLabels.get(feature.get(j)));
              tokPosCount.incrementCount(featIndex);
            }

          }
        }

        system.apply(c, oracle);
      }
      //System.out.println("Gold graph:");
      //graph.printGraph();
      //System.out.println("Built oracle graph:");
      //c.graph.printGraph();
    }
    //System.exit(1);
    //System.err.println("#Train Examples: " + ret.n);

    if (preComputed == null) {
      List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);
      preComputed = new ArrayList<>(sortedTokens.subList(0, Math.min(config.numPreComputed, sortedTokens.size())));
    }

    return ret;
  }

  //Generate a map from word to a set of the possible concepts.
  //Maybe should filter the concept mappings for too infrequent concept choices.
  public void generateConceptMap(List<String[]> sents, List<String[]> poss, List<AMRGraph> graphs) {
    int sent_id = 0;
    Counter<String> unalignConceptCounter = new IntCounter<>();

    for (AMRGraph g : graphs) {

      String[] tokSeq = sents.get(sent_id);
      String[] posSeq = poss.get(sent_id);

      Set<Integer> aligned_set = new HashSet<>();

      for (ConceptLabel c : g.concepts) {
        //int conceptID = -1;
        String concept = c.value();
        if (c.aligned) {
          //String conceptTran = "conID:" + concept;
          //if (!conceptIDTargetIDs.containsKey(conceptTran))
          //  conceptIDTargetIDs.put(conceptTran, conceptIDTargetIDs.size());

          //conceptID = conceptIDTargetIDs.get(conceptTran);
          if (c.alignments.size() != 1) {
            System.err.println("Alignment size is not one");
            System.exit(1);
          }
          for (int index : c.alignments) {
            aligned_set.add(index);
            String currWord = tokSeq[index];
            if (wordIDs.containsKey(currWord)) {
              int wordID = wordIDs.get(currWord);
              if (!wordConceptCounts.containsKey(wordID))
                wordConceptCounts.put(wordID, new HashMap<String, Integer>());
              Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
              if (conceptCounts.containsKey(concept))
                conceptCounts.put(concept, conceptCounts.get(concept) + 1);
              else
                conceptCounts.put(concept, 1);
            }
          }
        }
        else {
          String conceptTran = "conGen:" + concept;
          //if (!conceptIDTargetIDs.containsKey(conceptTran))
          //  conceptIDTargetIDs.put(conceptTran, conceptIDTargetIDs.size());
          //conceptID = conceptIDTargetIDs.get(conceptTran);
          unalignConceptCounter.incrementCount(concept);
        }
      }

      for (int i = 0; i < tokSeq.length; i++) {
        if (aligned_set.contains(i))
          continue;
        int wordID = wordIDs.get(tokSeq[i]);
        if (!wordConceptCounts.containsKey(wordID))
          wordConceptCounts.put(wordID, new HashMap<>());
        Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
        //int nullID = conceptIDs.get(config.NULL);
        if (conceptCounts.containsKey(config.NULL))
          conceptCounts.put(config.NULL, conceptCounts.get(config.NULL)+1);
        else
          conceptCounts.put(config.NULL, 1);
      }
      List<String> sortedUnalign = Counters.toSortedList(unalignConceptCounter, false);
      unalignedSet = freqFilter(unalignConceptCounter, sortedUnalign, 1);
      ratioFilter(0.2);

      //System.err.println(wordConceptCounts);
      //for (int wordID: wordConceptCounts.keySet()) {
      //  System.err.println(knownWords.get(wordID));
      //  Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
      //  for (String concept: conceptCounts.keySet()) {
      //    System.err.println(concept+ " : "+ conceptCounts.get(concept));
      //  }
      //}
      //System.exit(1);

      //System.out.println("Unalign concepts:");
      //unalignConceptCounter.
      //for (String s: sortedUnalign) {
      //  unalignConceptCounter.getCount(s)
      //}

        //System.out.println(s);
      sent_id += 1;
    }
  }

  public List<String> freqFilter(Counter<String>counts, List<String> sortedList, int freq) {
    List<String> ret = new ArrayList<>();
    for (String s: sortedList) {
      if (counts.getCount(s) < freq)
        break;
      ret.add(s);
    }
    return ret;
  }

  //Adjust the alignment to avoid possible alignment errors
  public void ratioFilter(double ratio) {
    for(int wordID: wordConceptCounts.keySet()) {
      Map<String, Integer> newMap = new HashMap<>();
      Map<String, Integer> oldMap = wordConceptCounts.get(wordID);

      int maxCount = 0;
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount > maxCount)
          maxCount = currCount;
      }
      int threshold = (int)(maxCount * ratio);
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount >= threshold) {
          newMap.put(concept, currCount);
        }
      }
      wordConceptCounts.put(wordID, newMap);
    }
  }
  /**
   * Generate unique integer IDs for all known words / part-of-speech
   * tags / dependency relation labels.
   *
   * All three of the aforementioned types are assigned IDs from a
   * continuous range of integers; all IDs 0 <= ID < n_w are word IDs,
   * all IDs n_w <= ID < n_w + n_pos are POS tag IDs, and so on.
   */
  private void generateIDs() {
    wordIDs = new HashMap<>();
    posIDs = new HashMap<>();
    depIDs = new HashMap<>();
    conceptIDs= new HashMap<>();
    arcIDs = new HashMap<>();

    int index = 0;
    for (String word : knownWords)
      wordIDs.put(word, (index++));
    for (String pos : knownPos)
      posIDs.put(pos, (index++));
    for (String dep: knownDeps)
      depIDs.put(dep, (index++));
    for (String concept : knownConcepts)
      conceptIDs.put(concept, (index++));
    for (String arc : knownArcs)
      arcIDs.put(arc, (index++));
  }

  /**
   * Scan a corpus and store all words, part-of-speech tags, and
   * dependency relation labels observed. Prepare other structures
   * which support word / POS / label lookup at train- / run-time.
   */
  private void genDictionaries(List<String[]> tokens, List<String[]> posTags, List<DependencyTree> trees, List<AMRGraph> graphs) {
    // Collect all words (!), etc. in lists, tacking on one sentence
    // after the other
    List<String> word = new ArrayList<>();
    List<String> pos = new ArrayList<>();
    List<String> dep = new ArrayList<>();
    List<String> concepts = new ArrayList<>();
    List<String> arcs = new ArrayList<>();

    for (String[] tokSeq : tokens)
      for (String tok : tokSeq)
        word.add(tok);

    for (String[] posSeq : posTags)
      for (String tok : posSeq)
        pos.add(tok);

    for (DependencyTree tree: trees) {
      for (String lab: tree.label) {
        dep.add(lab);
      }
    }

    //String rootLabel = null;
    for (AMRGraph graph : graphs) {
      //int rootIndex = graph.root;
      for (ConceptLabel c : graph.concepts) {
        concepts.add(c.value());
        for (String s : c.rels)
          arcs.add(s);
      }
    }

    // Generate "dictionaries," possibly with frequency cutoff
    knownWords = Util.generateDict(word, config.wordCutOff);
    knownPos = Util.generateDict(pos);
    knownDeps = Util.generateDict(dep);
    knownConcepts = Util.generateDict(concepts);
    //System.err.println(arcs);
    //knownArcs = Util.generateDict(arcs, config.arcCutOff);
    knownArcs = Util.topDict(arcs, config.arcCutOff);
    //System.err.println(knownArcs + "   "+ config.arcCutOff);

    knownWords.add(0, Config.UNKNOWN); //Tokens in the buffer to be processed
    knownWords.add(1, Config.NULL);
    knownPos.add(0, Config.UNKNOWN);
    knownPos.add(1, Config.NULL);
    knownDeps.add(0, Config.UNKNOWN);
    knownDeps.add(1, Config.NULL);

    //knownLabels.add(0, Config.NULL);
    knownArcs.add(0, Config.UNKNOWN);
    knownArcs.add(1, Config.NULL); //In case in the current setting there is not

    knownConcepts.add(0, Config.UNKNOWN);
    knownConcepts.add(1, Config.NULL);

    allLabels = new ArrayList<>(knownWords);
    allLabels.addAll(knownPos);
    allLabels.addAll(knownDeps);
    allLabels.addAll(knownConcepts);
    allLabels.addAll(knownArcs);

    generateIDs();

    System.err.println(Config.SEPARATOR);
    System.err.println("#Word: " + knownWords.size());
    System.err.println("#POS: " + knownPos.size());
    System.err.println("#Deps: " + knownDeps.size());
    System.err.println("#Concepts: " + knownConcepts.size());
    System.err.println("#Arcs: " + knownArcs.size());
  }

  public void writeModelFile(String modelFile) {
    //try {
    //  double[][] W1 = classifier.getW1();
    //  double[] b1 = classifier.getb1();
    //  double[][] W2 = classifier.getW2();
    //  double[][] E = classifier.getE();

    //  Writer output = IOUtils.getPrintWriter(modelFile);

    //  output.write("dict=" + knownWords.size() + "\n");
    //  output.write("pos=" + knownPos.size() + "\n");
    //  output.write("label=" + knownLabels.size() + "\n");
    //  output.write("embeddingSize=" + E[0].length + "\n");
    //  output.write("hiddenSize=" + b1.length + "\n");
    //  output.write("numTokens=" + (W1[0].length / E[0].length) + "\n");
    //  output.write("preComputed=" + preComputed.size() + "\n");

    //  int index = 0;

    //  // First write word / POS / label embeddings
    //  for (String word : knownWords) {
    //    output.write(word);
    //    for (int k = 0; k < E[index].length; ++k)
    //      output.write(" " + E[index][k]);
    //    output.write("\n");
    //    index = index + 1;
    //  }
    //  for (String pos : knownPos) {
    //    output.write(pos);
    //    for (int k = 0; k < E[index].length; ++k)
    //      output.write(" " + E[index][k]);
    //    output.write("\n");
    //    index = index + 1;
    //  }
    //  for (String label : knownLabels) {
    //    output.write(label);
    //    for (int k = 0; k < E[index].length; ++k)
    //      output.write(" " + E[index][k]);
    //    output.write("\n");
    //    index = index + 1;
    //  }

    //  // Now write classifier weights
    //  for (int j = 0; j < W1[0].length; ++j)
    //    for (int i = 0; i < W1.length; ++i) {
    //      output.write("" + W1[i][j]);
    //      if (i == W1.length - 1)
    //        output.write("\n");
    //      else
    //        output.write(" ");
    //    }
    //  for (int i = 0; i < b1.length; ++i) {
    //    output.write("" + b1[i]);
    //    if (i == b1.length - 1)
    //      output.write("\n");
    //    else
    //      output.write(" ");
    //  }
    //  for (int j = 0; j < W2[0].length; ++j)
    //    for (int i = 0; i < W2.length; ++i) {
    //      output.write("" + W2[i][j]);
    //      if (i == W2.length - 1)
    //        output.write("\n");
    //      else
    //        output.write(" ");
    //    }

    //  // Finish with pre-computation info
    //  for (int i = 0; i < preComputed.size(); ++i) {
    //    output.write("" + preComputed.get(i));
    //    if ((i + 1) % 100 == 0 || i == preComputed.size() - 1)
    //      output.write("\n");
    //    else
    //      output.write(" ");
    //  }

    //  output.close();
    //} catch (IOException e) {
    //  throw new RuntimeIOException(e);
    //}
  }

  /**
   * Convenience method; see {@link #loadFromModelFile(String, java.util.Properties)}.
   *
   * @see #loadFromModelFile(String, java.util.Properties)
   */
  //public static AMRParser loadFromModelFile(String modelFile) {
  //  return loadFromModelFile(modelFile, null);
  //}

  /**
   * Load a saved parser model.
   *
   * @param modelFile       Path to serialized model (may be GZipped)
   * @param extraProperties Extra test-time properties not already associated with model (may be null)
   *
   */
  //public static AMRParser loadFromModelFile(String modelFile, Properties extraProperties) {
  //  AMRParser parser = extraProperties == null ? new AMRParser() : new AMRParser(extraProperties);
  //  parser.loadModelFile(modelFile, false);
  //  return parser;
  //}

  /** Load a parser model file, printing out some messages about the grammar in the file.
   *
   *  @param modelFile The file (classpath resource, etc.) to load the model from.
   */
  public void loadModelFile(String modelFile) {
    loadModelFile(modelFile, true);
  }

  private void loadModelFile(String modelFile, boolean verbose) {
    Timing t = new Timing();
    //try {

    //  System.err.println("Loading depparse model file: " + modelFile + " ... ");
    //  String s;
    //  BufferedReader input = IOUtils.readerFromString(modelFile);

    //  s = input.readLine();
    //  int nDict = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int nPOS = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int nLabel = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int eSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int hSize = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int nTokens = Integer.parseInt(s.substring(s.indexOf('=') + 1));
    //  s = input.readLine();
    //  int nPreComputed = Integer.parseInt(s.substring(s.indexOf('=') + 1));

    //  knownWords = new ArrayList<String>();
    //  knownPos = new ArrayList<String>();
    //  knownLabels = new ArrayList<String>();
    //  double[][] E = new double[nDict + nPOS + nLabel][eSize];
    //  String[] splits;
    //  int index = 0;

    //  for (int k = 0; k < nDict; ++k) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    knownWords.add(splits[0]);
    //    for (int i = 0; i < eSize; ++i)
    //      E[index][i] = Double.parseDouble(splits[i + 1]);
    //    index = index + 1;
    //  }
    //  for (int k = 0; k < nPOS; ++k) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    knownPos.add(splits[0]);
    //    for (int i = 0; i < eSize; ++i)
    //      E[index][i] = Double.parseDouble(splits[i + 1]);
    //    index = index + 1;
    //  }
    //  for (int k = 0; k < nLabel; ++k) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    knownLabels.add(splits[0]);
    //    for (int i = 0; i < eSize; ++i)
    //      E[index][i] = Double.parseDouble(splits[i + 1]);
    //    index = index + 1;
    //  }
    //  generateIDs();

    //  double[][] W1 = new double[hSize][eSize * nTokens];
    //  for (int j = 0; j < W1[0].length; ++j) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    for (int i = 0; i < W1.length; ++i)
    //      W1[i][j] = Double.parseDouble(splits[i]);
    //  }

    //  double[] b1 = new double[hSize];
    //  s = input.readLine();
    //  splits = s.split(" ");
    //  for (int i = 0; i < b1.length; ++i)
    //    b1[i] = Double.parseDouble(splits[i]);

    //  double[][] W2 = new double[nLabel * 2 - 1][hSize];
    //  for (int j = 0; j < W2[0].length; ++j) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    for (int i = 0; i < W2.length; ++i)
    //      W2[i][j] = Double.parseDouble(splits[i]);
    //  }

    //  preComputed = new ArrayList<Integer>();
    //  while (preComputed.size() < nPreComputed) {
    //    s = input.readLine();
    //    splits = s.split(" ");
    //    for (String split : splits) {
    //      preComputed.add(Integer.parseInt(split));
    //    }
    //  }
    //  input.close();
    //  classifier = new Classifier(config, E, W1, b1, W2, preComputed);
    //} catch (IOException e) {
    //  throw new RuntimeIOException(e);
    //}

    // initialize the loaded parser
    //initialize(verbose);
    //t.done("Initializing dependency parser");
  }

  // TODO this should be a function which returns the embeddings array + embedID
  // otherwise the class needlessly carries around the extra baggage of `embeddings`
  // (never again used) for the entire training process
  private double[][] readEmbedFile(String embedFile, Map<String, Integer> embedID) {

    double[][] embeddings = null;
    if (embedFile != null) {
      BufferedReader input = null;
      try {
        input = IOUtils.readerFromString(embedFile);
        List<String> lines = new ArrayList<String>();
        for (String s; (s = input.readLine()) != null; ) {
          lines.add(s);
        }

        int nWords = lines.size();
        String[] splits = lines.get(0).split("\\s+");

        int dim = splits.length - 1;
        embeddings = new double[nWords][dim];
        System.err.println("Embedding File " + embedFile + ": #Words = " + nWords + ", dim = " + dim);

        if (dim != config.embeddingSize)
            throw new IllegalArgumentException("The dimension of embedding file does not match config.embeddingSize");

        for (int i = 0; i < lines.size(); ++i) {
          splits = lines.get(i).split("\\s+");
          embedID.put(splits[0], i);
          for (int j = 0; j < dim; ++j)
            embeddings[i][j] = Double.parseDouble(splits[j + 1]);
        }
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      } finally {
        IOUtils.closeIgnoringExceptions(input);
      }
    }

    if (embeddings != null)
      embeddings = Util.scaling(embeddings, 0, 1.0);
    else
      System.out.println("No embeddings loaded!");
    return embeddings;
  }

  /**
   * Train a new dependency parser model.
   *
   * @param trainDir Training data directory
   * @param devDir Development data directory (used for regular UAS evaluation
   *                of model)
   * @param modelFile String to which model should be saved
   * @param embedFile File containing word embeddings for words used in
   *                  training corpus
   */
  public void train(String trainDir, String devDir, String modelFile, String embedFile, String preModel) {

    System.err.println("Train directory: " + trainDir);
    System.err.println("Dev directory: " + devDir);
    System.err.println("Model File: " + modelFile);
    System.err.println("Embedding File: " + embedFile);
    System.err.println("Pre-trained Model File: " + preModel);

    //List<CoreMap> trainSents = new ArrayList<>();
    List<String[]> trainSents = new ArrayList<>();
    List<String[]> trainPoss = new ArrayList<>();
    List<AMRGraph> trainGraphs = new ArrayList<AMRGraph>();
    List<DependencyTree> trainTrees = new ArrayList<DependencyTree>();
    Util.loadAMRFile(trainDir, trainSents, trainPoss, trainTrees, trainGraphs);

    //List<CoreMap> devSents = new ArrayList<CoreMap>();
    List<String[]> devSents = new ArrayList<>();
    List<String[]> devPoss = new ArrayList<>();
    List<AMRGraph> devGraphs = new ArrayList<AMRGraph>();
    List<DependencyTree> devTrees = new ArrayList<DependencyTree>();

    if (devDir != null) {
      Util.loadAMRFile(devDir, devSents, devPoss, devTrees, devGraphs);
    }

    genDictionaries(trainSents, trainPoss, trainTrees, trainGraphs);
    generateConceptMap(trainSents, trainPoss, trainGraphs);

    //NOTE: remove -NULL-, and the pass it to ParsingSystem
    List<String> cDict = new ArrayList<>(knownConcepts);
    List<String> lDict = new ArrayList<>(knownArcs);
    Set<String> uSet = new HashSet<>(unalignedSet);

    system = new CacheTransition(cDict, lDict, uSet, config.CACHE);
    conceptIDDict = system.makeTransitions(wordConceptCounts);

    //Initialize a classifier; prepare for training.
    //Now we should have three independent classifiers.
    //Each should have a separate initialization.
    if (devDir != null) {
      System.err.println("Dev evaluation included");
      setupClassifierForTraining(trainSents, trainPoss, trainTrees, trainGraphs,
              devSents, devPoss, devTrees, devGraphs, embedFile, preModel);  //If has dev, also evaluate on the dev set
    }
    else
      setupClassifierForTraining(trainSents, trainPoss, trainTrees, trainGraphs, embedFile, preModel);

    config.printParameters();

    long startTime = System.currentTimeMillis();
    ///**
    // * Track the best UAS performance we've seen.
    // */
    double bestUAS = 0;
    //System.exit(1);

    //System.err.println(system.arcTransitionIDs);
    //System.err.println(system.arcTransitions);
    //System.err.println(system.arcLabels);

    //System.err.println(conceptIDClassifier.numLabels);
    //System.err.println(arcConnectClassifier.numLabels);
    //System.err.println(pushIndexClassifier.numLabels);
    //System.exit(1);

    for (int iter = 0; iter < config.maxIter; ++iter) {
      if (iter % 100 == 0)
        System.err.println("##### Iteration " + iter);

      //Classifier.Cost cost = classifier.computeCostFunction(config.batchSize, config.regParameter, config.dropProb);
      //System.out.println("Concept id transitions:"+ conceptIDClassifier.numLabels);
      if (config.conceptID) {
        Classifier.Cost conceptIDCost = conceptIDClassifier.computeCostFunction(config.batchSize, config.regParameter, config.dropProb);
        if (iter % 100 == 0) {
          System.err.println("concept ID Cost = " + conceptIDCost.getCost() + ", Correct(%) = " + conceptIDCost.getPercentCorrect());
          if (devDir != null) {
            double acc = conceptIDClassifier.computeAccuracy(config.regParameter, config.dropProb);
            System.err.println("Concept identification Iter " + iter + " on dev: " + acc);
          }
        }
        conceptIDClassifier.takeAdaGradientStep(conceptIDCost, config.adaAlpha, config.adaEps);
        if (config.clearGradientsPerIter > 0 && iter % config.clearGradientsPerIter == 0) {
          System.err.println("Clearing gradient histories..");
          conceptIDClassifier.clearGradientHistories();
        }
      }

      if (config.arcConnect) {
        Classifier.Cost arcConnectCost = arcConnectClassifier.computeCostFunction(config.batchSize, config.regParameter, config.dropProb);
        if (iter % 100 == 0) {
          System.err.println("arc connect Cost = " + arcConnectCost.getCost() + ", Correct(%) = " + arcConnectCost.getPercentCorrect());
          if (devDir != null) {
            double acc = arcConnectClassifier.computeAccuracy(config.regParameter, config.dropProb);
            System.err.println("ARC Iter " + iter + " on dev: " + acc);
          }
        }
        arcConnectClassifier.takeAdaGradientStep(arcConnectCost, config.adaAlpha, config.adaEps);
        if (config.clearGradientsPerIter > 0 && iter % config.clearGradientsPerIter == 0) {
          System.err.println("Clearing gradient histories..");
          arcConnectClassifier.clearGradientHistories();
        }
      }

      if (config.pushIndex) {
        Classifier.Cost pushIndexCost = pushIndexClassifier.computeCostFunction(config.batchSize, config.regParameter, config.dropProb);
        if (iter % 100 == 0) {
          System.err.println("push index Cost = " + pushIndexCost.getCost() + ", Correct(%) = " + pushIndexCost.getPercentCorrect());
          if (devDir != null) {
            double acc = pushIndexClassifier.computeAccuracy(config.regParameter, config.dropProb);
            System.err.println("PUSH index Iter " + iter + " on dev: " + acc);
          }
        }
        pushIndexClassifier.takeAdaGradientStep(pushIndexCost, config.adaAlpha, config.adaEps);
        if (config.clearGradientsPerIter > 0 && iter % config.clearGradientsPerIter == 0) {
          System.err.println("Clearing gradient histories..");
          pushIndexClassifier.clearGradientHistories();
        }
      }

      if (iter % 100 == 0)
        System.err.println("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");

      // UAS evaluation
      if (devDir != null && iter % config.evalPerIter == 0) {
      //  // Redo precomputation with updated weights. This is only
      //  // necessary because we're updating weights -- for normal
      //  // prediction, we just do this once in #initialize
      //  classifier.preCompute();

      //  List<AMRGraph> predicted = devSents.stream().map(this::predictInner).collect(toList());

      //  double uas = config.noPunc ? system.getUASnoPunc(devSents, predicted, devTrees) : system.getUAS(devSents, predicted, devTrees);
      //  System.err.println("UAS: " + uas);

      //  if (config.saveIntermediate && uas > bestUAS) {
      //    System.err.printf("Exceeds best previous UAS of %f. Saving model file..%n", bestUAS);

      //    bestUAS = uas;
      //    writeModelFile(modelFile);
      //  }
      }
      //writeModelFile(modelFile);
    }

    //classifier.finalizeTraining();

    //if (devFile != null) {
    //  // Do final UAS evaluation and save if final model beats the
    //  // best intermediate one
    //  List<DependencyTree> predicted = devSents.stream().map(this::predictInner).collect(toList());
    //  double uas = config.noPunc ? system.getUASnoPunc(devSents, predicted, devTrees) : system.getUAS(devSents, predicted, devTrees);

    //  if (uas > bestUAS) {
    //    System.err.printf("Final model UAS: %f%n", uas);
    //    System.err.printf("Exceeds best previous UAS of %f. Saving model file..%n", bestUAS);

    //    writeModelFile(modelFile);
    //  }
    //} else {
    //  writeModelFile(modelFile);
    //}
  }

  /**
  * @see #train(String, String, String, String, String)
  */
  public void train(String trainFile, String devFile, String modelFile, String embedFile) {
    train(trainFile, devFile, modelFile, embedFile, null);
  }

  /**
   * @see #train(String, String, String, String)
   */
  public void train(String trainFile, String devFile, String modelFile) {
    train(trainFile, devFile, modelFile, null);
  }

  /**
   * @see #train(String, String, String)
   */
  public void train(String trainFile, String modelFile) {
    train(trainFile, null, modelFile);
  }

  public void setupClassifierForTraining(List<String[]> trainSents, List<String[]> trainPoss, List<DependencyTree> trainTrees,
                                          List<AMRGraph> trainGraphs, String embedFile, String preModel) {
    setupClassifierForTraining(trainSents, trainPoss, trainTrees, trainGraphs, null, null, null, null, embedFile, preModel);
  }

  /**
   * Prepare a classifier for training with the given dataset.
   * Three different classifiers for three different procedures.
   */
  private void setupClassifierForTraining(List<String[]> trainSents, List<String[]> trainPoss, List<DependencyTree> trainTrees,
                                          List<AMRGraph> trainGraphs, List<String[]> devSents, List<String[]> devPoss, List<DependencyTree> devTrees,
                                          List<AMRGraph> devGraphs,
                                          String embedFile, String preModel) {

    int numEmbeddings = knownWords.size()+ knownPos.size()+ knownDeps.size()+
            knownArcs.size()+ knownConcepts.size(); //Number of embeddings
    int conceptIDInputNum = 12, conceptIDHiddenSize = config.hiddenSize;
    int arcConnectInputNum = 41, arcConnectHiddenSize = config.hiddenSize;
    int pushIndexInputNum = 12, pushIndexHiddenSize = config.hiddenSize;

    int conceptIDOutputSize = system.numTransitions(0);
    int arcConnectOutputSize = system.numTransitions(1);
    int pushIndexOutputSize = system.numTransitions(2);

    NeuralParam conceptIDParam = new NeuralParam(numEmbeddings, config.embeddingSize, conceptIDInputNum, conceptIDHiddenSize,
            conceptIDOutputSize, Config.conIDTokens);

    conceptIDParam.randomInitialize(config.initRange);

    NeuralParam arcConnectParam = new NeuralParam(numEmbeddings, config.embeddingSize, arcConnectInputNum, arcConnectHiddenSize,
            arcConnectOutputSize, Config.arcConnectTokens);
    arcConnectParam.randomInitialize(config.initRange);
    NeuralParam pushIndexParam = new NeuralParam(numEmbeddings, config.embeddingSize, pushIndexInputNum, pushIndexHiddenSize,
            pushIndexOutputSize, Config.pushIndexTokens);
    pushIndexParam.randomInitialize(config.initRange);

    // Read embeddings into `embedID`, `embeddings`
    Map<String, Integer> embedID = new HashMap<String, Integer>();
    double[][] embeddings = readEmbedFile(embedFile, embedID);

    // Try to match loaded embeddings with words in dictionary
    Random random = Util.getRandom();
    int foundEmbed = 0;
    //for (int i = 0; i < E.length; ++i) {
    for (int i = 0; i < numEmbeddings; ++i) {
      int index = -1;
      if (i < knownWords.size()) {
        String str = knownWords.get(i);
        if (embedID.containsKey(str)) index = embedID.get(str);
        else if (embedID.containsKey(str.toLowerCase())) index = embedID.get(str.toLowerCase());
      }
      if (index >= 0) {
        ++foundEmbed;
        for (int j = 0; j < config.embeddingSize; ++j) {
          double val = embeddings[index][j];
          conceptIDParam.setEmbedding(i, j, val);
          arcConnectParam.setEmbedding(i, j, val);
          pushIndexParam.setEmbedding(i, j, val);
        }
      } else {
        for (int j = 0; j < config.embeddingSize; ++j) {
          //E[i][j] = random.nextDouble() * config.initRange * 2 - config.initRange;
          //E[i][j] = random.nextDouble() * 0.2 - 0.1;
          //E[i][j] = random.nextGaussian() * Math.sqrt(0.1);
          double val = random.nextDouble() * 0.02 - 0.01;
          conceptIDParam.setEmbedding(i, j, val);
          arcConnectParam.setEmbedding(i, j, val);
          pushIndexParam.setEmbedding(i, j, val);
        }
      }
    }
    System.err.println("Found embeddings: " + foundEmbed + " / " + knownWords.size());

    AMRData trainSet = genTrainExamples(trainSents, trainPoss, trainTrees, trainGraphs, true);
    AMRData devSet = null;
    if (trainSents != null) {
      devSet = genTrainExamples(devSents, devPoss, devTrees, devGraphs, false);
    }

    //if (devSet == null) {
    //  conceptIDClassifier = new Classifier(config, trainSet.conceptIDExamples, conceptIDParam, preComputed, conceptIDDict, null);
    //  arcConnectClassifier = new Classifier(config, trainSet.arcConnectExamples, arcConnectParam, preComputed, null, null);
    //  pushIndexClassifier = new Classifier(config, trainSet.pushPopExamples, pushIndexParam, preComputed, null, null);
    //}
    //else {
      conceptIDClassifier = new Classifier(config, trainSet.conceptIDExamples, conceptIDParam, preComputed, conceptIDDict, devSet.conceptIDExamples);
      arcConnectClassifier = new Classifier(config, trainSet.arcConnectExamples, arcConnectParam, preComputed, null, devSet.arcConnectExamples);
      pushIndexClassifier = new Classifier(config, trainSet.pushPopExamples, pushIndexParam, preComputed, null, devSet.pushPopExamples);
    //}
    //classifier = new Classifier(config, trainSet, E, W1, b1, W2, preComputed);
  }

  /**
   * Determine the dependency parse of the given sentence.
   * <p>
   * This "inner" method returns a structure unique to this package; use {@link #predict(int[] tok)}
   * for general parsing purposes.
   */
  private AMRGraph predictInner(int[] tokSeq) {
    int numTrans = system.numTransitions(2);

    AMRConfiguration c = system.initialConfiguration(tokSeq);
    while (!system.isTerminal(c)) {
      //double[] scores = classifier.computeScores(getFeatureArray(c));
      double [] scores = null;

      double optScore = Double.NEGATIVE_INFINITY;
      String optTrans = null;

      for (int j = 0; j < numTrans; ++j) {
        if (scores[j] > optScore && system.canApply(c, system.transitions.get(j))) {
          optScore = scores[j];
          optTrans = system.transitions.get(j);
        }
      }
      system.apply(c, optTrans);
    }
    //return c.tree;
    return null;
  }

  /**
   * Determine the dependency parse of the given sentence using the loaded model.
   * You must first load a parser before calling this method.
   *
   * @throws java.lang.IllegalStateException If parser has not yet been loaded and initialized
   */
  public GrammaticalStructure predict(int[] sentence) {
    if (system == null)
      throw new IllegalStateException("Parser has not been  " +
          "loaded and initialized; first load a model.");

    AMRGraph result = predictInner(sentence);

    // The rest of this method is just busy-work to convert the
    // package-local representation into a CoreNLP-standard
    // GrammaticalStructure.

    //List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    //List<TypedDependency> dependencies = new ArrayList<>();

    //IndexedWord root = new IndexedWord(new Word("ROOT"));
    //root.set(CoreAnnotations.IndexAnnotation.class, 0);

    //for (int i = 1; i <= result.n; i++) {
    //  int head = result.getHead(i);
    //  String label = result.getLabel(i);

    //  IndexedWord thisWord = new IndexedWord(tokens.get(i - 1));
    //  IndexedWord headWord = head == 0 ? root
    //                                   : new IndexedWord(tokens.get(head - 1));

    //  GrammaticalRelation relation = head == 0
    //                                 ? GrammaticalRelation.ROOT
    //                                 : makeGrammaticalRelation(label);

    //  dependencies.add(new TypedDependency(relation, headWord, thisWord));
    //}

    // Build GrammaticalStructure
    // TODO ideally submodule should just return GrammaticalStructure
    //TreeGraphNode rootNode = new TreeGraphNode(root);
    //return makeGrammaticalStructure(dependencies, rootNode);
    return null;
  }

  private GrammaticalRelation makeGrammaticalRelation(String label) {
    GrammaticalRelation stored;

    switch (language) {
      case English:
        stored = EnglishGrammaticalRelations.shortNameToGRel.get(label);
        if (stored != null)
          return stored;
        break;
      case UniversalEnglish:
        stored = UniversalEnglishGrammaticalRelations.shortNameToGRel.get(label);
        if (stored != null)
          return stored;
        break;
      case Chinese:
        stored = ChineseGrammaticalRelations.shortNameToGRel.get(label);
        if (stored != null)
          return stored;
        break;
    }

    return new GrammaticalRelation(language, label, null, GrammaticalRelation.DEPENDENT);
  }

  private GrammaticalStructure makeGrammaticalStructure(List<TypedDependency> dependencies, TreeGraphNode rootNode) {
    switch (language) {
      case English: return new EnglishGrammaticalStructure(dependencies, rootNode);
      case UniversalEnglish: return new UniversalEnglishGrammaticalStructure(dependencies, rootNode);
      case Chinese: return new ChineseGrammaticalStructure(dependencies, rootNode);

      // TODO suboptimal: default to UniversalEnglishGrammaticalStructure return
      default: return new UniversalEnglishGrammaticalStructure(dependencies, rootNode);
    }
  }

  /*
  public GrammaticalStructure predict(List<? extends HasWord> sentence) {
    CoreLabel sentenceLabel = new CoreLabel();
    List<CoreLabel> tokens = new ArrayList<>();

    int i = 1;
    for (HasWord wd : sentence) {
      CoreLabel label;
      if (wd instanceof CoreLabel) {
        label = (CoreLabel) wd;
        if (label.tag() == null)
          throw new IllegalArgumentException("Parser requires words " +
              "with part-of-speech tag annotations");
      } else {
        label = new CoreLabel();
        label.setValue(wd.word());
        label.setWord(wd.word());

        if (!(wd instanceof HasTag))
          throw new IllegalArgumentException("Parser requires words " +
              "with part-of-speech tag annotations");

        label.setTag(((HasTag) wd).tag());
      }

      label.setIndex(i);
      i++;

      tokens.add(label);
    }

    sentenceLabel.set(CoreAnnotations.TokensAnnotation.class, tokens);

    return predict(sentenceLabel);
  }
  */

  //TODO: support sentence-only files as input

  /** Run the parser in the modelFile on a testFile and perhaps save output.
   *
   *  testFile File to parse. In CoNLL-X format. Assumed to have gold answers included.
   *  outFile File to write results to in CoNLL-X format.  If null, no output is written
   *  @return The LAS score on the dataset
   */
  /*
  public double testCoNLL(String testFile, String outFile) {
    System.err.println("Test File: " + testFile);
    Timing timer = new Timing();
    List<CoreMap> testSents = new ArrayList<>();
    List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
    Util.loadConllFile(testFile, testSents, testTrees, config.unlabeled, config.cPOS);

    // count how much to parse
    int numWords = 0;
    int numOOVWords = 0;
    int numSentences = 0;
    for (CoreMap testSent : testSents) {
      numSentences += 1;
      List<CoreLabel> tokens = testSent.get(CoreAnnotations.TokensAnnotation.class);
      for (int k = 0; k < tokens.size(); ++ k) {  
        String word = tokens.get(k).word();
        numWords += 1;
        if (!wordIDs.containsKey(word))
          numOOVWords += 1;
      }
    }
    System.err.printf("OOV Words: %d / %d = %.2f%%\n", numOOVWords, numWords, numOOVWords * 100.0 / numWords);

    List<DependencyTree> predicted = testSents.stream().map(this::predictInner).collect(toList());
    //Map<String, Double> result = system.evaluate(testSents, predicted, testTrees);
    Map<String, Double> result = null;
    
    double uas = config.noPunc ? result.get("UASnoPunc") : result.get("UAS");
    double las = config.noPunc ? result.get("LASnoPunc") : result.get("LAS");
    System.err.printf("UAS = %.4f%n", uas);
    System.err.printf("LAS = %.4f%n", las);

    long millis = timer.stop();
    double wordspersec = numWords / (((double) millis) / 1000);
    double sentspersec = numSentences / (((double) millis) / 1000);
    System.err.printf("%s parsed %d words in %d sentences in %.1fs at %.1f w/s, %.1f sent/s.%n",
            StringUtils.getShortClassName(this), numWords, numSentences, millis / 1000.0, wordspersec, sentspersec);

    if (outFile != null) {
        Util.writeConllFile(outFile, testSents, predicted);
    }
    return las;
  }
  */

  /*
  private void parseTextFile(BufferedReader input, PrintWriter output) {
    DocumentPreprocessor preprocessor = new DocumentPreprocessor(input);
    preprocessor.setSentenceFinalPuncWords(config.tlp.sentenceFinalPunctuationWords());
    preprocessor.setEscaper(config.escaper);
    preprocessor.setSentenceDelimiter(config.sentenceDelimiter);
    preprocessor.setTokenizerFactory(config.tlp.getTokenizerFactory());

    Timing timer = new Timing();

    MaxentTagger tagger = new MaxentTagger(config.tagger);
    List<List<TaggedWord>> tagged = new ArrayList<>();
    for (List<HasWord> sentence : preprocessor) {
      tagged.add(tagger.tagSentence(sentence));
    }

    System.err.printf("Tagging completed in %.2f sec.%n",
        timer.stop() / 1000.0);

    timer.start();

    int numSentences = 0;
    for (List<TaggedWord> taggedSentence : tagged) {
      GrammaticalStructure parse = predict(taggedSentence);

      Collection<TypedDependency> deps = parse.typedDependencies();
      for (TypedDependency dep : deps)
        output.println(dep);
      output.println();

      numSentences++;
    }

    long millis = timer.stop();
    double seconds = millis / 1000.0;
    System.err.printf("Parsed %d sentences in %.2f seconds (%.2f sents/sec).%n",
        numSentences, seconds, numSentences / seconds);
  }
  */

  /**
   * Prepare for parsing after a model has been loaded.
   */
  //private void initialize(boolean verbose) {
  //  if (knownLabels == null)
  //    throw new IllegalStateException("Model has not been loaded or trained");

  //  // NOTE: remove -NULL-, and then pass the label set to the ParsingSystem
  //  List<String> lDict = new ArrayList<>(knownLabels);
  //  lDict.remove(0);

  //  system = new ArcStandard(config.tlp, lDict, verbose);

  //  // Pre-compute matrix multiplications
  //  if (config.numPreComputed > 0) {
  //    classifier.preCompute();
  //  }
  //}

  /**
   * Explicitly specifies the number of arguments expected with
   * particular command line options.
   */
  private static final Map<String, Integer> numArgs = new HashMap<>();
  static {
    numArgs.put("textFile", 1);
    numArgs.put("outFile", 1);
  }

  /**
   * A main program for training, testing and using the parser.
   *
   * <p>
   * You can use this program to train new parsers from treebank data,
   * evaluate on test treebank data, or parse raw text input.
   *
   * <p>
   * Sample usages:
   * <ul>
   *   <li>
   *     <strong>Train a parser with CoNLL treebank data:</strong>
   *     <code>java edu.stanford.nlp.parser.nndep.DependencyParser -trainFile trainPath -devFile devPath -embedFile wordEmbeddingFile -embeddingSize wordEmbeddingDimensionality -model modelOutputFile.txt.gz</code>
   *   </li>
   *   <li>
   *     <strong>Parse raw text from a file:</strong>
   *     <code>java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile rawTextToParse -outFile dependenciesOutputFile.txt</code>
   *   </li>
   *   <li>
   *     <strong>Parse raw text from standard input, writing to standard output:</strong>
   *     <code>java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile - -outFile -</code>
   *   </li>
   * </ul>
   *
   * <p>
   * See below for more information on all of these training / test options and more.
   *
   * <p>
   * Input / output options:
   * <table>
   *   <tr><th>Option</th><th>Required for training</th><th>Required for testing / parsing</th><th>Description</th></tr>
   *   <tr><td><tt>&#8209;devFile</tt></td><td>Optional</td><td>No</td><td>Path to a development-set treebank in <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a>. If provided, the </td></tr>
   *   <tr><td><tt>&#8209;embedFile</tt></td><td>Optional (highly recommended!)</td><td>No</td><td>A word embedding file, containing distributed representations of English words. Each line of the provided file should contain a single word followed by the elements of the corresponding word embedding (space-delimited). It is not absolutely necessary that all words in the treebank be covered by this embedding file, though the parser's performance will generally improve if you are able to provide better embeddings for more words.</td></tr>
   *   <tr><td><tt>&#8209;model</tt></td><td>Yes</td><td>Yes</td><td>Path to a model file. If the path ends in <tt>.gz</tt>, the model will be read as a Gzipped model file. During training, we write to this path; at test time we read a pre-trained model from this path.</td></tr>
   *   <tr><td><tt>&#8209;textFile</tt></td><td>No</td><td>Yes (or <tt>testFile</tt>)</td><td>Path to a plaintext file containing sentences to be parsed.</td></tr>
   *   <tr><td><tt>&#8209;testFile</tt></td><td>No</td><td>Yes (or <tt>textFile</tt>)</td><td>Path to a test-set treebank in <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a> for final evaluation of the parser.</td></tr>
   *   <tr><td><tt>&#8209;trainFile</tt></td><td>Yes</td><td>No</td><td>Path to a training treebank in <a href="http://ilk.uvt.nl/conll/#dataformat">CoNLL-X format</a></td></tr>
   * </table>
   *
   * Training options:
   * <table>
   *   <tr><th>Option</th><th>Default</th><th>Description</th></tr>
   *   <tr><td><tt>&#8209;adaAlpha</tt></td><td>0.01</td><td>Global learning rate for AdaGrad training</td></tr>
   *   <tr><td><tt>&#8209;adaEps</tt></td><td>1e-6</td><td>Epsilon value added to the denominator of AdaGrad update expression for numerical stability</td></tr>
   *   <tr><td><tt>&#8209;batchSize</tt></td><td>10000</td><td>Size of mini-batch used for training</td></tr>
   *   <tr><td><tt>&#8209;clearGradientsPerIter</tt></td><td>0</td><td>Clear AdaGrad gradient histories every <em>n</em> iterations. If zero, no gradient clearing is performed.</td></tr>
   *   <tr><td><tt>&#8209;dropProb</tt></td><td>0.5</td><td>Dropout probability. For each training example we randomly choose some amount of units to disable in the neural network classifier. This parameter controls the proportion of units "dropped out."</td></tr>
   *   <tr><td><tt>&#8209;embeddingSize</tt></td><td>50</td><td>Dimensionality of word embeddings provided</td></tr>
   *   <tr><td><tt>&#8209;evalPerIter</tt></td><td>100</td><td>Run full UAS (unlabeled attachment score) evaluation every time we finish this number of iterations. (Only valid if a development treebank is provided with <tt>&#8209;devFile</tt>.)</td></tr>
   *   <tr><td><tt>&#8209;hiddenSize</tt></td><td>200</td><td>Dimensionality of hidden layer in neural network classifier</td></tr>
   *   <tr><td><tt>&#8209;initRange</tt></td><td>0.01</td><td>Bounds of range within which weight matrix elements should be initialized. Each element is drawn from a uniform distribution over the range <tt>[-initRange, initRange]</tt>.</td></tr>
   *   <tr><td><tt>&#8209;maxIter</tt></td><td>20000</td><td>Number of training iterations to complete before stopping and saving the final model.</td></tr>
   *   <tr><td><tt>&#8209;numPreComputed</tt></td><td>100000</td><td>The parser pre-computes hidden-layer unit activations for particular inputs words at both training and testing time in order to speed up feedforward computation in the neural network. This parameter determines how many words for which we should compute hidden-layer activations.</td></tr>
   *   <tr><td><tt>&#8209;regParameter</tt></td><td>1e-8</td><td>Regularization parameter for training</td></tr>
   *   <tr><td><tt>&#8209;saveIntermediate</tt></td><td><tt>true</tt></td><td>If <tt>true</tt>, continually save the model version which gets the highest UAS value on the dev set. (Only valid if a development treebank is provided with <tt>&#8209;devFile</tt>.)</td></tr>
   *   <tr><td><tt>&#8209;trainingThreads</tt></td><td>1</td><td>Number of threads to use during training. Note that depending on training batch size, it may be unwise to simply choose the maximum amount of threads for your machine. On our 16-core test machines: a batch size of 10,000 runs fastest with around 6 threads; a batch size of 100,000 runs best with around 10 threads.</td></tr>
   *   <tr><td><tt>&#8209;wordCutOff</tt></td><td>1</td><td>The parser can optionally ignore rare words by simply choosing an arbitrary "unknown" feature representation for words that appear with frequency less than <em>n</em> in the corpus. This <em>n</em> is controlled by the <tt>wordCutOff</tt> parameter.</td></tr>
   * </table>
   *
   * Runtime parsing options:
   * <table>
   *   <tr><th>Option</th><th>Default</th><th>Description</th></tr>
   *   <tr><td><tt>&#8209;escaper</tt></td><td>N/A</td><td>Only applicable for testing with <tt>-textFile</tt>. If provided, use this word-escaper when parsing raw sentences. (Should be a fully-qualified class name like <tt>edu.stanford.nlp.trees.international.arabic.ATBEscaper</tt>.)</td></tr>
   *   <tr><td><tt>&#8209;numPreComputed</tt></td><td>100000</td><td>The parser pre-computes hidden-layer unit activations for particular inputs words at both training and testing time in order to speed up feedforward computation in the neural network. This parameter determines how many words for which we should compute hidden-layer activations.</td></tr>
   *   <tr><td><tt>&#8209;sentenceDelimiter</tt></td><td>N/A</td><td>Only applicable for testing with <tt>-textFile</tt>.  If provided, assume that the given <tt>textFile</tt> has already been sentence-split, and that sentences are separated by this delimiter.</td></tr>
   *   <tr><td><tt>&#8209;tagger.model</tt></td><td>edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger</td><td>Only applicable for testing with <tt>-textFile</tt>. Path to a part-of-speech tagger to use to pre-tag the raw sentences before parsing.</td></tr>
   * </table>
   */
  public static void main(String[] args) {
    Properties props = StringUtils.argsToProperties(args, numArgs);
    AMRParser parser = new AMRParser(props);
    System.out.println("Start AMR parser");
    // Train with CoNLL-X data
    if (props.containsKey("trainDir"))
      parser.train(props.getProperty("trainDir"), props.getProperty("devDir"), props.getProperty("model"),
          props.getProperty("embedFile"), props.getProperty("preModel"));

    boolean loaded = false;
    // Test with CoNLL-X data
    //if (props.containsKey("testFile")) {
    //  parser.loadModelFile(props.getProperty("model"));
    //  loaded = true;
    //  parser.testCoNLL(props.getProperty("testFile"), props.getProperty("outFile"));
    //}

    // Parse raw text data
    //if (props.containsKey("textFile")) {
    //  if (!loaded) {
    //    parser.loadModelFile(props.getProperty("model"));
    //    loaded = true;
    //  }

    //  String encoding = parser.config.tlp.getEncoding();
    //  String inputFilename = props.getProperty("textFile");
    //  BufferedReader input;
    //  try {
    //    input = inputFilename.equals("-")
    //            ? IOUtils.readerFromStdin(encoding)
    //            : IOUtils.readerFromString(inputFilename, encoding);
    //  } catch (IOException e) {
    //    throw new RuntimeIOException("No input file provided (use -textFile)", e);
    //  }

    //  String outputFilename = props.getProperty("outFile");
    //  PrintWriter output;
    //  try {
    //    output = outputFilename == null || outputFilename.equals("-")
    //        ? IOUtils.encodedOutputStreamPrintWriter(System.out, encoding, true)
    //        : IOUtils.getPrintWriter(outputFilename, encoding);
    //  } catch (IOException e) {
    //    throw new RuntimeIOException("Error opening output file", e);
    //  }

    //  parser.parseTextFile(input, output);
    //}
  }
}
