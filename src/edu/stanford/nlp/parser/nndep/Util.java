
/*
* 	@Author:  Danqi Chen
* 	@Email:  danqi@cs.stanford.edu
*	@Created:  2014-08-25
* 	@Last Modified:  2014-10-05
*/

package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

import java.util.*;
import java.io.*;


/**
 *
 *  Some utility functions
 *
 *  @author Danqi Chen
 *  @author Jon Gauthier
 */

class Util {

  private Util() {} // static methods

  private static Random random;

  /**
   * Normalize word embeddings by setting mean = rMean, std = rStd
   */
  public static double[][] scaling(double[][] A, double rMean, double rStd) {
    int count = 0;
    double mean = 0.0;
    double std = 0.0;
    for (int i = 0; i < A.length; ++ i)
      for (int j = 0; j < A[i].length; ++ j) {
        count += 1;
        mean += A[i][j];
        std += A[i][j] * A[i][j];
      }
    mean = mean / count;
    std = Math.sqrt(std / count - mean * mean);

    System.err.printf("Scaling word embeddings:");
    System.err.printf("(mean = %.2f, std = %.2f) -> (mean = %.2f, std = %.2f)", mean, std, rMean, rStd);

    double[][] rA = new double[A.length][A[0].length];
    for (int i = 0; i < rA.length; ++ i)
      for (int j = 0; j < rA[i].length; ++ j)
        rA[i][j] = (A[i][j] - mean) * rStd / std + rMean;
    return rA;
  }

  /**
   *  Normalize word embeddings by setting mean = 0, std = 1
   */
  public static double[][] scaling(double[][] A) {
    return scaling(A, 0.0, 1.0);
  }

  // return strings sorted by frequency, and filter out those with freq. less than cutOff.

  /**
   * Build a dictionary of words collected from a corpus.
   * <p>
   * Filters out words with a frequency below the given {@code cutOff}.
   *
   * @return Words sorted by decreasing frequency, filtered to remove
   *         any words with a frequency below {@code cutOff}
   */
  public static List<String> generateDict(List<String> str, int cutOff)
  {
    Counter<String> freq = new IntCounter<>();
    for (String aStr : str)
      freq.incrementCount(aStr);

    List<String> keys = Counters.toSortedList(freq, false);
    List<String> dict = new ArrayList<>();
    for (String word : keys) {
      if (freq.getCount(word) >= cutOff)
        dict.add(word);
    }
    return dict;
  }

  public static List<String> topDict(List<String> str, int k)
  {
    Counter<String> freq = new IntCounter<>();
    for (String aStr : str)
      freq.incrementCount(aStr);

    List<String> keys = Counters.toSortedList(freq, false);
    List<String> dict = new ArrayList<>();
    for (int i = 0; i < k; i++) {
      dict.add(keys.get(i));
    }
    return dict;
  }

  public static List<String> generateDict(List<String> str)
  {
    return generateDict(str, 1);
  }

  /**
   * @return Shared random generator used in this package
   */
  static Random getRandom() {
    if (random != null)
      return random;
    else
      return getRandom(System.currentTimeMillis());
  }

  /**
   * Set up shared random generator to use the given seed.
   *
   * @return Shared random generator object
   */
  static Random getRandom(long seed) {
    random = new Random(seed);
    System.err.printf("Random generator initialized with seed %d%n", seed);

    return random;
  }

  public static <T> List<T> getRandomSubList(List<T> input, int subsetSize)
  {
    int inputSize = input.size();
    if (subsetSize > inputSize)
      subsetSize = inputSize;

    Random random = getRandom();
    for (int i = 0; i < subsetSize; i++)
    {
      int indexToSwap = i + random.nextInt(inputSize - i);
      T temp = input.get(i);
      input.set(i, input.get(indexToSwap));
      input.set(indexToSwap, temp);
    }
    return input.subList(0, subsetSize);
  }

  public static void loadAMRFile(String inDir, List<String[]> tokens, List<String[]> posTags, List<DependencyTree> trees, List<AMRGraph> graphs)
  {
    //CoreLabelTokenFactory tf = new CoreLabelTokenFactory(false);
    String tokFile = inDir + "/tok";
    String posFile = inDir + "/pos";
    String depFile = inDir + "/dep";
    String amrFile = inDir + "/amr";

    BufferedReader reader = null;

    //Map<String, Map<String, Integer>> tokToConcept = new HashMap<>();

    try {
      List<String[]> all_toks = IOUtils.readToks(tokFile);
      List<String[]> all_poss = IOUtils.readToks(posFile);

      loadConllFile(depFile, trees, all_toks);

      reader = IOUtils.readerFromString(amrFile);

      int sentIndex = 0;
      AMRGraph graph = new AMRGraph();
      Set<Integer> visited = new HashSet<>();
      int rootNum = 0;
      for (String line : IOUtils.getLineIterable(reader, false)) {
        String[] splits = line.trim().split(" ");

        if (splits.length < 2) { //A new sentence
          assert rootNum == 1;
          graph.buildEdgeMap();
          graph.buildWordToIndices();
          //graph.setSentence(all_toks.get(sentIndex));
          graphs.add(graph);

          graph = new AMRGraph();
          rootNum = 0;
        }
        else if (splits.length == 2) { //Sentence index, used for parse sentence info
          if (!splits[0].equals("sentence")) {
            System.exit(-1);
          }
          //sentIndex = Integer.parseInt(splits[1]);
          //if (sentIndex > 0)
          //  break;
          String[] toks = all_toks.get(sentIndex);
          String[] poss = all_poss.get(sentIndex);
          tokens.add(toks);
          posTags.add(poss);
          visited.clear();
          sentIndex += 1;
        }
        else {
          if (splits.length != 6) {
            System.err.println("Length inconsistent in the conll format" + " " + splits.length);
            System.err.println(StringUtils.join(splits));
            System.exit(1);
          }
          int conceptId = Integer.parseInt(splits[0]);
          boolean isVar = Boolean.parseBoolean(splits[1]);
          String concept = splits[2], wordIndex = splits[3],
                  outGoRels = splits[4], parRels = splits[5];

          ConceptLabel c = new ConceptLabel(concept);
          c.setVar(isVar);
          if (wordIndex.equals("NONE")) {
            c.aligned = false;
          }
          else {
            c.aligned = true;
            splits = wordIndex.split("#");
            boolean align = false;
            for (String s: splits) { //Each concept only aligns to a single word
              int wordId = Integer.parseInt(s);
              if (!visited.contains(wordId)) {
                c.addWord(wordId);
                align = true;
                visited.add(wordId);
                break;
              }
            }
            if (!align) {
              c.aligned = false;
            }
          }

          //Processing outgoing relations
          if (!outGoRels.equals("NONE")) {
            splits = outGoRels.split("#");
            for (String s: splits) {
              String[] fields = s.split(":");
              c.rels.add(fields[0]);
              c.tails.add(Integer.parseInt(fields[1]));
            }
          }

          //Processing parent relations
          if (parRels.equals("NONE")) { //Only root of the graph has not parent
            graph.setRoot(conceptId);
            rootNum += 1;
          }
          else {
            splits = parRels.split("#");
            for (String s: splits) {
              String[] fields = s.split(":");
              c.parRels.add(fields[0]);
              c.parConcepts.add(Integer.parseInt(fields[1]));
            }
          }

          c.buildRelMap();
          graph.add(c);
        }
      }
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  //Here we load the dependency trees for the input token sequence, will be used for feature extraction
  public static void loadConllFile(String inFile, List<DependencyTree> trees, List<String[]> toks)
  {
    BufferedReader reader = null;
    try {
      reader = IOUtils.readerFromString(inFile);

      //CoreMap sentence = new CoreLabel();
      //List<CoreLabel> sentenceTokens = new ArrayList<>();

      DependencyTree tree = new DependencyTree();
      int sent_index = 0;
      int tok_index = 0;
      String[] tokSeq = toks.get(sent_index);

      for (String line : IOUtils.getLineIterable(reader, false)) {
        String[] splits = line.split("\t");
        if (splits.length < 10) {
          trees.add(tree);
          //sentence.set(CoreAnnotations.TokensAnnotation.class, sentenceTokens);
          //sents.add(sentence);

          tree = new DependencyTree();
          sent_index += 1;
          if (sent_index < toks.size())
            tokSeq = toks.get(sent_index);
          //sentence = new CoreLabel();
          //sentenceTokens = new ArrayList<>();
        } else {
          String word = splits[1],
                  pos = splits[3],
                  depType = splits[7];
          tok_index = Integer.parseInt(splits[0]) - 1;
          int head = Integer.parseInt(splits[6]) - 1; //The root is -1 now
          if (!tokSeq[tok_index].equals(word)) {
            //System.err.println(tokSeq.toString());
            System.err.println(word + " : "+ tokSeq[tok_index]);
            System.exit(1);
          }

          //CoreLabel token = tf.makeToken(word, 0, 0);
          //token.setTag(pos);
          //token.set(CoreAnnotations.CoNLLDepParentIndexAnnotation.class, head);
          //token.set(CoreAnnotations.CoNLLDepTypeAnnotation.class, depType);
          //sentenceTokens.add(token);

          tree.add(head, depType);
        }
      }    
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  public static void writeConllFile(String outFile, List<CoreMap> sentences, List<DependencyTree> trees)
  {
    try
    {
      PrintWriter output = IOUtils.getPrintWriter(outFile);

      for (int i = 0; i < sentences.size(); i++)
      {
        CoreMap sentence = sentences.get(i);
        DependencyTree tree = trees.get(i);

        List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

        for (int j = 1, size = tokens.size(); j <= size; ++j)
        {
          CoreLabel token = tokens.get(j - 1);
          output.printf("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_%n",
              j, token.word(), token.tag(), token.tag(),
              tree.getHead(j), tree.getLabel(j));
        }
        output.println();
      }
      output.close();
    }
    catch (Exception e) {
      throw new RuntimeIOException(e);
    }
  }

  public static void printTreeStats(String str, List<DependencyTree> trees)
  {
    System.err.println(Config.SEPARATOR + " " + str);
    int nTrees = trees.size();
    int nonTree = 0;
    int multiRoot = 0;
    int nonProjective = 0;
    for (DependencyTree tree : trees) {
      if (!tree.isTree())
        ++nonTree;
      else
      {
        if (!tree.isProjective())
          ++nonProjective;
        if (!tree.isSingleRoot())
          ++multiRoot;
      }
    }
    System.err.printf("#Trees: %d%n", nTrees);
    System.err.printf("%d tree(s) are illegal (%.2f%%).%n", nonTree, nonTree * 100.0 / nTrees);
    System.err.printf("%d tree(s) are legal but have multiple roots (%.2f%%).%n", multiRoot, multiRoot * 100.0 / nTrees);
    System.err.printf("%d tree(s) are legal but not projective (%.2f%%).%n", nonProjective, nonProjective * 100.0 / nTrees);
  }

  public static void printTreeStats(List<DependencyTree> trees)
  {
    printTreeStats("", trees);
  }
}
