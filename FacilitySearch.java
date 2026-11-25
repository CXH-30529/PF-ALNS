package heuristic;

import static heuristic.Model.*;

import java.util.*;

/** Facility-selection operators that generate open, close, swap, and shaking candidates. */
public final class FacilitySearch {
  final Instance instance;
  final Solver.Config config;
  final FacilityMemory memory;
  /** Mean facility cost used to normalize individual facility costs. */
  final double meanFacilityCost;
  final boolean[] openCovered, openReached;
  final double[] scoreBuffer;

  FacilitySearch(Instance instance, Solver.Config config, FacilityMemory memory) {
    this.instance = instance;
    this.config = config;
    this.memory = memory;
    meanFacilityCost = Arrays.stream(instance.facilityCost).average().orElseThrow();
    openCovered = new boolean[instance.customerCount()];
    openReached = new boolean[instance.customerCount()];
    scoreBuffer = new double[instance.facilityCount()];
  }

  /** Enumerates all legal nonempty facility selections used by shaking. */
  List<List<Integer>> allAllowedSelections() {
    List<Integer> all = new ArrayList<>();
    for (int f = 0; f < instance.facilityCount(); f++) all.add(f);
    List<List<Integer>> selections = new ArrayList<>();
    for (int k = 1; k <= Math.min(instance.facilityCount(), config.maximumOpenFacilities); k++)
      addCombinations(all, k, selections);
    return selections;
  }

  /** Returns the standard selection operators allowed by the current number of open facilities. */
  List<String> availableOperators(List<Integer> current) {
    List<String> available = new ArrayList<>();
    if (current.size() < Math.min(config.maximumOpenFacilities, instance.facilityCount()))
      available.add("open");
    if (current.size() > 1) available.add("close");
    if (!current.isEmpty() && current.size() < instance.facilityCount()) available.add("swap");
    return available;
  }

  /**
   * Returns an unevaluated candidate: open tries descending scores, close tries ascending
   * contributions, and swap uses random order. Returns {@code null} when the neighborhood is
   * exhausted.
   */
  List<Integer> applyOperator(List<Integer> current, String operator, PythonRandom rng) {
    Set<Integer> selected = new HashSet<>(current);
    if (operator.equals("open")) {
      List<Integer> closed = new ArrayList<>();
      for (int f = 0; f < instance.facilityCount(); f++) if (!selected.contains(f)) closed.add(f);
      double[] scores = InitialSolution.facilityOpenScores(this, selected, closed);
      closed.sort(Comparator.<Integer>comparingDouble(f -> scores[f]).reversed()
          .thenComparingInt(f -> f));
      for (int opened : closed) {
        Set<Integer> proposal = new HashSet<>(selected);
        proposal.add(opened);
        List<Integer> canonical = canonicalFacilities(proposal);
        if (memory.get(canonical) == null) return canonical;
      }
    } else if (operator.equals("close")) {
      List<Integer> opened = new ArrayList<>(current);
      opened.sort(Comparator.<Integer>comparingDouble(f -> closeContributionScore(selected, f))
          .thenComparingInt(f -> f));
      for (int closed : opened) {
        Set<Integer> proposal = new HashSet<>(selected);
        proposal.remove(closed);
        List<Integer> canonical = canonicalFacilities(proposal);
        if (memory.get(canonical) == null) return canonical;
      }
    } else if (operator.equals("swap")) {
      List<List<Integer>> pairs = new ArrayList<>();
      for (int closed : current)
        for (int opened = 0; opened < instance.facilityCount(); opened++)
          if (!selected.contains(opened)) pairs.add(List.of(closed, opened));
      rng.shuffle(pairs);
      for (List<Integer> pair : pairs) {
        Set<Integer> proposal = new HashSet<>(selected);
        proposal.remove(pair.get(0));
        proposal.add(pair.get(1));
        List<Integer> canonical = canonicalFacilities(proposal);
        if (memory.get(canonical) == null) return canonical;
      }
    }
    return null;
  }

  /**
   * Scores a facility by its unique coverage, unique reachability, and cost; close removes
   * lower-scoring facilities first.
   */
  private double closeContributionScore(Set<Integer> selected, int facility) {
    Set<Integer> others = new HashSet<>(selected);
    others.remove(facility);

    Set<Integer> covered = new HashSet<>();
    for (int other : others) covered.addAll(instance.facilityCover.get(other));

    Set<Integer> unique = new HashSet<>(instance.facilityCover.get(facility));
    unique.removeAll(covered);

    int uniqueReach = 0;
    for (int customer = 0; customer < instance.customerCount(); customer++)
      if (instance.routeReachable[facility][customer]) {
        boolean reachedElsewhere = false;
        for (int other : others)
          if (instance.routeReachable[other][customer]) {
            reachedElsewhere = true;
            break;
        }
        if (!reachedElsewhere) uniqueReach++;
      }
    Scores.FuzzyScoreConfig score = config.score;
    double directContribution = (double) unique.size() / instance.customerCount(),
        reachContribution = (double) uniqueReach / instance.customerCount(),
        boundedCost =
            instance.facilityCost[facility]
                / Math.max(EPS, instance.facilityCost[facility] + meanFacilityCost);
    return score.facilityCoverGain * directContribution
        + score.facilityReachGain * reachContribution
        - score.facilityCostPenalty * boundedCost;
  }

  private static void addCombinations(List<Integer> source, int size, List<List<Integer>> output) {
    combinations(source, size, 0, new ArrayList<>(), output);
  }

  /** Generates fixed-size facility combinations by backtracking. */
  private static void combinations(
      List<Integer> source, int size, int start, List<Integer> current,
      List<List<Integer>> output) {
    if (current.size() == size) {
      output.add(List.copyOf(current));
      return;
    }
    for (int i = start; i <= source.size() - (size - current.size()); i++) {
      current.add(source.get(i));
      combinations(source, size, i + 1, current, output);
      current.remove(current.size() - 1);
    }
  }
}

/** Heuristic scoring coefficients for opening and closing facilities. */
final class Scores {
  private Scores() {}

  static final class FuzzyScoreConfig {
    /** Weight of the direct-coverage gain. */
    public double facilityCoverGain = .5;

    /** Weight of the route-reachability gain. */
    public double facilityReachGain = .5;

    /** Weight of the relative fixed-cost penalty. */
    public double facilityCostPenalty = .5;
  }
}
