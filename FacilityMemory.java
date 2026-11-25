package heuristic;

import heuristic.AssignmentSearch.Result;

import java.util.*;

/** Caches facility-selection evaluations to avoid repeated inner solves. */
final class FacilityMemory {
  /** A facility selection and its evaluation result. */
  static final class Entry {
    final List<Integer> selection;
    boolean feasible;
    double objective;
    Result result;
    final int firstIteration;
    int lastIteration;
    int evaluations = 1;

    Entry(List<Integer> selection, boolean feasible, double objective, Result result,
        int iteration) {
      this.selection = selection;
      this.feasible = feasible;
      this.objective = objective;
      this.result = result;
      firstIteration = lastIteration = iteration;
    }
  }

  final Map<List<Integer>, Entry> entries = new LinkedHashMap<>();

  /** Returns the cached evaluation for a selection, or {@code null} if none exists. */
  Entry get(List<Integer> selection) { return entries.get(selection); }

  /** Stores the first evaluation; infeasible selections receive an infinite objective value. */
  Entry remember(List<Integer> selection, Result result, double facilityCost, int iteration) {
    double objective = result.feasible ? facilityCost + result.serviceCost
        : Double.POSITIVE_INFINITY;
    Entry entry =
        new Entry(List.copyOf(selection), result.feasible, objective, result, iteration);
    entries.put(entry.selection, entry);
    return entry;
  }

  /** Updates the most recent iteration in which a cached selection was accessed. */
  void touch(List<Integer> selection, int iteration) {
    Entry entry = get(selection);
    if (entry != null) entry.lastIteration = iteration;
  }

  /** Returns the {@code count} feasible selections with the lowest objective values. */
  List<Entry> bestFeasible(int count) {
    return entries.values().stream().filter(entry -> entry.feasible)
        .sorted(Comparator.comparingDouble(entry -> entry.objective)).limit(count).toList();
  }

  /** Returns the number of distinct facility selections evaluated. */
  int uniqueEvaluations() { return entries.size(); }
}
