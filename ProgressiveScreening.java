package heuristic;

import java.util.Arrays;

/** Candidate promotion and evaluation budgets; the final winner reuses a stored result. */
public final class ProgressiveScreening {
  private ProgressiveScreening() {}

  public static final class Config {
    /** L result levels contain L-1 evaluations and a final, single-winner promotion. */
    public int resultLevelCount = 3;
    public double filterRatio = .10;
    /** Optional ratios in chronological order; one value applies to every ordinary promotion. */
    public double[] filterRatios = {};
    /** Optional bounds for each ordinary promotion, in chronological order. Missing entries and
     * -1 values are unbounded. The default instance bounds only the first promotion to 4--6. */
    public int[] promotionMins = {4}, promotionMaxs = {6};
    /** Explicit B[i] / B[i+1] coefficients. One value repeats at every transition. */
    public double[] assignmentDestroyRepairRatios = {.03},
        assignmentLocalSearchRatios = {.1},
        routeLocalSearchRatios = {.1},
        routeDestroyRepairRatios = {.05},
        routeSweepRatios = {.3};

    /** Convenience configuration: apply a shared coefficient sequence to every module. */
    public void setBudgetRatios(double... ratios) {
      assignmentDestroyRepairRatios = ratios.clone();
      assignmentLocalSearchRatios = ratios.clone();
      routeLocalSearchRatios = ratios.clone();
      routeDestroyRepairRatios = ratios.clone();
      routeSweepRatios = ratios.clone();
    }

    void readProperties() {
      resultLevelCount = Integer.getInteger("screening.resultLevelCount", resultLevelCount);
      promotionMins = integersProperty("screening.promotionMins", promotionMins);
      promotionMaxs = integersProperty("screening.promotionMaxs", promotionMaxs);
      String ratio = System.getProperty("screening.filterRatio");
      if (ratio != null) filterRatio = Double.parseDouble(ratio);
      filterRatios = ratiosProperty("screening.filterRatios", filterRatios);
      double[] common = ratiosProperty("screening.budgetRatios", new double[0]);
      if (common.length > 0) setBudgetRatios(common);
      assignmentDestroyRepairRatios =
          ratiosProperty("screening.assignmentDestroyRepairRatios", assignmentDestroyRepairRatios);
      assignmentLocalSearchRatios =
          ratiosProperty("screening.assignmentLocalSearchRatios", assignmentLocalSearchRatios);
      routeLocalSearchRatios = ratiosProperty("screening.routeLocalSearchRatios", routeLocalSearchRatios);
      routeDestroyRepairRatios =
          ratiosProperty("screening.routeDestroyRepairRatios", routeDestroyRepairRatios);
      routeSweepRatios = ratiosProperty("screening.routeSweepRatios", routeSweepRatios);
    }

    private static double[] ratiosProperty(String name, double[] fallback) {
      String raw = System.getProperty(name);
      if (raw == null || raw.isBlank()) return fallback;
      return Arrays.stream(raw.split(",", -1)).map(String::trim)
          .mapToDouble(Double::parseDouble).toArray();
    }

    private static int[] integersProperty(String name, int[] fallback) {
      String raw = System.getProperty(name);
      if (raw == null || raw.isBlank()) return fallback;
      return Arrays.stream(raw.split(",", -1)).map(String::trim)
          .mapToInt(Integer::parseInt).toArray();
    }

    void validate() {
      if (resultLevelCount < 3)
        throw new IllegalArgumentException("resultLevelCount must be at least 3");
      checkRatio(filterRatio, "filterRatio");
      checkRatios(filterRatios, "filterRatios", true);
      checkRatios(assignmentDestroyRepairRatios, "assignmentDestroyRepairRatios", false);
      checkRatios(assignmentLocalSearchRatios, "assignmentLocalSearchRatios", false);
      checkRatios(routeLocalSearchRatios, "routeLocalSearchRatios", false);
      checkRatios(routeDestroyRepairRatios, "routeDestroyRepairRatios", false);
      checkRatios(routeSweepRatios, "routeSweepRatios", false);
      checkBounds(promotionMins, "promotionMins");
      checkBounds(promotionMaxs, "promotionMaxs");
      for (int fromLevel = 1; fromLevel <= resultLevelCount - 2; fromLevel++) {
        if (minimum(fromLevel) > maximum(fromLevel))
          throw new IllegalArgumentException("invalid promotion bounds at round " + fromLevel);
      }
    }

    private int minimum(int fromLevel) {
      return bound(promotionMins, fromLevel, 0);
    }

    private int maximum(int fromLevel) {
      return bound(promotionMaxs, fromLevel, Integer.MAX_VALUE);
    }

    private static int bound(int[] bounds, int fromLevel, int unbounded) {
      int index = fromLevel - 1;
      if (index >= bounds.length) return unbounded;
      int value = bounds[index];
      return value < 0 ? unbounded : value;
    }

    /** fromLevel is one-based. The last promotion always has exactly one winner. */
    public int promotionCount(int available, int fromLevel) {
      validate();
      if (available < 0 || fromLevel < 1 || fromLevel >= resultLevelCount)
        throw new IllegalArgumentException("invalid promotion level or candidate count");
      if (available == 0) return 0;
      if (fromLevel == resultLevelCount - 1) return 1;
      double ratio = filterRatios.length == 0 ? filterRatio
          : filterRatios[filterRatios.length == 1 ? 0 : fromLevel - 1];
      int count = (int) Math.ceil(available * ratio);
      count = Math.max(minimum(fromLevel), Math.min(maximum(fromLevel), count));
      return Math.min(available, count);
    }

    private void checkBounds(int[] bounds, String name) {
      if (bounds == null || bounds.length > resultLevelCount - 2)
        throw new IllegalArgumentException(name + " needs at most L-2 values");
      for (int bound : bounds)
        if (bound < -1) throw new IllegalArgumentException(name + " values must be at least -1");
    }

    private void checkRatios(double[] ratios, String name, boolean optionalFilter) {
      if (ratios == null || ((!optionalFilter || ratios.length != 0) && ratios.length != 1
          && ratios.length != resultLevelCount - 2))
        throw new IllegalArgumentException(name + " needs "
            + (optionalFilter ? "0, " : "") + "1 or L-2 values");
      for (double ratio : ratios) {
        if (optionalFilter) checkRatio(ratio, name);
        else if (!Double.isFinite(ratio) || ratio < 0 || ratio > 1)
          throw new IllegalArgumentException(name + " must be in [0, 1]");
      }
    }

    private static void checkRatio(double ratio, String name) {
      if (!Double.isFinite(ratio) || ratio <= 0 || ratio > 1)
        throw new IllegalArgumentException(name + " must be in (0, 1]");
    }

    private int budget(int deep, int evaluationIndex, double[] ratios) {
      int transitions = resultLevelCount - 2;
      if (deep < 0 || evaluationIndex < 0 || evaluationIndex > transitions)
        throw new IllegalArgumentException("invalid evaluation budget or index");
      if (evaluationIndex == transitions) return deep;
      double value = deep;
      for (int index = evaluationIndex; index < transitions; index++)
        value *= ratios[ratios.length == 1 ? 0 : index];
      return value == 0 ? 0 : Math.max(1, roundBudget(value));
    }

    private static int roundBudget(double value) {
      // Avoid turning, for example, 14.000000000000002 into 15.
      double nearest = Math.rint(value);
      return (int) Math.ceil(Math.abs(value - nearest) <= 1e-9 ? nearest : value);
    }

  }

  public record Budget(int assignmentIterations, int assignmentLocalPasses,
      int routeLocalPasses, int routeRuinIterations, int routeSweepRotations) {}

  /** evaluationIndex is zero-based; the last evaluated level is L-2. */
  public static Budget budgetFor(Solver.Config solver, int evaluationIndex) {
    Config c = solver.screening;
    c.validate();
    return new Budget(
        c.budget(solver.deepAssignmentIterations,
            evaluationIndex, c.assignmentDestroyRepairRatios),
        c.budget(solver.deepAssignmentLocalSearchPasses,
            evaluationIndex, c.assignmentLocalSearchRatios),
        c.budget(solver.route.maxVnsPasses,
            evaluationIndex, c.routeLocalSearchRatios),
        c.budget(solver.route.ruinIterations,
            evaluationIndex, c.routeDestroyRepairRatios),
        c.budget(solver.route.sweepRotations,
            evaluationIndex, c.routeSweepRatios));
  }

}
