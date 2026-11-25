package heuristic;

import java.util.*;

/** Shared windowed adaptive selection for destroy and repair operator pools. */
final class AdaptiveOperatorSelection {
  private static final double NUMERIC_EPS = 1e-12;

  /** Parameters shared by the assignment and routing adaptive searches. */
  record Parameters(
      int windowLength,
      double adaptiveProbability,
      double probabilityFluctuationCoefficient,
      double historyWeight,
      double scoreDecay,
      double baseReward,
      double improvementReward,
      double rejectionPenalty,
      double minimumWeight,
      double maximumWeight) {
    Parameters {
      if (windowLength <= 0) throw new IllegalArgumentException("windowLength must be positive");
      requireUnitInterval(adaptiveProbability, "adaptiveProbability");
      requireUnitInterval(probabilityFluctuationCoefficient, "probabilityFluctuationCoefficient");
      requireUnitInterval(historyWeight, "historyWeight");
      requireUnitInterval(scoreDecay, "scoreDecay");
      if (baseReward < 0 || improvementReward < 0 || rejectionPenalty < 0)
        throw new IllegalArgumentException("adaptive rewards and penalties must be non-negative");
      if (!(minimumWeight > 0) || maximumWeight < minimumWeight)
        throw new IllegalArgumentException("invalid adaptive weight bounds");
    }

    private static void requireUnitInterval(double value, String name) {
      if (value < 0 || value > 1) throw new IllegalArgumentException(name + " must be in [0, 1]");
    }
  }

  /** One independently selected destroy/repair pair. */
  record Selection(
      String destroy,
      String repair,
      int destroyIndex,
      int repairIndex,
      boolean adaptiveWindow) {}

  /** Per-search adaptive state. Weights are intentionally reset for each ALNS invocation. */
  static final class State {
    private final Parameters parameters;
    private final OperatorPool destroy;
    private final OperatorPool repair;
    private int currentWindow = -1;
    private boolean adaptiveWindow;
    private double adaptiveCredit;

    State(List<String> destroyOperators, List<String> repairOperators, Parameters parameters) {
      this.parameters = Objects.requireNonNull(parameters);
      destroy = new OperatorPool(destroyOperators, parameters);
      repair = new OperatorPool(repairOperators, parameters);
      // This phase makes the first post-warm-up window adaptive whenever p_A > 0.
      adaptiveCredit = 1.0 - parameters.adaptiveProbability();
    }

    Selection select(PythonRandom rng, int iteration) {
      updateWindow(iteration / parameters.windowLength());
      int destroyIndex = destroy.select(rng, adaptiveWindow);
      int repairIndex = repair.select(rng, adaptiveWindow);
      return new Selection(
          destroy.name(destroyIndex),
          repair.name(repairIndex),
          destroyIndex,
          repairIndex,
          adaptiveWindow);
    }

    /** Records the two-state reward and updates both pools at completed window boundaries. */
    void record(Selection selection, boolean improved, double improvement, int iteration) {
      double reward =
          improved
              ? parameters.baseReward()
                  + parameters.improvementReward() * improvement / (1.0 + improvement)
              : -parameters.rejectionPenalty();
      destroy.record(selection.destroyIndex(), reward);
      repair.record(selection.repairIndex(), reward);
      if ((iteration + 1) % parameters.windowLength() == 0) {
        destroy.updateWeights();
        repair.updateWeights();
      }
    }

    private void updateWindow(int targetWindow) {
      while (currentWindow < targetWindow) {
        currentWindow++;
        if (currentWindow == 0) {
          // The first window is a uniform-random warm-up window.
          adaptiveWindow = false;
          continue;
        }
        adaptiveCredit += parameters.adaptiveProbability();
        if (parameters.adaptiveProbability() > 0 && adaptiveCredit + NUMERIC_EPS >= 1.0) {
          adaptiveWindow = true;
          adaptiveCredit -= 1.0;
        } else {
          adaptiveWindow = false;
        }
      }
    }
  }

  private static final class OperatorPool {
    private final List<String> names;
    private final Parameters parameters;
    private final double[] weights;
    private final double[] scores;
    private final int[] counts;

    OperatorPool(List<String> operators, Parameters parameters) {
      if (operators == null || operators.isEmpty())
        throw new IllegalArgumentException("adaptive operator pool must be nonempty");
      names = List.copyOf(operators);
      this.parameters = parameters;
      weights = new double[names.size()];
      scores = new double[names.size()];
      counts = new int[names.size()];
      Arrays.fill(weights, 1.0);
    }

    String name(int index) {
      return names.get(index);
    }

    int select(PythonRandom rng, boolean adaptiveWindow) {
      if (!adaptiveWindow) return (int) (rng.random() * names.size());
      double[] probabilities = boundedProbabilities();
      double draw = rng.random();
      double cumulative = 0;
      for (int i = 0; i < probabilities.length; i++) {
        cumulative += probabilities[i];
        if (draw <= cumulative) return i;
      }
      return probabilities.length - 1;
    }

    void record(int index, double reward) {
      scores[index] = parameters.scoreDecay() * scores[index] + reward;
      counts[index]++;
    }

    void updateWeights() {
      for (int i = 0; i < weights.length; i++) {
        if (counts[i] > 0) {
          double averageScore = scores[i] / counts[i];
          double updated =
              parameters.historyWeight() * weights[i]
                  + (1.0 - parameters.historyWeight()) * averageScore;
          weights[i] = clamp(updated, parameters.minimumWeight(), parameters.maximumWeight());
        }
      }
      Arrays.fill(scores, 0);
      Arrays.fill(counts, 0);
    }

    /**
     * Directly clips probabilities to uniform +/- rho, then redistributes the residual without
     * allowing any probability to cross the same bounds.
     */
    private double[] boundedProbabilities() {
      int size = weights.length;
      double uniform = 1.0 / size;
      double lower = uniform * (1.0 - parameters.probabilityFluctuationCoefficient());
      double upper = uniform * (1.0 + parameters.probabilityFluctuationCoefficient());
      double weightSum = Arrays.stream(weights).sum();
      double[] probabilities = new double[size];
      for (int i = 0; i < size; i++)
        probabilities[i] = clamp(weights[i] / weightSum, lower, upper);

      // Independent clipping may change the sum for pools with more than two operators.
      // Redistribute only into entries that still have room, preserving every hard bound.
      for (int pass = 0; pass < size + 2; pass++) {
        double sum = Arrays.stream(probabilities).sum();
        double residual = 1.0 - sum;
        if (Math.abs(residual) <= NUMERIC_EPS) break;
        int eligible = 0;
        for (double probability : probabilities) {
          if ((residual > 0 && probability < upper - NUMERIC_EPS)
              || (residual < 0 && probability > lower + NUMERIC_EPS)) eligible++;
        }
        if (eligible == 0) break;
        double share = residual / eligible;
        for (int i = 0; i < probabilities.length; i++) {
          if ((residual > 0 && probabilities[i] < upper - NUMERIC_EPS)
              || (residual < 0 && probabilities[i] > lower + NUMERIC_EPS)) {
            probabilities[i] = clamp(probabilities[i] + share, lower, upper);
          }
        }
      }
      return probabilities;
    }
  }

  private static double clamp(double value, double lower, double upper) {
    return Math.max(lower, Math.min(upper, value));
  }

  private AdaptiveOperatorSelection() {}
}
