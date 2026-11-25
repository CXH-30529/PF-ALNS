package heuristic;

import static heuristic.Model.*;
import static heuristic.RouteSearch.*;

import java.util.*;

/**
 * Searches direct warehouse coverage, representative coverage, and vehicle routes for fixed facilities.
 */
public final class AssignmentSearch {
  /** Search budgets and operator parameters for the assignment layer. */
  public static final class Config {
    public int alnsIterations = 24,
        assignmentBacktrackNodes = 30000,
        unifiedLocalSearchPasses = 5, // Single-operator iterations; legacy field name.
        unifiedLocalCandidateLimit = 12,
        unifiedRandomSamples = 3,
        unifiedDeepCandidates = 6,
        representativeSplitAttempts = 10;
    public double destroyMin = .10,
        destroyMax = .28,
        cooling = .985;

    // Windowed adaptive destroy/repair selection.
    public int adaptiveWindowLength = 10;
    public double adaptiveProbability = .50,
        probabilityFluctuationCoefficient = .10,
        adaptiveHistoryWeight = .50,
        adaptiveScoreDecay = .90,
        adaptiveBaseReward = 1.0,
        adaptiveImprovementReward = 2.0,
        adaptiveRejectionPenalty = .20,
        adaptiveMinimumWeight = .10,
        adaptiveMaximumWeight = 5.0;

    /** Low-coverage weight used by the worst-removal operator. */
    public double worstLowCoverage = .20;

    /** Load-pressure weight used by the worst-removal operator. */
    public double worstLoadPressure = .80;

    double worstRemovalScore(double lowCoverage, double loadPressure) {
      return worstLowCoverage * lowCoverage + worstLoadPressure * loadPressure;
    }
    public boolean enableClusterOperators = true;

    public Config copy() {
      Config c = new Config();
      try {
        for (var f : Config.class.getFields()) f.set(c, f.get(this));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return c;
    }
  }

  /** Assignment-search result for a fixed facility selection. */
  public static final class Result {
    public CoveragePlan coverage;
    public List<RoutePlan> routes;
    public double routeCost, warehouseCoverageCost, customerCoverageCost, serviceCost;
    public boolean feasible;
    public String reason;
    public Map<String, Integer> operatorStatistics;

    public Result(CoveragePlan c, List<RoutePlan> r, double x, boolean f) {
      this(c, r, x, f, "", new HashMap<>());
    }

    public Result(
        CoveragePlan c, List<RoutePlan> r, double x, boolean f, String s, Map<String, Integer> o) {
      coverage = c;
      routes = r;
      routeCost = x;
      serviceCost = x;
      feasible = f;
      reason = s;
      operatorStatistics = o;
    }
  }

  static final List<String>
      DESTROY =
          List.of(
              "worst", "related", "random", "coverage_count"),
      REPAIR =
          List.of(
              "randomized",
              "alternating_coverage",
              "farthest_distance");
  // ==================== Fixed inputs, precomputed sets, and runtime state ====================
  final Instance in;
  final List<Integer> fac;
  final Config cfg;
  final RouteSearch.Config routeCfg, quick;
  final PythonRandom rng;
  final Set<Integer> all, warehouseEligible, reachable;
  final int[] nearestWarehouse;
  final double[] nearestOpenFacilityDistance;
  final Map<Integer, List<Integer>> candidates = new HashMap<>();
  final Map<Integer, Set<Integer>> targets = new HashMap<>();
  final Map<String, Integer> stats = new LinkedHashMap<>();
  int nodes;
  private int relatedDestroyCalls;

  // ==================== Cache keys ====================
  private static final class SelectionKey {
    final int[] values;
    final int hash;

    SelectionKey(Set<Integer> s) {
      values = s.stream().mapToInt(x -> x).sorted().toArray();
      hash = Arrays.hashCode(values);
    }

    public int hashCode() {
      return hash;
    }

    public boolean equals(Object o) {
      return o instanceof SelectionKey k && Arrays.equals(values, k.values);
    }
  }

  private static final class CoverageKey {
    final int[] selected, assignment;
    final int hash;

    CoverageKey(CoveragePlan p, int n) {
      selected = p.selected.stream().mapToInt(x -> x).sorted().toArray();
      assignment = new int[n];
      Arrays.fill(assignment, Integer.MIN_VALUE);
      for (var e : p.assignment.entrySet()) assignment[e.getKey()] = e.getValue();
      for (var e : p.directAssignment.entrySet()) assignment[e.getKey()] = -1 - e.getValue();
      hash = 31 * Arrays.hashCode(selected) + Arrays.hashCode(assignment);
    }

    public int hashCode() {
      return hash;
    }

    public boolean equals(Object o) {
      return o instanceof CoverageKey k
          && Arrays.equals(selected, k.selected)
          && Arrays.equals(assignment, k.assignment);
    }
  }

  /** Deterministic results shared by sequential restarts with the same facilities and budgets. */
  static final class SharedCaches {
    final Map<SelectionKey, Optional<CoveragePlan>> assignments = lruCache(4096);
    final Map<CoverageKey, Optional<List<RoutePlan>>> quick = lruCache(4096);
    final Map<CoverageKey, Optional<List<RoutePlan>>> full = lruCache(2048);
  }

  // Optional.empty represents a known infeasible state.
  final Map<SelectionKey, Optional<CoveragePlan>> assignmentCache;
  final Map<CoverageKey, Optional<List<RoutePlan>>> evaluationCache;
  final Map<CoverageKey, Optional<List<RoutePlan>>> fullEvaluationCache;

  private static <K, V> Map<K, V> lruCache(int maximumSize) {
    return new LinkedHashMap<K, V>(256, .75f, true) {
      @Override
      protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > maximumSize;
      }
    };
  }

  // ==================== Preprocessing and main flow ====================

  /** Initializes reachable customers and coverage candidates for fixed facilities. */
  public AssignmentSearch(
      Instance i,
      List<Integer> f,
      Config c,
      RouteSearch.Config r,
      long seed) {
    this(i, f, c, r, seed, new SharedCaches());
  }

  AssignmentSearch(
      Instance i,
      List<Integer> f,
      Config c,
      RouteSearch.Config r,
      long seed,
      SharedCaches sharedCaches) {
    in = i;
    fac = f;
    cfg = c;
    routeCfg = r;
    assignmentCache = sharedCaches.assignments;
    evaluationCache = sharedCaches.quick;
    fullEvaluationCache = sharedCaches.full;
    quick = r.copy();
    quick.ruinIterations = 0;
    if (i.customerCount() >= 200) {
      quick.maxVnsPasses = 0;
      quick.enableSwap = false;
      quick.sweepRotations = 2;
    }
    rng = new PythonRandom(seed);
    all = range(i.customerCount());
    warehouseEligible = new HashSet<>();
    for (int x : f) warehouseEligible.addAll(i.facilityCover.get(x));
    nearestWarehouse = new int[i.customerCount()];
    nearestOpenFacilityDistance = new double[i.customerCount()];
    Arrays.fill(nearestWarehouse, -1);
    Arrays.fill(nearestOpenFacilityDistance, Double.POSITIVE_INFINITY);
    for (int t : all)
      for (int k : fac) {
        nearestOpenFacilityDistance[t] =
            Math.min(nearestOpenFacilityDistance[t], i.facilityDistance[k][t]);
        if (i.facilityCoverMembership[k][t] && (nearestWarehouse[t] < 0 ||
            i.facilityDistance[k][t] < i.facilityDistance[nearestWarehouse[t]][t]))
          nearestWarehouse[t] = k;
      }
    reachable = new HashSet<>();
    for (int x : all)
      for (int d : f)
        if (i.routeReachable[d][x]) {
          reachable.add(x);
          break;
        }
    for (int t : all) {
      List<Integer> x = new ArrayList<>();
      for (int j : reachable) if (i.customerCoverMembership[t][j]) x.add(j);
      Collections.sort(x);
      candidates.put(t, x);
    }
    for (int j : reachable) {
      Set<Integer> x = new HashSet<>(all);
      x.retainAll(i.customerCover.get(j));
      targets.put(j, x);
    }
    for (String k :
        List.of(
            "coverage_count_destroy_calls",
            "coverage_count_destroy_best_improvements")) stats.put(k, 0);
  }

  /** Runs initial construction, ALNS, local search, and final route evaluation. */
  public Result solve() {
    if (fac.isEmpty()) return infeasible("no open facility");
    for (var e : candidates.entrySet())
      if (e.getValue().isEmpty() && nearestWarehouse[e.getKey()] < 0) return infeasible("customers without a reachable coverer");
    Result best = InitialSolution.jointAssignment(this);
    if (best == null) return infeasible("coverage assignment or route construction failed");
    if (cfg.alnsIterations > 0) best = assignmentDestroyRepairSearch(best);
    if (cfg.unifiedLocalSearchPasses > 0) best = unifiedAssignmentLocalSearch(best);
    List<RoutePlan> deep = cachedRoutes(best.coverage, routeCfg, fullEvaluationCache);
    if (deep != null && serviceCost(best.coverage, deep) + EPS < best.serviceCost)
      best = result(best.coverage, deep, true);
    best.operatorStatistics = new LinkedHashMap<>(stats);
    return best;
  }

  private Result infeasible(String s) {
    return result(
        new CoveragePlan(new HashSet<>(), new HashMap<>(), new HashMap<>()),
        new ArrayList<>(),
        false,
        s,
        new HashMap<>());
  }

  // ==================== 1. Assignment destroy operators: complete -> partial ====================

  /** Applies the random, related, worst, or coverage-count destroy operator. */
  private Set<Integer> applyAssignmentDestroyOperator(
      Result cur, String kind, double fraction) {
    if (!DESTROY.contains(kind))
      throw new IllegalArgumentException("Unknown assignment destroy operator: " + kind);
    Set<Integer> sel = cur.coverage.selected;
    if (kind.equals("coverage_count")) {
      inc("coverage_count_destroy_calls");
      return coverageCountDestroy(cur);
    }
    if (kind.equals("related")) {
      relatedDestroyCalls++;
      if (relatedDestroyCalls % 2 == 0) fraction = rng.uniform(.4, .7);
    }
    int count = Math.min(sel.size(), Math.max(1, pythonRound(sel.size() * fraction)));
    if (count >= sel.size()) count = Math.max(0, sel.size() - 1);
    if (count == 0) return new HashSet<>(sel);
    Set<Integer> removed = new HashSet<>();
    List<Integer> sorted = new ArrayList<>(new TreeSet<>(sel));
    if (kind.equals("random")) removed.addAll(rng.sample(sorted, count));
    else if (kind.equals("related")) {
      int anchor = rng.choice(sorted);
      sorted.sort(Comparator.comparingDouble(j -> in.customerDistance[anchor][j]));
      removed.addAll(sorted.subList(0, count));
    } else {
      int[] coveredCount = new int[in.customerCount()];
      for (int assigned : cur.coverage.assignment.values()) coveredCount[assigned]++;
      List<double[]> rank = new ArrayList<>();
      for (int c : sel) {
        double sc =
            cfg.worstRemovalScore(
                1.0 / (1 + coveredCount[c]),
                cur.coverage.groupLoad.get(c) / in.capacity);
        rank.add(new double[] {sc + rng.random() * 1e-5, c});
      }
      rank.sort((a, b) -> Double.compare(b[0], a[0]));
      for (int i = 0; i < count; i++) removed.add((int) rank.get(i)[1]);
    }
    Set<Integer> b = new HashSet<>(sel);
    b.removeAll(removed);
    return b;
  }

  private Set<Integer> coverageCountDestroy(Result cur) {
    Set<Integer> sel = cur.coverage.selected;
    if (sel.size() <= 1) return new HashSet<>(sel);
    int count = Math.min(sel.size() - 1, Math.max(1, (int) Math.ceil(in.customerCount() * .10)));
    List<double[]> rank = new ArrayList<>();
    for (int c : sel) {
      int coveredOthers = in.customerCover.get(c).size() - 1;
      rank.add(new double[] {coveredOthers + rng.random() * 1e-5, c});
    }
    rank.sort((a, b) -> Double.compare(b[0], a[0]));
    Set<Integer> b = new HashSet<>(sel);
    for (int i = 0; i < count; i++) b.remove((int) rank.get(i)[1]);
    return b;
  }

  // ==================== 2. Assignment repair operators: partial -> complete ====================

  private int uncoveredTargetCount(int representative, Set<Integer> uncovered) {
    int count = 0;
    for (int t : targets.get(representative)) if (uncovered.contains(t)) count++;
    return count;
  }

  private double nearestFacilityDistance(int customer) {
    return nearestOpenFacilityDistance[customer];
  }

  /** Completes the representative set, then builds a capacity-feasible assignment. */
  private CoveragePlan construct(Set<Integer> base, String strategy) {
    Set<Integer> sel = new HashSet<>(base);
    sel.retainAll(reachable);
    Set<Integer> covered = new HashSet<>();
    for (int j : sel) covered.addAll(targets.get(j));
    Set<Integer> unc = new HashSet<>(all);
    unc.removeAll(covered);
    unc.removeAll(warehouseEligible); // Feasibility only; assign() chooses the actual service mode.
    boolean alternatingFirstRound = strategy.equals("alternating_coverage"),
        alternatingChooseMaximum = true;
    while (!unc.isEmpty()) {
      List<Integer> pool = new ArrayList<>();
      for (int j : reachable) if (!sel.contains(j)) pool.add(j);
      if (strategy.equals("randomized")) {
        pool.removeIf(j -> Collections.disjoint(targets.get(j), unc));
        if (pool.isEmpty()) return null;
        Collections.sort(pool);
        int j = rng.choice(pool);
        sel.add(j);
        unc.removeAll(targets.get(j));
        continue;
      }
      if (strategy.equals("alternating_coverage")) {
        Map<Integer, Integer> gains = new HashMap<>();
        for (int j : pool) gains.put(j, uncoveredTargetCount(j, unc));
        pool.removeIf(j -> gains.get(j) == 0);
        if (pool.isEmpty()) return null;
        Collections.sort(pool);

        if (alternatingFirstRound) {
          int minimumGain = pool.stream().mapToInt(gains::get).min().orElseThrow();
          for (int j : pool)
            if (gains.get(j) == minimumGain) {
              sel.add(j);
              unc.removeAll(targets.get(j));
            }
          alternatingFirstRound = false;
        } else {
          int selectedGain;
          if (alternatingChooseMaximum) {
            selectedGain = pool.stream().mapToInt(gains::get).max().orElseThrow();
          } else {
            int minimumAtLeastTwo =
                pool.stream().mapToInt(gains::get).filter(gain -> gain >= 2).min().orElse(-1);
            selectedGain =
                minimumAtLeastTwo >= 0
                    ? minimumAtLeastTwo
                    : pool.stream().mapToInt(gains::get).min().orElseThrow();
          }
          List<Integer> tied = pool.stream().filter(j -> gains.get(j) == selectedGain).toList();
          int j = rng.choice(tied);
          sel.add(j);
          unc.removeAll(targets.get(j));
          alternatingChooseMaximum = !alternatingChooseMaximum;
        }
        continue;
      }
      if (strategy.equals("farthest_distance")) {
        int farthest =
            unc.stream()
                .max(
                    Comparator.comparingDouble((Integer t) -> nearestFacilityDistance(t))
                        .thenComparing((a, b) -> Integer.compare(b, a)))
                .orElseThrow();

        List<Integer> coverers =
            candidates.get(farthest).stream()
                .filter(j -> !sel.contains(j) && j != farthest)
                .toList();
        int representative;
        if (coverers.isEmpty()) {
          if (!reachable.contains(farthest) || sel.contains(farthest)) return null;
          representative = farthest;
        } else {
          double nearest =
              coverers.stream().mapToDouble(this::nearestFacilityDistance).min().orElseThrow();
          List<Integer> tied =
              coverers.stream()
                  .filter(j -> Math.abs(nearestFacilityDistance(j) - nearest) <= EPS)
                  .sorted()
                  .toList();
          representative = tied.size() == 1 ? tied.get(0) : rng.choice(tied);
        }
        sel.add(representative);
        unc.removeAll(targets.get(representative));
        continue;
      }
      throw new IllegalArgumentException("Unknown assignment repair operator: " + strategy);
    }
    CoveragePlan p = assign(sel);
    if (p != null) return p;
    List<Integer> remaining = new ArrayList<>(all);
    remaining.removeAll(sel);
    remaining.sort(
        Comparator.comparingInt((Integer t) -> candidates.get(t).size())
            .thenComparing((a, b) -> Double.compare(in.demand[b], in.demand[a])));
    for (int t : remaining) {
      List<Integer> a =
          candidates.get(t).stream()
              .filter(j -> !sel.contains(j))
              .sorted(
                  Comparator.comparingDouble((Integer j) -> in.demand[j])
                      .thenComparingDouble(
                          j -> nearestOpenFacilityDistance[j]))
              .toList();
      if (a.isEmpty()) continue;
      sel.add(a.get(0));
      p = assign(sel);
      if (p != null) return p;
    }
    return null;
  }

  // ==================== Capacity-constrained coverage assignment ====================

  /** Assigns every customer exactly once to a warehouse or selected representative. */
  private CoveragePlan assign(Set<Integer> sel) {
    SelectionKey key = new SelectionKey(sel);
    Optional<CoveragePlan> cached = assignmentCache.get(key);
    if (cached != null) return cached.orElse(null);
    CoveragePlan result = assignUncached(sel);
    assignmentCache.put(key, Optional.ofNullable(result));
    return result;
  }

  private CoveragePlan assignUncached(Set<Integer> sel) {
    double[] loadValues = new double[in.customerCount()];
    for (int j : sel) {
      if (!reachable.contains(j) || in.demand[j] > in.capacity + EPS) return null;
      loadValues[j] = in.demand[j];
    }
    Map<Integer, Integer> a = new HashMap<>();
    for (int j : sel) a.put(j, j);
    List<Integer> ts = new ArrayList<>();
    for (int j : all) if (!sel.contains(j)) ts.add(j);
    Map<Integer, List<Integer>> opts = new HashMap<>();
    for (int t : ts) {
      List<Integer> x = new ArrayList<>();
      for (int j : candidates.get(t)) if (sel.contains(j)) x.add(j);
      if (nearestWarehouse[t] >= 0) x.add(-1 - nearestWarehouse[t]);
      if (x.isEmpty()) return null;
      opts.put(t, x);
    }
    ts.sort(
        Comparator.comparingInt((Integer t) -> opts.get(t).size())
            .thenComparing((x, y) -> Double.compare(in.demand[y], in.demand[x])));
    nodes = 0;
    if (!place(0, ts, opts, loadValues, a)) return null;
    Map<Integer, Double> loads = new HashMap<>();
    for (int j : sel) loads.put(j, loadValues[j]);
    return planFromProviders(sel, a, loads);
  }

  /** Runs capacity-constrained backtracking within the node budget. */
  private boolean place(
      int pos,
      List<Integer> ts,
      Map<Integer, List<Integer>> opts,
      double[] loads,
      Map<Integer, Integer> a) {
    if (++nodes > cfg.assignmentBacktrackNodes) return false;
    if (pos == ts.size()) return true;
    int t = ts.get(pos);
    double demand = in.demand[t];
    List<Integer> choices = opts.get(t);
    int size = choices.size();
    int[] order = new int[size];
    double[] first = new double[size], second = new double[size];
    for (int i = 0; i < size; i++) {
      int candidate = choices.get(i);
      double load = candidate < 0 ? 0 : (loads[candidate] + demand) / in.capacity,
          distance = serviceLinkCost(t, candidate),
          f = load,
          s = distance;
      int at = i;
      while (at > 0
          && comparePlacement(f, s, candidate, first[at - 1], second[at - 1], order[at - 1])
              < 0) {
        first[at] = first[at - 1];
        second[at] = second[at - 1];
        order[at] = order[at - 1];
        at--;
      }
      first[at] = f;
      second[at] = s;
      order[at] = candidate;
    }
    for (int j : order) {
      if (j < 0) {
        a.put(t, j);
        if (place(pos + 1, ts, opts, loads, a)) return true;
        a.remove(t);
        continue;
      }
      if (loads[j] + demand > in.capacity + EPS) continue;
      loads[j] += demand;
      a.put(t, j);
      if (place(pos + 1, ts, opts, loads, a)) return true;
      loads[j] -= demand;
      a.remove(t);
    }
    return false;
  }

  private static int comparePlacement(
      double firstA,
      double secondA,
      int candidateA,
      double firstB,
      double secondB,
      int candidateB) {
    int c = Double.compare(firstA, firstB);
    if (c == 0) c = Double.compare(secondA, secondB);
    return c != 0 ? c : Integer.compare(candidateA, candidateB);
  }

  /** Evaluates and caches an assignment with the quick route configuration. */
  private Result evaluate(CoveragePlan p) {
    List<RoutePlan> routes = cachedRoutes(p, quick, evaluationCache);
    return routes == null ? null : result(p, routes, true);
  }

  private List<RoutePlan> cachedRoutes(
      CoveragePlan plan,
      RouteSearch.Config routeConfig,
      Map<CoverageKey, Optional<List<RoutePlan>>> cache) {
    CoverageKey key = new CoverageKey(plan, in.customerCount());
    Optional<List<RoutePlan>> cached = cache.get(key);
    if (cached != null) return cached.orElse(null);
    List<RoutePlan> routes =
        buildAndImproveRoutes(in, fac, plan.selected, plan.groupLoad, routeConfig);
    cache.put(key, Optional.ofNullable(routes));
    return routes;
  }

  // ==================== Assignment destroy-repair search ====================

  /** Runs windowed random/adaptive destroy-repair search with strict improvement acceptance. */
  private Result assignmentDestroyRepairSearch(Result initial) {
    Result cur = initial, best = initial;
    List<String>
        dops =
            cfg.enableClusterOperators
                ? DESTROY
                : DESTROY.stream().filter(x -> !x.equals("coverage_count")).toList(),
        rops = REPAIR;
    AdaptiveOperatorSelection.State adaptive =
        new AdaptiveOperatorSelection.State(dops, rops, adaptiveParameters());
    for (int it = 0; it < cfg.alnsIterations; it++) {
      AdaptiveOperatorSelection.Selection selection = adaptive.select(rng, it);
      String d = selection.destroy(), r = selection.repair();
      Set<Integer> base =
          applyAssignmentDestroyOperator(cur, d, rng.uniform(cfg.destroyMin, cfg.destroyMax));
      CoveragePlan p = construct(base, r);
      Result cand = p == null ? null : evaluate(p);
      boolean improved = cand != null && cand.serviceCost + EPS < cur.serviceCost;
      double improvement = improved ? cur.serviceCost - cand.serviceCost : 0;
      if (cand != null) {
        if (cand.serviceCost + EPS < best.serviceCost) {
          best = cand;
          if (d.equals("coverage_count")) inc("coverage_count_destroy_best_improvements");
        }
        if (improved) cur = cand;
      }
      adaptive.record(selection, improved, improvement, it);
    }
    return best;
  }

  private AdaptiveOperatorSelection.Parameters adaptiveParameters() {
    return new AdaptiveOperatorSelection.Parameters(
        cfg.adaptiveWindowLength,
        cfg.adaptiveProbability,
        cfg.probabilityFluctuationCoefficient,
        cfg.adaptiveHistoryWeight,
        cfg.adaptiveScoreDecay,
        cfg.adaptiveBaseReward,
        cfg.adaptiveImprovementReward,
        cfg.adaptiveRejectionPenalty,
        cfg.adaptiveMinimumWeight,
        cfg.adaptiveMaximumWeight);
  }

  // ==================== 3. Assignment local-search operators: complete -> complete ====================

  /** Runs one move per iteration, reshuffling all five moves at the start of each group. */
  private Result unifiedAssignmentLocalSearch(Result initial) {
    Result current = initial;
    List<String> operators =
        List.of(
            "representative_demotion",
            "representative_promotion",
            "representative_swap",
            "representative_split",
            "customer_reassignment");
    // The stage budget already counts individual operator iterations.
    long iterationLimit = Math.max(0, cfg.unifiedLocalSearchPasses);
    List<String> order = new ArrayList<>();
    for (long iteration = 0; iteration < iterationLimit; iteration++) {
      int operatorIndex = (int) (iteration % operators.size());
      if (operatorIndex == 0) {
        order = new ArrayList<>(operators);
        rng.shuffle(order);
      }
      String operator = order.get(operatorIndex);
      inc("unified_local_iterations");
      inc("unified_" + operator + "_attempts");
      List<CoveragePlan> plans = unifiedLocalCandidates(current, operator);
      Result move = deepenUnifiedLocal(plans);
      if (move != null && move.serviceCost + EPS < current.serviceCost) {
        current = move;
        inc("unified_" + operator + "_improvements");
      }
    }
    return current;
  }

  private List<CoveragePlan> unifiedLocalCandidates(Result current, String operator) {
    boolean greedy = rng.random() < .5;
    return switch (operator) {
      case "representative_demotion" ->
          greedy ? redundantDemotionCandidates(current) : randomDemotionCandidates(current);
      case "representative_promotion" ->
          greedy ? maximumCoveragePromotionCandidates(current) : randomPromotionCandidates(current);
      case "representative_swap" ->
          greedy ? densityRepresentativeSwapCandidates(current)
              : randomRepresentativeSwapCandidates(current);
      case "representative_split" ->
          greedy ? greedyRepresentativeSplitCandidates(current)
              : randomRepresentativeSplitCandidates(current);
      case "customer_reassignment" ->
          greedy ? greedyReassignmentCandidates(current) : randomReassignmentCandidates(current);
      default -> throw new IllegalArgumentException("Unknown unified local operator: " + operator);
    };
  }

  private List<CoveragePlan> randomDemotionCandidates(Result current) {
    List<Integer> eligible =
        current.coverage.selected.stream()
            .filter(
                representative ->
                    warehouseEligible.contains(representative) || current.coverage.selected.stream()
                        .anyMatch(
                            other ->
                                other != representative
                                    && in.customerCoverMembership[representative][other]))
            .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
    if (eligible.isEmpty()) return List.of();
    rng.shuffle(eligible);
    int limit = Math.min(3, eligible.size());
    List<CoveragePlan> plans = new ArrayList<>();
    for (int count = 1; count <= limit; count++) {
      Set<Integer> selected = new HashSet<>(current.coverage.selected);
      selected.removeAll(eligible.subList(0, count));
      addLocallyCompleted(plans, current.coverage, selected);
    }
    return plans;
  }

  private List<CoveragePlan> redundantDemotionCandidates(Result current) {
    Map<Integer, Integer> routeOf = new HashMap<>();
    for (int route = 0; route < current.routes.size(); route++)
      for (int representative : current.routes.get(route).customers)
        routeOf.put(representative, route);
    List<Integer> eligible = new ArrayList<>();
    for (int representative : current.coverage.selected) {
      Integer route = routeOf.get(representative);
      if (route == null) continue;
      List<Integer> alternatives =
          current.routes.get(route).customers.stream()
              .filter(other -> other != representative)
              .toList();
      List<Integer> group = new ArrayList<>();
      if (all.contains(representative)) group.add(representative);
      for (var entry : current.coverage.assignment.entrySet())
        if (entry.getValue() == representative && entry.getKey() != representative)
          group.add(entry.getKey());
      boolean repairable = true;
      for (int customer : group)
        if (!warehouseEligible.contains(customer) && alternatives.stream()
            .noneMatch(other -> in.customerCoverMembership[customer][other])) {
          repairable = false;
          break;
        }
      if (repairable) eligible.add(representative);
    }
    eligible.sort(
        Comparator.comparingInt(
                (Integer representative) ->
                    (int)
                        current.coverage.assignment.values().stream()
                            .filter(value -> value == representative)
                            .count())
            .thenComparingDouble(
                representative -> current.coverage.groupLoad.get(representative)));
    List<CoveragePlan> plans = new ArrayList<>();
    for (int representative :
        eligible.subList(0, Math.min(cfg.unifiedLocalCandidateLimit, eligible.size()))) {
      Set<Integer> selected = new HashSet<>(current.coverage.selected);
      selected.remove(representative);
      addLocallyCompleted(plans, current.coverage, selected);
    }
    return plans;
  }

  private List<CoveragePlan> maximumCoveragePromotionCandidates(Result current) {
    List<Integer> ranked =
        reachable.stream()
            .filter(candidate -> !current.coverage.selected.contains(candidate))
            .filter(candidate -> localCoverageGain(candidate, current.coverage.selected) > 0)
            .sorted(
                Comparator.comparingInt(
                        (Integer candidate) ->
                            -localCoverageGain(candidate, current.coverage.selected))
                    .thenComparingInt(candidate -> candidate))
            .limit(2)
            .toList();
    if (ranked.isEmpty()) return List.of();
    Set<Integer> selected = new HashSet<>(current.coverage.selected);
    selected.addAll(ranked);
    List<CoveragePlan> plans = new ArrayList<>();
    addLocallyCompleted(plans, current.coverage, selected);
    return plans;
  }

  private List<CoveragePlan> randomPromotionCandidates(Result current) {
    List<Integer> eligible =
        reachable.stream()
            .filter(candidate -> !current.coverage.selected.contains(candidate))
            .filter(candidate -> targets.getOrDefault(candidate, Set.of()).stream()
                .anyMatch(customer -> customer != candidate))
            .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
    if (eligible.isEmpty()) return List.of();
    rng.shuffle(eligible);
    List<CoveragePlan> plans = new ArrayList<>();
    for (int candidate :
        eligible.subList(0, Math.min(cfg.unifiedRandomSamples, eligible.size()))) {
      Set<Integer> selected = new HashSet<>(current.coverage.selected);
      selected.add(candidate);
      addLocallyCompleted(plans, current.coverage, selected);
    }
    return plans;
  }

  private List<CoveragePlan> randomRepresentativeSwapCandidates(Result current) {
    List<int[]> pairs = new ArrayList<>();
    for (int representative : current.coverage.selected)
      for (int candidate : reachable)
        if (!current.coverage.selected.contains(candidate)
            && in.customerCoverMembership[representative][candidate]
            && in.customerCoverMembership[candidate][representative])
          pairs.add(new int[] {representative, candidate});
    if (pairs.isEmpty()) return List.of();
    rng.shuffle(pairs);
    List<CoveragePlan> plans = new ArrayList<>();
    for (int[] pair : pairs.subList(0, Math.min(cfg.unifiedRandomSamples, pairs.size()))) {
      Set<Integer> selected = new HashSet<>(current.coverage.selected);
      selected.remove(pair[0]);
      selected.add(pair[1]);
      addLocallyCompleted(plans, current.coverage, selected);
    }
    return plans;
  }

  private List<CoveragePlan> densityRepresentativeSwapCandidates(Result current) {
    Integer densest =
        reachable.stream()
            .max(
                Comparator.comparingInt(
                        (Integer candidate) ->
                            localCoverageGain(candidate, current.coverage.selected))
                    .thenComparingInt(candidate -> -candidate))
            .orElse(null);
    if (densest == null) return List.of();
    List<Integer> alternatives;
    Set<Integer> selected = new HashSet<>(current.coverage.selected);
    if (selected.contains(densest)) {
      alternatives =
          reachable.stream()
              .filter(candidate -> !selected.contains(candidate))
              .filter(candidate -> in.customerCoverMembership[candidate][densest])
              .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
      if (alternatives.isEmpty()) return List.of();
      rng.shuffle(alternatives);
      List<CoveragePlan> plans = new ArrayList<>();
      for (int candidate :
          alternatives.subList(0, Math.min(cfg.unifiedRandomSamples, alternatives.size()))) {
        Set<Integer> trial = new HashSet<>(selected);
        trial.remove(densest);
        trial.add(candidate);
        addLocallyCompleted(plans, current.coverage, trial);
      }
      return plans;
    }
    alternatives =
        selected.stream()
            .filter(representative -> in.customerCoverMembership[representative][densest])
            .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
    if (alternatives.isEmpty()) {
      selected.add(densest);
      List<CoveragePlan> plans = new ArrayList<>();
      addLocallyCompleted(plans, current.coverage, selected);
      return plans;
    }
    rng.shuffle(alternatives);
    List<CoveragePlan> plans = new ArrayList<>();
    for (int representative :
        alternatives.subList(0, Math.min(cfg.unifiedRandomSamples, alternatives.size()))) {
      Set<Integer> trial = new HashSet<>(selected);
      trial.remove(representative);
      trial.add(densest);
      addLocallyCompleted(plans, current.coverage, trial);
    }
    return plans;
  }

  private List<CoveragePlan> randomReassignmentCandidates(Result current) {
    List<Integer> movable = new ArrayList<>();
    for (int t : all)
      if (!current.coverage.selected.contains(t) &&
          providers(t, current.coverage.selected).stream().anyMatch(k -> k != providerOf(current.coverage,t)))
        movable.add(t);
    rng.shuffle(movable);
    List<CoveragePlan> plans = new ArrayList<>();
    for (int t : movable.subList(0, Math.min(cfg.unifiedRandomSamples, movable.size()))) {
      List<Integer> alternatives = providers(t, current.coverage.selected);
      alternatives.remove(Integer.valueOf(providerOf(current.coverage,t)));
      rng.shuffle(alternatives);
      CoveragePlan p = changed(current.coverage, List.of(new int[]{t,alternatives.get(0)}));
      if (p != null) plans.add(p);
    }
    return plans;
  }

  private record ReassignmentCandidate(
      double saving, int customer, int provider, CoveragePlan plan) {}

  /** Keeps only single-customer reassignments with a strict connection-cost saving. */
  private List<CoveragePlan> greedyReassignmentCandidates(Result current) {
    List<ReassignmentCandidate> ranked = new ArrayList<>();
    for (int customer : all.stream().sorted().toList()) {
      if (current.coverage.selected.contains(customer)) continue;
      int oldProvider = providerOf(current.coverage, customer);
      for (int newProvider : providers(customer, current.coverage.selected)) {
        if (newProvider == oldProvider) continue;
        CoveragePlan plan =
            changed(current.coverage, List.of(new int[] {customer, newProvider}));
        if (plan == null) continue;
        double saving =
            serviceLinkCost(customer, oldProvider) - serviceLinkCost(customer, newProvider);
        if (saving > EPS)
          ranked.add(new ReassignmentCandidate(saving, customer, newProvider, plan));
      }
    }
    ranked.sort(
        Comparator.comparingDouble(ReassignmentCandidate::saving)
            .reversed()
            .thenComparing(
                Comparator.comparingDouble(
                        (ReassignmentCandidate candidate) -> in.demand[candidate.customer()])
                    .reversed())
            .thenComparingInt(ReassignmentCandidate::customer)
            .thenComparingInt(ReassignmentCandidate::provider));
    return ranked.stream()
        .limit(cfg.unifiedLocalCandidateLimit)
        .map(ReassignmentCandidate::plan)
        .toList();
  }

  private record SplitPair(int first, int second, int firstGain, int secondGain) {}

  private record SplitNeighborhood(int removed, Set<Integer> group, List<SplitPair> pairs) {}

  private List<CoveragePlan> randomRepresentativeSplitCandidates(Result current) {
    List<SplitNeighborhood> neighborhoods = splitNeighborhoods(current.coverage);
    if (neighborhoods.isEmpty()) return List.of();
    rng.shuffle(neighborhoods);
    SplitNeighborhood neighborhood = neighborhoods.get(0);
    List<SplitPair> pairs = new ArrayList<>(neighborhood.pairs());
    rng.shuffle(pairs);
    return firstFeasibleStrictSplit(current.coverage, neighborhood, pairs);
  }

  private List<CoveragePlan> greedyRepresentativeSplitCandidates(Result current) {
    List<SplitNeighborhood> neighborhoods = splitNeighborhoods(current.coverage);
    if (neighborhoods.isEmpty()) return List.of();
    neighborhoods.sort(
        Comparator.comparingDouble(
                (SplitNeighborhood neighborhood) ->
                    -current.coverage.groupLoad.getOrDefault(neighborhood.removed(), 0.0))
            .thenComparingInt(SplitNeighborhood::removed));
    SplitNeighborhood neighborhood = neighborhoods.get(0);
    List<SplitPair> pairs = new ArrayList<>(neighborhood.pairs());
    pairs.sort(
        Comparator.comparingInt((SplitPair pair) -> -pair.firstGain())
            .thenComparingInt(pair -> -pair.secondGain())
            .thenComparingInt(SplitPair::first)
            .thenComparingInt(SplitPair::second));
    return firstFeasibleStrictSplit(current.coverage, neighborhood, pairs);
  }

  private List<SplitNeighborhood> splitNeighborhoods(CoveragePlan base) {
    List<SplitNeighborhood> neighborhoods = new ArrayList<>();
    for (int removed : new TreeSet<>(base.selected)) {
      Set<Integer> group = new HashSet<>();
      for (var entry : base.assignment.entrySet())
        if (entry.getValue() == removed) group.add(entry.getKey());
      if (group.size() < 3) continue;
      List<Integer> alternatives =
          group.stream()
              .filter(candidate -> candidate != removed)
              .filter(reachable::contains)
              .filter(candidate -> in.demand[candidate] <= in.capacity + EPS)
              .filter(
                  candidate -> {
                    Set<Integer> others = new HashSet<>(group);
                    others.remove(candidate);
                    return overlapCount(targets.getOrDefault(candidate, Set.of()), others) > 0;
                  })
              .sorted()
              .toList();
      if (alternatives.size() < 2) continue;
      Map<Long, SplitPair> unique = new LinkedHashMap<>();
      for (int first : alternatives) {
        Set<Integer> remaining = new HashSet<>(group);
        remaining.removeAll(targets.getOrDefault(first, Set.of()));
        if (remaining.isEmpty()) continue;
        int firstGain = overlapCount(targets.getOrDefault(first, Set.of()), group);
        for (int second : alternatives) {
          if (second == first || !remaining.contains(second)) continue;
          int secondGain = overlapCount(targets.getOrDefault(second, Set.of()), remaining);
          if (secondGain == 0) continue;
          int low = Math.min(first, second), high = Math.max(first, second);
          long key = ((long) low << 32) ^ Integer.toUnsignedLong(high);
          SplitPair candidate = new SplitPair(first, second, firstGain, secondGain);
          SplitPair old = unique.get(key);
          if (old == null
              || candidate.firstGain() > old.firstGain()
              || (candidate.firstGain() == old.firstGain()
                  && candidate.secondGain() > old.secondGain())) unique.put(key, candidate);
        }
      }
      if (!unique.isEmpty())
        neighborhoods.add(new SplitNeighborhood(removed, group, new ArrayList<>(unique.values())));
    }
    return neighborhoods;
  }

  private List<CoveragePlan> firstFeasibleStrictSplit(
      CoveragePlan base, SplitNeighborhood neighborhood, List<SplitPair> pairs) {
    int attempts = Math.min(Math.max(0, cfg.representativeSplitAttempts), pairs.size());
    for (SplitPair pair : pairs.subList(0, attempts)) {
      CoveragePlan plan =
          strictSplitPlan(
              base,
              neighborhood.removed(),
              neighborhood.group(),
              pair.first(),
              pair.second());
      if (plan != null) return List.of(plan);
    }
    return List.of();
  }

  /** Strictly assigns the original cluster to two new in-cluster representatives. */
  private CoveragePlan strictSplitPlan(
      CoveragePlan base, int removed, Set<Integer> group, int first, int second) {
    if (first == second
        || first == removed
        || second == removed
        || !group.contains(first)
        || !group.contains(second)
        || !reachable.contains(first)
        || !reachable.contains(second)) return null;
    Set<Integer> jointlyCovered = new HashSet<>(targets.getOrDefault(first, Set.of()));
    jointlyCovered.addAll(targets.getOrDefault(second, Set.of()));
    if (!jointlyCovered.containsAll(group)) return null;

    Set<Integer> selected = new HashSet<>(base.selected);
    selected.remove(removed);
    selected.add(first);
    selected.add(second);
    Map<Integer, Integer> providers = providerMap(base);
    providers.put(first, first);
    providers.put(second, second);
    Map<Integer, Double> loads = new HashMap<>(base.groupLoad);
    loads.remove(removed);
    loads.put(first, in.demand[first]);
    loads.put(second, in.demand[second]);
    if (loads.get(first) > in.capacity + EPS || loads.get(second) > in.capacity + EPS) return null;

    List<Integer> pending = new ArrayList<>();
    for (int customer : group)
      if (customer != first && customer != second) pending.add(customer);
    pending.sort(
        Comparator.comparingInt(
                (Integer customer) ->
                    (targets.get(first).contains(customer) ? 1 : 0)
                        + (targets.get(second).contains(customer) ? 1 : 0))
            .thenComparing((a, b) -> Double.compare(in.demand[b], in.demand[a]))
            .thenComparingInt(customer -> customer));
    int[] budget = {Math.max(1, cfg.assignmentBacktrackNodes)};
    if (!placeStrictSplit(0, pending, first, second, providers, loads, budget)) return null;
    return planFromProviders(selected, providers, loads);
  }

  private boolean placeStrictSplit(
      int position,
      List<Integer> pending,
      int first,
      int second,
      Map<Integer, Integer> providers,
      Map<Integer, Double> loads,
      int[] budget) {
    if (--budget[0] < 0) return false;
    if (position == pending.size()) return true;
    int customer = pending.get(position);
    List<Integer> choices = new ArrayList<>(2);
    if (targets.get(first).contains(customer)) choices.add(first);
    if (targets.get(second).contains(customer)) choices.add(second);
    choices.sort(
        Comparator.comparingDouble(
                (Integer representative) ->
                    (loads.get(representative) + in.demand[customer]) / in.capacity)
            .thenComparingDouble(representative -> serviceLinkCost(customer, representative))
            .thenComparingInt(representative -> representative));
    for (int representative : choices) {
      if (loads.get(representative) + in.demand[customer] > in.capacity + EPS) continue;
      loads.put(representative, loads.get(representative) + in.demand[customer]);
      providers.put(customer, representative);
      if (placeStrictSplit(position + 1, pending, first, second, providers, loads, budget))
        return true;
      loads.put(representative, loads.get(representative) - in.demand[customer]);
    }
    return false;
  }

  private int overlapCount(Set<Integer> first, Set<Integer> second) {
    if (first == null || first.isEmpty() || second.isEmpty()) return 0;
    int count = 0;
    Set<Integer> smaller = first.size() <= second.size() ? first : second;
    Set<Integer> larger = smaller == first ? second : first;
    for (int value : smaller) if (larger.contains(value)) count++;
    return count;
  }

  private int localCoverageGain(int representative, Set<Integer> selected) {
    int gain = 0;
    for (int customer : targets.getOrDefault(representative, Set.of()))
      if (!selected.contains(customer)) gain++;
    return gain;
  }

  private void addLocallyCompleted(
      List<CoveragePlan> plans, CoveragePlan base, Set<Integer> selected) {
    CoveragePlan completed = locallyCompleteChangedRegion(base, selected);
    if (completed != null) plans.add(completed);
  }

  /** Rebuilds only the assignment region affected by representative changes. */
  private CoveragePlan locallyCompleteChangedRegion(CoveragePlan base, Set<Integer> requested) {
    Set<Integer> selected = new HashSet<>(requested);
    if (!reachable.containsAll(selected)) return null;
    Set<Integer> changedNodes = new HashSet<>(base.selected);
    changedNodes.removeAll(selected);
    Set<Integer> added = new HashSet<>(selected);
    added.removeAll(base.selected);
    changedNodes.addAll(added);
    Set<Integer> affected = new HashSet<>(changedNodes);
    for(int j:changedNodes) affected.addAll(targets.getOrDefault(j,Set.of()));
    for(var e:base.assignment.entrySet())
      if(!selected.contains(e.getValue()) || changedNodes.contains(e.getValue())) affected.add(e.getKey());
    Map<Integer,Integer> assignment=new HashMap<>();
    Map<Integer,Double> loads=new HashMap<>();
    for(int j:selected) { loads.put(j,in.demand[j]); assignment.put(j,j); }
    for(int t:new TreeSet<>(all)) {
      if(selected.contains(t)) continue;
      int provider=providerOf(base,t);
      if(!affected.contains(t) && canServe(t,provider,selected)) {
        assignment.put(t,provider);
        if(provider>=0) loads.put(provider,loads.get(provider)+in.demand[t]);
      } else affected.add(t);
    }
    for(double load:loads.values()) if(load>in.capacity+EPS) return null;
    List<Integer> pending=affected.stream().filter(t->!selected.contains(t))
        .sorted(Comparator.comparingInt((Integer t)->providers(t,selected).size())
            .thenComparing((a,b)->Double.compare(in.demand[b],in.demand[a]))).toList();
    boolean preferLowLoad=rng.randrange(4)==0;
    inc(preferLowLoad?"local_repair_low_load_calls":"local_repair_distance_calls");
    for (int customer : pending) {
      List<Integer> covering = new ArrayList<>(providers(customer, selected));
      covering.sort(Comparator.comparingDouble((Integer k) -> serviceLinkCost(customer, k))
          .thenComparingDouble(k -> k < 0 ? 0 : loads.get(k)).thenComparingInt(k -> k));
      Integer chosen = null;
      if (!covering.isEmpty()) {
        if (preferLowLoad) {
          chosen =
              covering.stream()
                  .filter(
                      representative ->
                          representative < 0 || loads.get(representative) + in.demand[customer] <= in.capacity + EPS)
                  .min(
                      Comparator.comparingDouble(
                              (Integer representative) -> (representative < 0 ? 0 : loads.get(representative)))
                          .thenComparingDouble(
                              representative ->
                                  serviceLinkCost(customer, representative))
                          .thenComparingInt(representative -> representative))
                  .orElse(null);
        } else {
          int nearest = covering.get(0);
          if (nearest < 0 || loads.get(nearest) + in.demand[customer] <= in.capacity + EPS) chosen = nearest;
          else
            chosen =
                covering.stream()
                    .filter(
                        representative -> representative < 0 || loads.get(representative) + in.demand[customer] <= in.capacity + EPS)
                    .min(
                        Comparator.comparingDouble(
                                (Integer representative) ->
                                    .70
                                            * serviceLinkCost(customer, representative)
                                            / Math.max(EPS, ExperimentParameters.W_C * ExperimentParameters.C * in.customerRadius)
                                        + .30 * (representative < 0 ? 0 : loads.get(representative)) / in.capacity)
                            .thenComparingInt(representative -> representative))
                    .orElse(null);
        }
      }
      if (chosen != null) {
        assignment.put(customer, chosen);
        if (chosen >= 0) loads.put(chosen, loads.get(chosen) + in.demand[customer]);
      } else if (reachable.contains(customer) && in.demand[customer] <= in.capacity + EPS) {
        selected.add(customer);
        loads.put(customer, in.demand[customer]);
        assignment.put(customer, customer);
      } else return null;
    }
    if(!assignment.keySet().equals(all) || !loads.keySet().equals(selected)) return null;
    return planFromProviders(selected,assignment,loads);
  }

  /** Re-evaluates screened candidates with the full route budget. */
  private Result deepenUnifiedLocal(List<CoveragePlan> plans) {
    Map<CoverageKey, CoveragePlan> unique = new LinkedHashMap<>();
    for (CoveragePlan plan : plans)
      unique.putIfAbsent(new CoverageKey(plan, in.customerCount()), plan);
    List<Object[]> screened = new ArrayList<>();
    for (CoveragePlan plan : unique.values()) {
      Result evaluation = evaluate(plan);
      if (evaluation != null) screened.add(new Object[] {evaluation.serviceCost, plan});
    }
    screened.sort(Comparator.comparingDouble(item -> (double) item[0]));
    Result best = null;
    for (Object[] item :
        screened.subList(0, Math.min(cfg.unifiedDeepCandidates, screened.size()))) {
      CoveragePlan plan = (CoveragePlan) item[1];
      List<RoutePlan> routes = cachedRoutes(plan, routeCfg, fullEvaluationCache);
      if (routes != null && (best == null || serviceCost(plan, routes) + EPS < best.serviceCost))
        best = result(plan, routes, true);
    }
    return best;
  }

  // ==================== Complete-assignment local-search entry point ====================

  public Result improveCompleteAssignment(Result incumbent, int trials) {
    Map<String, Integer> inheritedStatistics =
        new LinkedHashMap<>(incumbent.operatorStatistics);
    Result best =
        result(
            incumbent.coverage.copy(),
            incumbent.routes.stream()
                .map(RoutePlan::copy)
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new)),
            incumbent.feasible,
            incumbent.reason,
            new HashMap<>(incumbent.operatorStatistics));
    for (int trial = 0;
        trial < Math.max(0, trials) && cfg.unifiedLocalSearchPasses > 0;
        trial++) {
      inc("assignment_postprocess_trials");
      double before = best.serviceCost;
      best = unifiedAssignmentLocalSearch(best);
      if (best.serviceCost + EPS >= before) break;
    }
    for (var entry : stats.entrySet())
      inheritedStatistics.merge(entry.getKey(), entry.getValue(), Integer::sum);
    best.operatorStatistics = inheritedStatistics;
    return best;
  }

  private CoveragePlan changed(CoveragePlan base, List<int[]> changes) {
    Map<Integer,Integer> a=new HashMap<>(base.assignment);
    for(var e:base.directAssignment.entrySet()) a.put(e.getKey(),-1-e.getValue());
    Map<Integer,Double> loads=new HashMap<>(base.groupLoad);
    for(int[] x:changes) {
      int t=x[0], add=x[1];
      Integer rem=a.get(t);
      if(rem==null || base.selected.contains(t) || rem==add || !canServe(t,add,base.selected)) return null;
      if(rem>=0) loads.put(rem,loads.get(rem)-in.demand[t]);
      if(add>=0) loads.put(add,loads.get(add)+in.demand[t]);
      a.put(t,add);
    }
    for(var e:loads.entrySet())
      if(e.getValue()>in.capacity+EPS || e.getValue()+EPS<in.demand[e.getKey()]) return null;
    return planFromProviders(base.selected,a,loads);
  }

  // Transient provider encoding: -1-k represents warehouse k.
  private Map<Integer,Integer> providerMap(CoveragePlan p) {
    Map<Integer,Integer> values=new HashMap<>(p.assignment);
    for(var e:p.directAssignment.entrySet()) values.put(e.getKey(),-1-e.getValue());
    return values;
  }

  private int providerOf(CoveragePlan p,int t) {
    Integer j=p.assignment.get(t);
    if(j!=null) return j;
    Integer k=p.directAssignment.get(t);
    return k==null?Integer.MIN_VALUE:-1-k;
  }

  private boolean canServe(int t,int k,Set<Integer> selected) {
    if(k==Integer.MIN_VALUE) return false;
    return k<0 ? fac.contains(-1-k) && in.facilityCoverMembership[-1-k][t]
        : selected.contains(k) && in.customerCoverMembership[k][t];
  }

  private List<Integer> providers(int t,Set<Integer> selected) {
    List<Integer> options=new ArrayList<>();
    for(int j:candidates.get(t)) if(selected.contains(j)) options.add(j);
    // Warehouses have no capacity limit and share one rate, so only the nearest is retained.
    if(nearestWarehouse[t]>=0) options.add(-1-nearestWarehouse[t]);
    return options;
  }

  double serviceLinkCost(int t,int k) {
    return k<0 ? ExperimentParameters.W_A*ExperimentParameters.L*in.facilityDistance[-1-k][t]
        : ExperimentParameters.W_C*ExperimentParameters.C*in.customerDistance[k][t];
  }

  private CoveragePlan planFromProviders(Set<Integer> selected,Map<Integer,Integer> providers,
      Map<Integer,Double> loads) {
    Map<Integer,Integer> a=new HashMap<>(), d=new HashMap<>();
    for(var e:providers.entrySet())
      if(e.getValue()<0) d.put(e.getKey(),-1-e.getValue());
      else a.put(e.getKey(),e.getValue());
    return new CoveragePlan(new HashSet<>(selected),a,new HashMap<>(loads),d);
  }

  private double serviceCost(CoveragePlan p,List<RoutePlan> routes) {
    return Model.costs(in,fac,p,routes).service();
  }

  Result result(CoveragePlan p,List<RoutePlan> routes,boolean feasible) {
    return result(p,routes,feasible,"",new HashMap<>());
  }

  private Result result(CoveragePlan p,List<RoutePlan> routes,boolean feasible,
      String reason,Map<String,Integer> statistics) {
    Result r=new Result(p,routes,feasible ? 0.0 : Double.POSITIVE_INFINITY,
        feasible,reason,statistics);
    if(feasible) {
      CostBreakdown c=Model.costs(in,fac,p,routes);
      r.routeCost=c.route();
      r.warehouseCoverageCost=c.warehouseCoverage();
      r.customerCoverageCost=c.customerCoverage();
      r.serviceCost=c.service();
      if(Boolean.getBoolean("audit.validateCandidates")) {
        HeuristicSolution solution=new HeuristicSolution(in.name,fac,p,routes,p.directAssignment,
            c.total(),c.facility(),c.route(),c.warehouseCoverage(),c.customerCoverage(),true,"",0);
        String error=Solver.validate(in,solution);
        if(error!=null) throw new AssertionError("Invalid candidate: "+error);
      }
    }
    return r;
  }


  // ==================== Statistics and collection helpers ====================

  private void inc(String k) {
    stats.put(k, stats.getOrDefault(k, 0) + 1);
  }

  private static Set<Integer> range(int n) {
    Set<Integer> s = new HashSet<>();
    for (int i = 0; i < n; i++) s.add(i);
    return s;
  }

}
