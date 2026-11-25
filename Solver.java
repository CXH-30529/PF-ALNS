package heuristic;

import static heuristic.AssignmentSearch.*;
import static heuristic.Model.*;

import java.nio.charset.StandardCharsets;
import java.util.*;

/** Coordinates the complete heuristic search. */
public final class Solver {
  /** Search configuration. */
  public static final class Config {
    // Facility-search budgets.
    public int seed = 20260803,
        facilitySearchIterations = 60,
        shakingInterval = 5,
        shakingBatchSize = 10,
        initialGreedyStarts = 4,
        maximumOpenFacilities = 6,
        deepAssignmentIterations = 140,
        deepAssignmentLocalSearchPasses = 50; // Single-operator iterations; legacy field name.
    // Optional search mechanisms.
    public boolean enableClusterOperators = true;
    // Simulated-annealing cooling rate.
    public double cooling = .975;
    // Component configurations.
    public Scores.FuzzyScoreConfig score = new Scores.FuzzyScoreConfig();
    public AssignmentSearch.Config coverage = new AssignmentSearch.Config();
    public RouteSearch.Config route = new RouteSearch.Config();
    public ProgressiveScreening.Config screening = new ProgressiveScreening.Config();
    /** Assignment destroy/repair, assignment local, route destroy/repair, route local, sweep.
     * Applied once to deep budgets, independently of screening levels and promotion ratios. */
    public double[] finalIntensificationRatios = {1.3, 1.0, 1.3, 1.0, 1.0};

    /** Creates the full search configuration. */
    public static Config quality(int seed) {
      Config c = new Config();
      c.seed = seed;
      c.facilitySearchIterations = 60;
      c.deepAssignmentIterations = 140;
      c.coverage.alnsIterations = 40;
      c.route.maxVnsPasses = 30;
      c.route.enableSwap = true;
      c.route.ruinIterations = 140;
      return c;
    }

    /** Adjusts the full configuration for the facility count. */
    public static Config quality(int seed, int facilityCount) {
      Config c = quality(seed);
      if (facilityCount > 5) c.initialGreedyStarts = 6;
      c.facilitySearchIterations = intProperty("allops.facilityIterations", c.facilitySearchIterations);
      c.deepAssignmentIterations =
          intProperty("allops.deepAssignmentIterations", c.deepAssignmentIterations);
      c.deepAssignmentLocalSearchPasses =
          intProperty("assignment.deepLocalPasses", c.deepAssignmentLocalSearchPasses);
      c.route.maxVnsPasses = intProperty("allops.deepRouteVnsPasses", c.route.maxVnsPasses);
      c.route.ruinIterations =
          intProperty("allops.deepRouteRuinIterations", c.route.ruinIterations);
      c.screening.readProperties();
      String finalRatios = System.getProperty("finalIntensification.budgetRatios");
      if (finalRatios != null && !finalRatios.isBlank())
        c.finalIntensificationRatios = Arrays.stream(finalRatios.split(",", -1))
            .map(String::trim).mapToDouble(Double::parseDouble).toArray();
      c.finalIntensificationBudget();
      return c;
    }

    /** Final-winner budgets; no screening coefficient or level count participates here. */
    public ProgressiveScreening.Budget finalIntensificationBudget() {
      if (finalIntensificationRatios == null || finalIntensificationRatios.length != 5)
        throw new IllegalArgumentException("finalIntensificationRatios needs exactly 5 values");
      return new ProgressiveScreening.Budget(
          finalBudget(deepAssignmentIterations, 0),
          finalBudget(deepAssignmentLocalSearchPasses, 1),
          finalBudget(route.maxVnsPasses, 3),
          finalBudget(route.ruinIterations, 2),
          finalBudget(route.sweepRotations, 4));
    }

    private int finalBudget(int deep, int index) {
      double ratio = finalIntensificationRatios[index];
      if (deep < 0 || !Double.isFinite(ratio) || ratio < 0)
        throw new IllegalArgumentException("Final budgets and coefficients must be finite and non-negative");
      double value = deep * ratio;
      if (!Double.isFinite(value) || value > Integer.MAX_VALUE)
        throw new IllegalArgumentException("Final budget exceeds the integer range");
      double nearest = Math.rint(value);
      return value == 0 ? 0 : Math.max(1,
          (int) Math.ceil(Math.abs(value - nearest) <= 1e-9 ? nearest : value));
    }

    private static int intProperty(String name, int fallback) {
      String value = System.getProperty(name);
      if (value == null || value.isBlank()) return fallback;
      int parsed = Integer.parseInt(value);
      if (parsed < 0) throw new IllegalArgumentException(name + " must be non-negative");
      return parsed;
    }

  }

  /** Statistics from one solver run. */
  public record Diagnostics(
      double runtimeSeconds,
      int uniqueFacilitySelections,
      int feasibleFacilitySelections,
      int cacheHits,
      int deepFacilityEvaluations,
      Map<String, Integer> operatorStatistics) {}

  private final Config cfg;
  /** Diagnostics from the latest run. */
  public Diagnostics lastDiagnostics;

  public Solver(Config c) {
    System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
    cfg = c == null ? new Config() : c;
  }

  // Per-run state.
  private Instance in;
  private FacilityMemory mem;
  private FacilityMemory.Entry incumbent;
  private int cacheHits;
  private FacilitySearch facilitySearch;
  private AssignmentSearch.Config exploreC, configuredC;
  private RouteSearch.Config exploreR, configuredR;
  private ProgressiveScreening.Budget finalBudget;
  private final Map<String, Integer> finalIntensificationStatistics = new LinkedHashMap<>();

  // Main workflow.

  /** Runs the complete layered heuristic. */
  public HeuristicSolution solve(Instance instance) {
    long started = System.nanoTime();
    initializeRun(instance);

    PythonRandom rng = new PythonRandom(Integer.toUnsignedLong(cfg.seed ^ stableHash(in.name)));
    FacilityMemory.Entry current = buildInitialSolution();

    searchFacilitySelections(current, rng);

    int deepEvaluations = reevaluatePromisingFacilitySelections();

    intensifyFinalWinner();

    return finishRun(started, deepEvaluations);
  }

  /** Initializes state and stage-specific budgets. */
  private void initializeRun(Instance instance) {
    finalBudget = cfg.finalIntensificationBudget();
    finalIntensificationStatistics.clear();
    in = instance;
    mem = new FacilityMemory();
    facilitySearch = new FacilitySearch(in, cfg, mem);
    incumbent = null;
    cacheHits = 0;
    configuredC = cfg.coverage.copy();
    configuredC.enableClusterOperators = cfg.enableClusterOperators;
    configuredC.unifiedLocalSearchPasses = cfg.deepAssignmentLocalSearchPasses;
    configuredR = cfg.route.copy();
    configuredR.experimentSeed ^= Integer.toUnsignedLong(cfg.seed);
    ProgressiveScreening.Budget shallow = ProgressiveScreening.budgetFor(cfg, 0);
    exploreC = configuredC.copy();
    exploreC.alnsIterations = shallow.assignmentIterations();
    exploreC.unifiedLocalSearchPasses = shallow.assignmentLocalPasses();
    exploreR = configuredR.copy();
    exploreR.maxVnsPasses = shallow.routeLocalPasses();
    exploreR.sweepRotations = shallow.routeSweepRotations();
    exploreR.ruinIterations = shallow.routeRuinIterations();
    exploreR.enableSwap = true;
  }

  // Initial solution.

  /** Builds and evaluates greedy facility selections. */
  private FacilityMemory.Entry buildInitialSolution() {
    LinkedHashSet<List<Integer>> unique = new LinkedHashSet<>();
    for (List<Integer> x : InitialSolution.facilityGreedyChains(facilitySearch))
      unique.add(canonicalFacilities(x));
    FacilityMemory.Entry current = null;
    int ix = 0;
    for (List<Integer> s : unique) {
      if (s.size() <= cfg.maximumOpenFacilities) {
        FacilityMemory.Entry e = evaluate(s, -unique.size() + ix++);
        if (e.feasible && (current == null || e.objective < current.objective)) current = e;
      }
    }
    if (current == null || incumbent == null)
      throw new IllegalStateException("No feasible facility selection found for " + in.name);
    return current;
  }

  // Facility search.

  /** Searches facility selections with shaking and neighborhood moves. */
  private void searchFacilitySelections(FacilityMemory.Entry current, PythonRandom rng) {
    List<List<Integer>> shakingPool = facilitySearch.allAllowedSelections();
    rng.shuffle(shakingPool);
    Set<List<Integer>> shakingVisitedSelections = new HashSet<>();
    int shakingPoolCursor = 0;
    double temp = Math.max(2, .035 * current.objective);
    int iterations = cfg.facilitySearchIterations;
    for (int it = 1; it <= iterations; it++) {
      if (facilityCoverageComplete(shakingPool, shakingPoolCursor)) break;

      if (cfg.shakingInterval > 0 && it % cfg.shakingInterval == 0) {
        FacilityMemory.Entry batchBest = null;
        int end = Math.min(shakingPool.size(), shakingPoolCursor + cfg.shakingBatchSize);
        while (shakingPoolCursor < end) {
          List<Integer> proposal = shakingPool.get(shakingPoolCursor++);
          if (!shakingVisitedSelections.add(proposal))
            throw new AssertionError("duplicate selection in first shaking coverage");
          FacilityMemory.Entry cand = evaluate(proposal, it);
          if (cand.feasible && (batchBest == null || cand.objective < batchBest.objective))
            batchBest = cand;
        }
        current = acceptFacilityCandidate(current, batchBest, rng, temp);
        if (facilityCoverageComplete(shakingPool, shakingPoolCursor)) break;
      }

      List<String> available = facilitySearch.availableOperators(current.selection);
      if (!available.isEmpty()) {
        String op = available.get(rng.randrange(available.size()));
        List<Integer> proposal = facilitySearch.applyOperator(current.selection, op, rng);
        if (proposal != null)
          current = acceptFacilityCandidate(current, evaluate(proposal, it), rng, temp);
      }
      temp *= cfg.cooling;
    }
    if (shakingPoolCursor == shakingPool.size()
        && (!shakingVisitedSelections.equals(new HashSet<>(shakingPool))
            || mem.uniqueEvaluations() != shakingPool.size()))
      throw new AssertionError("incomplete shaking/facility-cache coverage");
  }

  /** Applies the simulated-annealing acceptance rule. */
  private FacilityMemory.Entry acceptFacilityCandidate(
      FacilityMemory.Entry current, FacilityMemory.Entry candidate, PythonRandom rng, double temp) {
    if (candidate == null || !candidate.feasible) return current;
    double delta = candidate.objective - current.objective;
    return delta <= 0 || rng.random() < Math.exp(-delta / Math.max(EPS, temp))
        ? candidate
        : current;
  }

  /** Checks whether all legal selections have been evaluated. */
  private boolean facilityCoverageComplete(List<List<Integer>> all, int shakingPoolCursor) {
    return shakingPoolCursor == all.size() && mem.uniqueEvaluations() == all.size();
  }

  // Deep facility evaluation.

  /** Selects the strongest candidates for deep evaluation. */
  private List<FacilityMemory.Entry> selectDeepEvaluationCandidates(
      List<FacilityMemory.Entry> ranked, int count) {
    if (ranked.isEmpty() || count <= 0) return new ArrayList<>();
    return new ArrayList<>(ranked.subList(0, Math.min(count, ranked.size())));
  }

  /** Promotes candidates through L-1 evaluated levels; the incumbent is the single final winner. */
  private int reevaluatePromisingFacilitySelections() {
    int deepEvaluations = 0;
    List<FacilityMemory.Entry> ranked = mem.bestFeasible(mem.uniqueEvaluations());
    for (int level = 2; level < cfg.screening.resultLevelCount; level++) {
      int count = cfg.screening.promotionCount(ranked.size(), level - 1);
      List<FacilityMemory.Entry> candidates = selectDeepEvaluationCandidates(ranked, count);
      ProgressiveScreening.Budget budget = ProgressiveScreening.budgetFor(cfg, level - 1);
      AssignmentSearch.Config deep = configuredC.copy();
      deep.alnsIterations = budget.assignmentIterations();
      deep.unifiedLocalSearchPasses = budget.assignmentLocalPasses();
      RouteSearch.Config routes = configuredR.copy();
      routes.maxVnsPasses = budget.routeLocalPasses();
      routes.ruinIterations = budget.routeRuinIterations();
      routes.sweepRotations = budget.routeSweepRotations();
      List<FacilityMemory.Entry> evaluated = new ArrayList<>();
      for (int index = 0; index < candidates.size(); index++) {
        FacilityMemory.Entry e = candidates.get(index);
        // Each evaluation level uses a disjoint deterministic seed-salt range.
        int salt = (level - 2) * 1000000 + index + 1;
        Result r = solveFixed(e.selection, deep, routes, selectionSeed(e.selection, salt));
        deepEvaluations++;
        FacilityMemory.Entry retained = e;
        if (r.feasible) {
          double obj = e.selection.stream().mapToDouble(f -> ExperimentParameters.W_F * in.facilityCost[f]).sum() + r.serviceCost;
          if (obj + EPS < e.objective)
            retained = new FacilityMemory.Entry(e.selection, true, obj, r, e.lastIteration);
          if (obj + EPS < incumbent.objective)
            incumbent = new FacilityMemory.Entry(e.selection, true, obj, r, e.lastIteration);
        }
        evaluated.add(retained);
      }
      evaluated.sort(Comparator.comparingDouble(e -> e.objective));
      ranked = evaluated;
    }
    // Pick the winner from stored results without another assignment or route solve.
    // If promotion leaves no candidates, keep the existing feasible incumbent.
    if (!ranked.isEmpty()) incumbent = ranked.get(0);
    return deepEvaluations;
  }

  /** Reconstructs assignment/routes once for the final facilities; no restarts or postprocess. */
  private void intensifyFinalWinner() {
    AssignmentSearch.Config assignments = configuredC.copy();
    assignments.alnsIterations = finalBudget.assignmentIterations();
    assignments.unifiedLocalSearchPasses = finalBudget.assignmentLocalPasses();
    RouteSearch.Config routes = configuredR.copy();
    routes.ruinIterations = finalBudget.routeRuinIterations();
    routes.maxVnsPasses = finalBudget.routeLocalPasses();
    routes.sweepRotations = finalBudget.routeSweepRotations();
    // Negative salt reserves a reproducible seed domain separate from ordinary evaluations.
    // Assignment and route randomness both derive from the same external experiment seed.
    long seed = selectionSeed(incumbent.selection, -1);
    routes.experimentSeed = seed;
    finalIntensificationStatistics.put("final_intensification_calls", 1);
    finalIntensificationStatistics.put("final_assignment_destroy_repair_iterations", assignments.alnsIterations);
    finalIntensificationStatistics.put("final_assignment_local_iterations", assignments.unifiedLocalSearchPasses);
    finalIntensificationStatistics.put("final_route_destroy_repair_iterations", routes.ruinIterations);
    finalIntensificationStatistics.put("final_route_local_passes", routes.maxVnsPasses);
    finalIntensificationStatistics.put("final_route_sweep_rotations", routes.sweepRotations);
    Result result = solveFixed(incumbent.selection, assignments, routes, seed);
    boolean improved = result.feasible && result.serviceCost + EPS < incumbent.result.serviceCost;
    finalIntensificationStatistics.put("final_intensification_improvements", improved ? 1 : 0);
    if (improved) {
      double facilityCost = incumbent.selection.stream()
          .mapToDouble(f -> ExperimentParameters.W_F * in.facilityCost[f]).sum();
      incumbent = new FacilityMemory.Entry(incumbent.selection, true,
          facilityCost + result.serviceCost, result, incumbent.lastIteration);
    }
  }

  /** Validates the final solution and records diagnostics. */
  private HeuristicSolution finishRun(long started, int deepEvaluations) {
    HeuristicSolution sol = materialize(incumbent);
    String error = validate(in, sol);
    if (error != null)
      throw new AssertionError("Internal solution validation failed for " + in.name + ": " + error);
    Map<String, Integer> statistics = new HashMap<>(incumbent.result.operatorStatistics);
    statistics.putAll(finalIntensificationStatistics);
    lastDiagnostics =
        new Diagnostics(
            (System.nanoTime() - started) / 1e9,
            mem.uniqueEvaluations(),
            (int) mem.entries.values().stream().filter(e -> e.feasible).count(),
            cacheHits,
            deepEvaluations,
            statistics);
    return sol;
  }

  // Facility evaluation.

  /** Evaluates a facility selection or returns its cached result. */
  private FacilityMemory.Entry evaluate(List<Integer> s, int it) {
    s = canonicalFacilities(s);
    FacilityMemory.Entry e = mem.get(s);
    if (e != null) {
      cacheHits++;
      mem.touch(s, it);
      return e;
    }
    Result r = solveFixed(s, exploreC, exploreR, selectionSeed(s, 0));
    double fc = s.stream().mapToDouble(f -> ExperimentParameters.W_F * in.facilityCost[f]).sum();
    e = mem.remember(s, r, fc, it);
    if (e.feasible && (incumbent == null || e.objective < incumbent.objective)) incumbent = e;
    return e;
  }

  /** Solves assignment and routing for fixed facilities. */
  private Result solveFixed(
      List<Integer> s, AssignmentSearch.Config c, RouteSearch.Config r, long seed) {
    return s.isEmpty()
        ? new Result(
            null,
            new ArrayList<>(),
            Double.POSITIVE_INFINITY,
            false,
            "no open facility",
            new HashMap<>())
        : new AssignmentSearch(in, s, c, r, seed).solve();
  }

  // Solution conversion and random seeds.

  /** Converts an internal result into the public solution model. */
  private HeuristicSolution materialize(FacilityMemory.Entry e) {
    CoveragePlan plan=e.result.coverage.copy();
    CostBreakdown c=Model.costs(in,e.selection,plan,e.result.routes);
    if(Math.abs(c.total()-e.objective)>1e-6)
      throw new AssertionError("Search/output objective mismatch: "+e.objective+" vs "+c.total());
    return new HeuristicSolution(in.name,List.copyOf(e.selection),plan,
        e.result.routes.stream().map(RoutePlan::copy).toList(),
        new HashMap<>(plan.directAssignment),c.total(),c.facility(),c.route(),
        c.warehouseCoverage(),c.customerCoverage(),true,"",cfg.seed);
  }

  /** Computes a stable hash for an instance name. */
  private static int stableHash(String name) {
    long v = 2166136261L;
    for (byte b : name.getBytes(StandardCharsets.UTF_8))
      v = ((v ^ Byte.toUnsignedInt(b)) * 16777619L) & 0xffffffffL;
    return (int) v;
  }

  /** Derives a reproducible seed for a facility selection. */
  private long selectionSeed(List<Integer> s, int salt) {
    long v = Integer.toUnsignedLong(cfg.seed ^ stableHash(in.name) ^ (salt * 0x9e3779b1));
    for (int f : s) v = ((v << 5) - v + f + 1) & 0xffffffffL;
    return v;
  }

  // Final validation.

  /** Returns a validation error, or {@code null} when valid. */
  public static String validate(Instance in, HeuristicSolution s) {
    if (s.coverage == null) return "missing coverage plan";
    if (!s.feasible) return "solution is marked infeasible";
    Set<Integer> fac = new HashSet<>(s.openFacilities);
    if (fac.isEmpty()) return "no open facility";
    if (fac.size() != s.openFacilities.size()) return "duplicate open facility";
    for (int f : fac) if (f < 0 || f >= in.facilityCount()) return "facility index out of range";
    Set<Integer> selected = s.coverage.selected, routed = new HashSet<>();
    for (int customer : selected)
      if (customer < 0 || customer >= in.customerCount()) return "customer index out of range";
    for (double load : s.coverage.groupLoad.values())
      if (!Double.isFinite(load) || load < 0) return "invalid coverage group load";
    if (!s.coverage.groupLoad.keySet().equals(selected))
      return "coverage group-load keys are inconsistent";
    for (RoutePlan r : s.routes) {
      if (!fac.contains(r.depot)) return "route uses a closed facility";
      if (r.customers.isEmpty()) return "empty route";
      for (int c : r.customers)
        if (c < 0 || c >= in.customerCount()) return "customer index out of range";
      double d = routeDistance(in, r.depot, r.customers);
      if (!closeCost(r.distance, d)) return "stored route distance is stale";
      if (d > in.routeLimit + EPS) return "route length limit exceeded";
      double q = 0;
      for (int c : r.customers) {
        Double load = s.coverage.groupLoad.get(c);
        if (load == null) return "route customer has no coverage load";
        q += load;
      }
      if (!closeCost(r.load, q)) return "stored route load is stale";
      if (q > in.capacity + EPS) return "route capacity exceeded";
      for (int c : r.customers)
        if (!routed.add(c)) return "selected customers are not visited exactly once";
    }
    if (!routed.equals(selected)) return "selected customers are not visited exactly once";
    if(!s.directAssignment.equals(s.coverage.directAssignment))
      return "public and internal direct assignments differ";
    Set<Integer> served=new HashSet<>(s.directAssignment.keySet());
    for(var e:s.directAssignment.entrySet()) {
      int t=e.getKey(), k=e.getValue();
      if(t<0 || t>=in.customerCount() || !fac.contains(k) || !in.facilityCoverMembership[k][t])
        return "illegal direct facility assignment";
      if(selected.contains(t)) return "route customer also warehouse covered";
    }
    for(int t:s.coverage.assignment.keySet()) {
      if(t<0 || t>=in.customerCount()) return "assignment customer index out of range";
      if(!served.add(t)) return "customer has multiple service modes";
    }
    if(served.size()!=in.customerCount()) return "customers are not served exactly once";
    Map<Integer, Double> loads = new HashMap<>();
    for (int c : selected) {
      if (c < 0 || c >= in.customerCount()) return "customer index out of range";
      loads.put(c, in.demand[c]);
    }
    for (var e : s.coverage.assignment.entrySet()) {
      int t = e.getKey(), c = e.getValue();
      if (!selected.contains(c)) return "covered customer points to a non-route customer";
      if (!in.customerCover.get(c).contains(t)) return "customer coverage radius exceeded";
      if (t != c) loads.put(c, loads.get(c) + in.demand[t]);
      else if (!selected.contains(t)) return "self-cover without route visit";
    }
    for (int c : selected)
      if (!Objects.equals(s.coverage.assignment.get(c), c))
        return "route customer is not self-served";
    for (int c : selected) {
      Double stored = s.coverage.groupLoad.get(c);
      if (stored == null || Math.abs(loads.get(c) - stored) > 1e-5)
        return "coverage group load is inconsistent";
    }
    double routeCost = ExperimentParameters.W_R * ExperimentParameters.S
        * s.routes.stream().mapToDouble(r -> routeDistance(in, r.depot, r.customers)).sum();
    double facilityCost = ExperimentParameters.W_F
        * fac.stream().mapToDouble(f -> in.facilityCost[f]).sum();
    double warehouseCoverageCost = ExperimentParameters.W_A * ExperimentParameters.L
        * s.directAssignment.entrySet().stream()
            .mapToDouble(e -> in.facilityDistance[e.getValue()][e.getKey()]).sum();
    double customerCoverageCost = ExperimentParameters.W_C * ExperimentParameters.C
        * s.coverage.assignment.entrySet().stream()
            .mapToDouble(e -> in.customerDistance[e.getValue()][e.getKey()]).sum();
    if (!closeCost(s.routeCost, routeCost)) return "stored route cost is stale";
    if (!closeCost(s.facilityCost, facilityCost)) return "stored facility cost is stale";
    if (!closeCost(s.warehouseCoverageCost, warehouseCoverageCost))
      return "stored warehouse coverage cost is stale";
    if (!closeCost(s.customerCoverageCost, customerCoverageCost))
      return "stored customer coverage cost is stale";
    if (!closeCost(s.objective, facilityCost + routeCost + warehouseCoverageCost + customerCoverageCost))
      return "objective breakdown is inconsistent";
    return null;
  }

  private static boolean closeCost(double stored, double expected) {
    return Double.isFinite(stored) && Double.isFinite(expected)
        && Math.abs(stored - expected) <= 1e-5;
  }
}
