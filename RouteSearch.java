package heuristic;

import static heuristic.Model.*;

import java.util.*;

/**
 * Vehicle-routing layer: constructs and improves routes for fixed open facilities,
 * representative customers, and aggregated loads.
 *
 * <p>This layer only determines each route's depot, customer order, and partitioning. It does
 * not change facility selection, representative customers, or coverage assignment. All operators
 * enforce vehicle-capacity and maximum-route-length constraints.
 *
 * <p>The layer is divided into route construction, destroy-repair, and route local search.
 * Sweep/split constructs routes; ruin with greedy or regret insertion rebuilds them; 2-opt,
 * relocate, and swap perform local search.
 */
public final class RouteSearch {
  private RouteSearch() {}

  /** Route-search parameters: local-search passes, sweep rotations, ruin iterations, and ratios. */
  public static final class Config {
    public int maxVnsPasses = 35, sweepRotations = 12, ruinIterations = 180;
    public boolean enableSwap = true;
    public double ruinMin = .30, ruinMax = .80;
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
    /** Experiment 8.1112: operator sets are explicit. */
    public List<String> destroyOperators = new ArrayList<>(List.of("random", "related"));
    public List<String> repairOperators = new ArrayList<>(List.of("greedy", "regret2"));
    public List<String> localSearchOperators =
        new ArrayList<>(List.of("relocate", "swap", "two_opt"));
    public long experimentSeed = 0L;

    /** Copies the configuration so screening and full evaluation can use separate budgets. */
    public Config copy() {
      Config c = new Config();
      c.maxVnsPasses = maxVnsPasses;
      c.sweepRotations = sweepRotations;
      c.ruinIterations = ruinIterations;
      c.enableSwap = enableSwap;
      c.ruinMin = ruinMin;
      c.ruinMax = ruinMax;
      c.adaptiveWindowLength = adaptiveWindowLength;
      c.adaptiveProbability = adaptiveProbability;
      c.probabilityFluctuationCoefficient = probabilityFluctuationCoefficient;
      c.adaptiveHistoryWeight = adaptiveHistoryWeight;
      c.adaptiveScoreDecay = adaptiveScoreDecay;
      c.adaptiveBaseReward = adaptiveBaseReward;
      c.adaptiveImprovementReward = adaptiveImprovementReward;
      c.adaptiveRejectionPenalty = adaptiveRejectionPenalty;
      c.adaptiveMinimumWeight = adaptiveMinimumWeight;
      c.adaptiveMaximumWeight = adaptiveMaximumWeight;
      c.destroyOperators = new ArrayList<>(destroyOperators);
      c.repairOperators = new ArrayList<>(repairOperators);
      c.localSearchOperators = new ArrayList<>(localSearchOperators);
      c.experimentSeed = experimentSeed;
      return c;
    }
  }

  // ==================== Public route-search entry point ====================

  /**
   * Constructs and improves all routes for a fixed coverage assignment.
   *
   * <p>Uses sweep/split to construct initial routes, then interleaves ruin/recreate with local
   * search.
   *
   * @return routes satisfying all constraints, or {@code null} when capacity or reachability is
   *     infeasible
   */
  public static List<RoutePlan> buildAndImproveRoutes(
      Instance in, List<Integer> fac, Set<Integer> sel, Map<Integer, Double> load, Config cfg) {
    if (cfg == null) cfg = new Config();
    double[] loadValues = new double[in.customerCount()];
    for (int c : new TreeSet<>(sel)) {
      Depot d = bestDepot(in, fac, List.of(c));
      loadValues[c] = load.get(c);
      if (d == null || loadValues[c] > in.capacity + EPS) return null;
    }
    List<RoutePlan> initial = sweepRoutes(in, fac, sel, loadValues, cfg.sweepRotations);
    return initial == null ? null : routeDestroyRepair(in, fac, sel, loadValues, initial, cfg);
  }

  /** Returns the total route distance for the fixed facilities and assignment. */
  public static double cost(List<RoutePlan> routes) {
    double total = 0;
    for (RoutePlan route : routes) total += route.distance;
    return total;
  }

  /** Deep-copies routes so candidate operators cannot mutate the incumbent solution. */
  private static List<RoutePlan> copy(List<RoutePlan> r) {
    List<RoutePlan> x = new ArrayList<>(r.size());
    for (RoutePlan p : r) x.add(p.copy());
    return x;
  }

  // ==================== Sweep/split route construction ====================

  /** Reuses split dynamic-programming arrays to avoid repeated allocations. */
  private static final class SplitWorkspace {
    double[] best = new double[0];
    int[] prev = new int[0];

    void ensure(int size) {
      if (best.length < size) {
        best = new double[size];
        prev = new int[size];
      }
    }
  }

  private static final ThreadLocal<SplitWorkspace> SPLIT_WORKSPACE =
      ThreadLocal.withInitial(SplitWorkspace::new);

  /** Splits a customer order into minimum-cost routes satisfying capacity and length limits. */
  private static List<RoutePlan> splitRoutes(
      Instance in, int depot, List<Integer> order, double[] load) {
    int n = order.size();
    SplitWorkspace workspace = SPLIT_WORKSPACE.get();
    workspace.ensure(n + 1);
    double[] best = workspace.best;
    int[] prev = workspace.prev;
    Arrays.fill(best, 0, n + 1, Double.POSITIVE_INFINITY);
    Arrays.fill(prev, 0, n + 1, -1);
    best[0] = 0;
    for (int start = 0; start < n; start++) {
      if (!Double.isFinite(best[start])) continue;
      double q = 0, forward = in.facilityDistance[depot][order.get(start)];
      for (int end = start + 1; end <= n; end++) {
        int last = order.get(end - 1);
        q += load[last];
        if (q > in.capacity + EPS) break;
        if (end > start + 1) forward += in.customerDistance[order.get(end - 2)][last];
        double distance = forward + in.facilityDistance[depot][last];
        if (distance > in.routeLimit + EPS) break;
        if (best[start] + distance + EPS < best[end]) {
          best[end] = best[start] + distance;
          prev[end] = start;
        }
      }
    }
    if (!Double.isFinite(best[n])) return null;
    List<RoutePlan> routes = new ArrayList<>();
    for (int end = n; end > 0; ) {
      int start = prev[end];
      if (start < 0) return null;
      List<Integer> customers = new ArrayList<>(order.subList(start, end));
      double q = sumLoads(customers, load);
      routes.add(new RoutePlan(depot, customers, q, routeDistance(in, depot, customers)));
      end = start;
    }
    Collections.reverse(routes);
    return routes;
  }

  /** Builds sweep routes by polar-angle sorting, rotating, and applying split. */
  private static List<RoutePlan> sweepRoutes(
      Instance in, List<Integer> facilities, Set<Integer> selected,
      double[] load, int rotations) {
    Map<Integer, List<Integer>> byFacility = new LinkedHashMap<>();
    for (int f : facilities) byFacility.put(f, new ArrayList<>());
    for (int customer : selected) {
      int bestFacility = facilities.get(0);
      for (int f : facilities)
        if (in.facilityDistance[f][customer] < in.facilityDistance[bestFacility][customer])
          bestFacility = f;
      byFacility.get(bestFacility).add(customer);
    }
    List<RoutePlan> all = new ArrayList<>();
    for (var entry : byFacility.entrySet()) {
      int depot = entry.getKey();
      List<Integer> customers = entry.getValue();
      if (customers.isEmpty()) continue;
      customers.sort(Comparator.comparingDouble(c -> in.polarAngle[depot][c]));
      Set<Integer> shifts = new TreeSet<>();
      if (customers.size() <= rotations)
        for (int i = 0; i < customers.size(); i++) shifts.add(i);
      else
        for (int i = 0; i < rotations; i++)
          shifts.add(pythonRound((double) i * customers.size() / rotations) % customers.size());
      List<RoutePlan> bestRoutes = null;
      double bestCost = Double.POSITIVE_INFINITY;
      for (int shift : shifts) {
        List<Integer> rotated = new ArrayList<>();
        rotated.addAll(customers.subList(shift, customers.size()));
        rotated.addAll(customers.subList(0, shift));
        for (int direction = 0; direction < 2; direction++) {
          List<Integer> order = new ArrayList<>(rotated);
          if (direction == 1) Collections.reverse(order);
          List<RoutePlan> routes = splitRoutes(in, depot, order, load);
          if (routes != null) {
            double candidateCost = cost(routes);
            if (candidateCost < bestCost) {
              bestCost = candidateCost;
              bestRoutes = routes;
            }
          }
        }
      }
      if (bestRoutes == null) return null;
      all.addAll(bestRoutes);
    }
    return all;
  }

  // ==================== Incremental-evaluation records and workspace ====================

  /** Best feasible depot and round-trip distance for a customer sequence. */
  private record Depot(int id, double distance) {}

  /** Regret-insertion candidate with delta, route position, depot, and inserted customer. */
  private record Insertion(
      double delta, int index, int position, int depot, double distance, int customer) {}

  // ==================== O(1) distance deltas and depot selection ====================

  /** Computes route distances for the same customer sequence at every open facility. */
  private static double[] routeDistances(Instance in, List<Integer> fac, List<Integer> seq) {
    double[] distances = new double[fac.size()];
    for (int i = 0; i < fac.size(); i++) distances[i] = routeDistance(in, fac.get(i), seq);
    return distances;
  }

  /** Incrementally finds the best feasible depot after removing one customer. */
  private static Depot bestDepotWithout(
      Instance in, List<Integer> fac, List<Integer> seq, int removed, double[] base) {
    if (seq.size() == 1) return null;
    int customer = seq.get(removed),
        previous = removed == 0 ? -1 : seq.get(removed - 1),
        next = removed + 1 == seq.size() ? -1 : seq.get(removed + 1),
        bestId = -1;
    double bestDistance = 0;
    for (int i = 0; i < fac.size(); i++) {
      int depot = fac.get(i);
      double
          before =
              previous < 0
                  ? in.facilityDistance[depot][customer]
                  : in.customerDistance[previous][customer],
          after =
              next < 0 ? in.facilityDistance[depot][customer] : in.customerDistance[customer][next],
          bridge =
              previous < 0
                  ? in.facilityDistance[depot][next]
                  : next < 0
                      ? in.facilityDistance[depot][previous]
                      : in.customerDistance[previous][next],
          distance = base[i] - before - after + bridge;
      if (distance <= in.routeLimit + EPS && (bestId < 0 || distance < bestDistance)) {
        bestId = depot;
        bestDistance = distance;
      }
    }
    return bestId < 0 ? null : new Depot(bestId, bestDistance);
  }

  /** Incrementally finds the best feasible depot after inserting a customer. */
  private static Depot bestDepotWithInsertion(
      Instance in,
      List<Integer> fac,
      List<Integer> seq,
      int position,
      int inserted,
      double[] base) {
    int previous = position == 0 ? -1 : seq.get(position - 1),
        next = position == seq.size() ? -1 : seq.get(position),
        bestId = -1;
    double bestDistance = 0;
    for (int i = 0; i < fac.size(); i++) {
      int depot = fac.get(i);
      double
          before =
              previous < 0
                  ? in.facilityDistance[depot][inserted]
                  : in.customerDistance[previous][inserted],
          after =
              next < 0 ? in.facilityDistance[depot][inserted] : in.customerDistance[inserted][next],
          removed =
              previous < 0
                  ? in.facilityDistance[depot][next]
                  : next < 0
                      ? in.facilityDistance[depot][previous]
                      : in.customerDistance[previous][next],
          distance = base[i] + before + after - removed;
      if (distance <= in.routeLimit + EPS && (bestId < 0 || distance < bestDistance)) {
        bestId = depot;
        bestDistance = distance;
      }
    }
    return bestId < 0 ? null : new Depot(bestId, bestDistance);
  }

  /** Incrementally finds the best feasible depot after replacing a customer. */
  private static Depot bestDepotWithReplacement(
      Instance in,
      List<Integer> fac,
      List<Integer> seq,
      int position,
      int replacement,
      double[] base) {
    int old = seq.get(position),
        previous = position == 0 ? -1 : seq.get(position - 1),
        next = position + 1 == seq.size() ? -1 : seq.get(position + 1),
        bestId = -1;
    double bestDistance = 0;
    for (int i = 0; i < fac.size(); i++) {
      int depot = fac.get(i);
      double
          oldBefore =
              previous < 0 ? in.facilityDistance[depot][old] : in.customerDistance[previous][old],
          oldAfter = next < 0 ? in.facilityDistance[depot][old] : in.customerDistance[old][next],
          newBefore =
              previous < 0
                  ? in.facilityDistance[depot][replacement]
                  : in.customerDistance[previous][replacement],
          newAfter =
              next < 0
                  ? in.facilityDistance[depot][replacement]
                  : in.customerDistance[replacement][next],
          distance = base[i] - oldBefore - oldAfter + newBefore + newAfter;
      if (distance <= in.routeLimit + EPS && (bestId < 0 || distance < bestDistance)) {
        bestId = depot;
        bestDistance = distance;
      }
    }
    return bestId < 0 ? null : new Depot(bestId, bestDistance);
  }

  /** Scans all open facilities and returns the best feasible depot for a customer sequence. */
  private static Depot bestDepot(Instance in, List<Integer> fac, List<Integer> seq) {
    int bestId = -1;
    double bestDistance = 0;
    for (int d : fac) {
      double x = routeDistance(in, d, seq);
      if (x <= in.routeLimit + EPS && (bestId < 0 || x < bestDistance)) {
        bestId = d;
        bestDistance = x;
      }
    }
    return bestId < 0 ? null : new Depot(bestId, bestDistance);
  }

  // ==================== Destroy operators ====================

  /** Random destroy: removes the requested number of representative customers uniformly. */
  private static void destroyRandom(
      List<Integer> sortedAll, int count, PythonRandom rng, Set<Integer> removed) {
    removed.addAll(rng.sample(sortedAll, count));
  }

  /** Shallow related destroy: removes customers near a randomly selected anchor. */
  private static void destroyRelatedArea(
      Instance in, List<Integer> sortedAll, int count, PythonRandom rng, Set<Integer> removed) {
    int anchor = rng.choice(sortedAll);
    List<Integer> related = new ArrayList<>(sortedAll);
    related.sort(Comparator.comparingDouble(customer -> in.customerDistance[anchor][customer]));
    removed.addAll(related.subList(0, count));
  }

  /** Medium related destroy: removes contiguous customer strings from existing routes. */
  private static void destroyRelatedString(
      List<RoutePlan> current,
      Set<Integer> all,
      int count,
      PythonRandom rng,
      Set<Integer> removed) {
    int attempts = 0;
    while (removed.size() < count && attempts++ < current.size() * 4 + 10) {
      RoutePlan route = rng.choice(current);
      List<Integer> available =
          route.customers.stream().filter(customer -> !removed.contains(customer)).toList();
      if (available.isEmpty()) continue;
      int start = rng.randrange(available.size());
      int maximum = Math.min(available.size(), count - removed.size());
      int length = 1 + rng.randrange(maximum);
      for (int offset = 0; offset < length; offset++)
        removed.add(available.get((start + offset) % available.size()));
    }
    if (removed.size() < count) {
      List<Integer> remainder = new ArrayList<>(all);
      remainder.removeAll(removed);
      rng.shuffle(remainder);
      removed.addAll(remainder.subList(0, count - removed.size()));
    }
  }

  /** Strong related destroy: removes one complete route and randomly fills any shortfall. */
  private static void destroyRelatedRoute(
      List<RoutePlan> current,
      Set<Integer> all,
      int count,
      PythonRandom rng,
      Set<Integer> removed) {
    RoutePlan route = rng.choice(current);
    removed.addAll(route.customers);
    List<Integer> remainder = new ArrayList<>(all);
    remainder.removeAll(removed);
    rng.shuffle(remainder);
    removed.addAll(
        remainder.subList(0, Math.max(0, Math.min(remainder.size(), count - removed.size()))));
  }

  private static Set<Integer> applyDestroy(
      String operator,
      Instance in,
      List<RoutePlan> current,
      Set<Integer> all,
      List<Integer> sortedAll,
      int count,
      PythonRandom rng) {
    Set<Integer> removed = new HashSet<>();
    switch (operator) {
      case "related" -> {
        switch (rng.randrange(3)) {
          case 0 -> destroyRelatedArea(in, sortedAll, count, rng, removed);
          case 1 -> destroyRelatedString(current, all, count, rng, removed);
          default -> destroyRelatedRoute(current, all, count, rng, removed);
        }
      }
      case "random" -> destroyRandom(sortedAll, count, rng, removed);
      default -> throw new IllegalArgumentException("Unknown destroy operator: " + operator);
    }
    return removed;
  }

  // ==================== Repair operators ====================

  /** Finds the two best existing-route or singleton insertion options for an unserved customer. */
  private static List<Insertion> options(
      Instance in,
      List<Integer> fac,
      double[] load,
      List<RoutePlan> rs,
      double[][] base,
      int c) {
    Insertion first = null, second = null;
    double customerLoad = load[c];
    for (int i = 0; i < rs.size(); i++) {
      RoutePlan r = rs.get(i);
      if (r.load + customerLoad > in.capacity + EPS) continue;
      List<Integer> seq = r.customers;
      for (int p = 0; p <= seq.size(); p++) {
        int previous = p == 0 ? -1 : seq.get(p - 1),
            next = p == seq.size() ? -1 : seq.get(p),
            bestDepot = -1;
        double bestDistance = 0;
        for (int f = 0; f < fac.size(); f++) {
          int depot = fac.get(f);
          double
              before =
                  previous < 0 ? in.facilityDistance[depot][c] : in.customerDistance[previous][c],
              after = next < 0 ? in.facilityDistance[depot][c] : in.customerDistance[c][next],
              removed =
                  previous < 0
                      ? in.facilityDistance[depot][next]
                      : next < 0
                          ? in.facilityDistance[depot][previous]
                          : in.customerDistance[previous][next],
              distance = base[i][f] + before + after - removed;
          if (distance <= in.routeLimit + EPS && (bestDepot < 0 || distance < bestDistance)) {
            bestDepot = depot;
            bestDistance = distance;
          }
        }
        if (bestDepot < 0) continue;
        double delta = bestDistance - r.distance;
        if (first == null || delta < first.delta) {
          second = first;
          first = new Insertion(delta, i, p, bestDepot, bestDistance, c);
        } else if (second == null || delta < second.delta)
          second = new Insertion(delta, i, p, bestDepot, bestDistance, c);
      }
    }
    int singletonDepot = -1;
    double singletonDistance = 0;
    for (int depot : fac) {
      double distance = in.facilityDistance[depot][c] + in.facilityDistance[depot][c];
      if (distance <= in.routeLimit + EPS && (singletonDepot < 0 || distance < singletonDistance)) {
        singletonDepot = depot;
        singletonDistance = distance;
      }
    }
    if (singletonDepot >= 0) {
      if (first == null || singletonDistance < first.delta) {
        second = first;
        first =
            new Insertion(singletonDistance, rs.size(), 0, singletonDepot, singletonDistance, c);
      } else if (second == null || singletonDistance < second.delta)
        second =
            new Insertion(singletonDistance, rs.size(), 0, singletonDepot, singletonDistance, c);
    }
    if (first == null) return List.of();
    return second == null ? List.of(first) : List.of(first, second);
  }

  /** Regret-2 insertion prioritizes the customer with the largest second-best penalty. */
  private static List<RoutePlan> regret(
      Instance in,
      List<Integer> fac,
      double[] load,
      List<RoutePlan> rs,
      Set<Integer> removed,
      PythonRandom rng) {
    Set<Integer> pending = new HashSet<>(removed);
    while (!pending.isEmpty()) {
      double[][] base = new double[rs.size()][];
      for (int i = 0; i < rs.size(); i++) base[i] = routeDistances(in, fac, rs.get(i).customers);
      int chosen = -1;
      Insertion ins = null;
      double bestReg = -1, bestFirst = 0, bestRnd = 0;
      for (int c : pending) {
        List<Insertion> o = options(in, fac, load, rs, base, c);
        if (o.isEmpty()) return null;
        double reg = o.size() > 1 ? o.get(1).delta - o.get(0).delta : 1e6, r = rng.random();
        if (reg > bestReg
            || (reg == bestReg
                && (-o.get(0).delta > bestFirst
                    || (-o.get(0).delta == bestFirst && r > bestRnd)))) {
          bestReg = reg;
          bestFirst = -o.get(0).delta;
          bestRnd = r;
          chosen = c;
          ins = o.get(0);
        }
      }
      if (ins.index == rs.size())
        rs.add(
            new RoutePlan(
                ins.depot,
                new ArrayList<>(List.of(ins.customer)),
                load[ins.customer],
                ins.distance));
      else {
        RoutePlan target = rs.get(ins.index);
        List<Integer> customers = new ArrayList<>(target.customers);
        customers.add(ins.position, ins.customer);
        rs.set(
            ins.index,
            new RoutePlan(
                ins.depot, customers, target.load + load[ins.customer], ins.distance));
      }
      pending.remove(chosen);
    }
    return rs;
  }

  /** Greedy insertion repeatedly applies the globally cheapest feasible distance increase. */
  private static List<RoutePlan> greedyInsertion(
      Instance in,
      List<Integer> fac,
      double[] load,
      List<RoutePlan> rs,
      Set<Integer> removed,
      PythonRandom rng) {
    Set<Integer> pending = new HashSet<>(removed);
    while (!pending.isEmpty()) {
      double[][] base = new double[rs.size()][];
      for (int i = 0; i < rs.size(); i++) base[i] = routeDistances(in, fac, rs.get(i).customers);
      int chosen = -1;
      Insertion ins = null;
      double bestDelta = Double.POSITIVE_INFINITY, bestRnd = 0;
      for (int customer : pending) {
        List<Insertion> candidates = options(in, fac, load, rs, base, customer);
        if (candidates.isEmpty()) return null;
        Insertion first = candidates.get(0);
        double tie = rng.random();
        if (first.delta < bestDelta
            || (first.delta == bestDelta && tie > bestRnd)) {
          chosen = customer;
          ins = first;
          bestDelta = first.delta;
          bestRnd = tie;
        }
      }
      if (ins.index == rs.size())
        rs.add(
            new RoutePlan(
                ins.depot,
                new ArrayList<>(List.of(ins.customer)),
                load[ins.customer],
                ins.distance));
      else {
        RoutePlan target = rs.get(ins.index);
        List<Integer> customers = new ArrayList<>(target.customers);
        customers.add(ins.position, ins.customer);
        rs.set(
            ins.index,
            new RoutePlan(
                ins.depot, customers, target.load + load[ins.customer], ins.distance));
      }
      pending.remove(chosen);
    }
    return rs;
  }

  private static List<RoutePlan> applyRepair(
      String operator,
      Instance in,
      List<Integer> fac,
      double[] load,
      List<RoutePlan> partial,
      Set<Integer> removed,
      PythonRandom rng) {
    return switch (operator) {
      case "greedy" -> greedyInsertion(in, fac, load, partial, removed, rng);
      case "regret2" -> regret(in, fac, load, partial, removed, rng);
      default -> throw new IllegalArgumentException("Unknown repair operator: " + operator);
    };
  }

  // ==================== Local-search operators ====================

  /**
   * Reverses customer subsequences and repeatedly accepts the best improving 2-opt move.
   *
   * <p>This operation changes neither route load nor depot; it only improves visit order.
   */
  private static boolean twoOpt(Instance in, RoutePlan r) {
    List<Integer> s = r.customers;
    if (s.size() < 3) return false;
    boolean improved = false;
    while (true) {
      double bd = -EPS;
      int bl = -1, br = -1;
      for (int l = 0; l < s.size() - 1; l++) {
        int a = l == 0 ? -1 : s.get(l - 1), b = s.get(l);
        for (int rr = l + 1; rr < s.size(); rr++) {
          int c = s.get(rr), d = rr + 1 == s.size() ? -1 : s.get(rr + 1);
          double ab = a < 0 ? in.facilityDistance[r.depot][b] : in.customerDistance[a][b];
          double cd = d < 0 ? in.facilityDistance[r.depot][c] : in.customerDistance[c][d];
          double ac = a < 0 ? in.facilityDistance[r.depot][c] : in.customerDistance[a][c];
          double bb = d < 0 ? in.facilityDistance[r.depot][b] : in.customerDistance[b][d];
          double delta = ac + bb - ab - cd;
          if (delta < bd) {
            bd = delta;
            bl = l;
            br = rr;
          }
        }
      }
      if (bl < 0) break;
      Collections.reverse(s.subList(bl, br + 1));
      r.distance += bd;
      improved = true;
    }
    r.distance = routeDistance(in, r.depot, s);
    return improved;
  }


  /** Moves one customer between routes and reselects the best depot for both routes. */
  private static boolean relocate(
      Instance in, List<Integer> fac, double[] load, List<RoutePlan> rs) {
    double best = -EPS;
    int bs = -1, bt = -1, bp = -1, bq = -1, brd = -1, bed = -1;
    double brx = 0, bex = 0;
    boolean empty = false;
    double[][] base = new double[rs.size()][];
    for (int i = 0; i < rs.size(); i++) base[i] = routeDistances(in, fac, rs.get(i).customers);
    for (int si = 0; si < rs.size(); si++) {
      RoutePlan src = rs.get(si);
      for (int p = 0; p < src.customers.size(); p++) {
        int c = src.customers.get(p);
        Depot reduced = bestDepotWithout(in, fac, src.customers, p, base[si]);
        double rc = 0;
        if (src.customers.size() > 1) {
          if (reduced == null) continue;
          rc = reduced.distance;
        }
        for (int ti = 0; ti < rs.size(); ti++) {
          if (ti == si) continue;
          RoutePlan tar = rs.get(ti);
          if (tar.load + load[c] > in.capacity + EPS) continue;
          for (int q = 0; q <= tar.customers.size(); q++) {
            Depot expanded = bestDepotWithInsertion(in, fac, tar.customers, q, c, base[ti]);
            if (expanded == null) continue;
            double delta = rc + expanded.distance - src.distance - tar.distance;
            if (delta < best) {
              best = delta;
              bs = si;
              bt = ti;
              bp = p;
              bq = q;
              empty = src.customers.size() == 1;
              brd = empty ? -1 : reduced.id;
              brx = empty ? 0 : reduced.distance;
              bed = expanded.id;
              bex = expanded.distance;
            }
          }
        }
      }
    }
    if (bs < 0) return false;
    RoutePlan src = rs.get(bs), tar = rs.get(bt);
    int customer = src.customers.get(bp);
    List<Integer> expandedCustomers = new ArrayList<>(tar.customers);
    expandedCustomers.add(bq, customer);
    RoutePlan expanded = new RoutePlan(bed, expandedCustomers, tar.load + load[customer], bex);
    if (empty) {
      rs.set(bt, expanded);
      rs.remove(bs);
    } else {
      List<Integer> reducedCustomers = new ArrayList<>(src.customers);
      reducedCustomers.remove(bp);
      RoutePlan reduced = new RoutePlan(brd, reducedCustomers, src.load - load[customer], brx);
      rs.set(bs, reduced);
      rs.set(bt, expanded);
    }
    return true;
  }

  /** Swaps customers between two routes while checking capacity, length, and depot choice. */
  private static boolean swap(
      Instance in, List<Integer> fac, double[] load, List<RoutePlan> rs) {
    double best = -EPS;
    int bi = -1, bj = -1, bp = -1, bq = -1, bad = -1, bbd = -1;
    double bax = 0, bbx = 0, bla = 0, blb = 0;
    double[][] base = new double[rs.size()][];
    for (int i = 0; i < rs.size(); i++) base[i] = routeDistances(in, fac, rs.get(i).customers);
    for (int i = 0; i < rs.size(); i++)
      for (int j = i + 1; j < rs.size(); j++) {
        RoutePlan a = rs.get(i), b = rs.get(j);
        for (int p = 0; p < a.customers.size(); p++)
          for (int q = 0; q < b.customers.size(); q++) {
            int x = a.customers.get(p), y = b.customers.get(q);
            double la = a.load - load[x] + load[y], lb = b.load - load[y] + load[x];
            if (la > in.capacity + EPS || lb > in.capacity + EPS) continue;
            Depot da = bestDepotWithReplacement(in, fac, a.customers, p, y, base[i]),
                db = bestDepotWithReplacement(in, fac, b.customers, q, x, base[j]);
            if (da == null || db == null) continue;
            double d = da.distance + db.distance - a.distance - b.distance;
            if (d < best) {
              best = d;
              bi = i;
              bj = j;
              bp = p;
              bq = q;
              bad = da.id;
              bbd = db.id;
              bax = da.distance;
              bbx = db.distance;
              bla = la;
              blb = lb;
            }
          }
      }
    if (bi < 0) return false;
    RoutePlan a = rs.get(bi), b = rs.get(bj);
    int x = a.customers.get(bp), y = b.customers.get(bq);
    List<Integer> as = new ArrayList<>(a.customers), bs = new ArrayList<>(b.customers);
    as.set(bp, y);
    bs.set(bq, x);
    rs.set(bi, new RoutePlan(bad, as, bla, bax));
    rs.set(bj, new RoutePlan(bbd, bs, blb, bbx));
    return true;
  }

  // ==================== Complete-route local search ====================

  /**
   * Runs every route local-search operator once per pass in a randomized order. This changes only
   * scheduling, not the relocate, swap, or 2-opt implementations or the pass limit.
   */
  private static List<RoutePlan> bundledRandomLocalSearch(
      Instance in,
      List<Integer> fac,
      double[] load,
      List<RoutePlan> routes,
      Config cfg,
      PythonRandom rng,
      int passLimit) {
    List<RoutePlan> rs = copy(routes);
    for (int pass = 0; pass < Math.max(0, passLimit); pass++) {
      double before = cost(rs);
      List<String> operators = new ArrayList<>(cfg.localSearchOperators);
      if (!cfg.enableSwap) operators.remove("swap");
      rng.shuffle(operators);
      for (String operator : operators) {
        switch (operator) {
          case "relocate" -> relocate(in, fac, load, rs);
          case "swap" -> swap(in, fac, load, rs);
          case "two_opt" -> {
            for (RoutePlan route : rs) {
              twoOpt(in, route);
              Depot depot = bestDepot(in, fac, route.customers);
              if (depot != null) {
                route.depot = depot.id;
                route.distance = depot.distance;
              }
            }
          }
          default -> throw new AssertionError("unknown route local-search operator " + operator);
        }
      }
      if (cost(rs) + EPS >= before) break;
    }
    rs.sort(
        Comparator.comparingInt((RoutePlan route) -> route.depot)
            .thenComparingInt(route -> route.customers.get(0)));
    return rs;
  }

  // ==================== Destroy-repair search flow ====================

  /** Removes customers, rebuilds routes with a repair operator, then performs local search. */
  private static List<RoutePlan> routeDestroyRepair(
      Instance in,
      List<Integer> fac,
      Set<Integer> selected,
      double[] load,
      List<RoutePlan> initial,
      Config cfg) {
    long seed = 0x811c9dc5L;
    for (int c : new TreeSet<>(selected))
      seed = ((seed ^ (c + 1)) * 0x01000193L) & 0xffffffffL;
    seed ^= cfg.experimentSeed;
    PythonRandom rng = new PythonRandom(seed);
    if (cfg.ruinIterations <= 0 || initial.stream().mapToInt(r -> r.customers.size()).sum() < 4)
      return bundledRandomLocalSearch(
          in, fac, load, initial, cfg, rng, Math.min(3, cfg.maxVnsPasses));
    List<RoutePlan> cur = copy(initial), best = copy(initial);
    double cc = cost(cur), bc = cc;
    Set<Integer> all = new HashSet<>(selected);
    List<Integer> sortedAll = new ArrayList<>(new TreeSet<>(all));
    if (cfg.destroyOperators.isEmpty() || cfg.repairOperators.isEmpty())
      throw new IllegalArgumentException("Destroy and repair operator sets must be nonempty");
    AdaptiveOperatorSelection.State adaptive =
        new AdaptiveOperatorSelection.State(
            cfg.destroyOperators, cfg.repairOperators, adaptiveParameters(cfg));
    for (int it = 0; it < cfg.ruinIterations; it++) {
      double frac = rng.uniform(cfg.ruinMin, cfg.ruinMax);
      int count = Math.max(2, Math.min(all.size() - 1, pythonRound(all.size() * frac)));
      AdaptiveOperatorSelection.Selection selection = adaptive.select(rng, it);
      Set<Integer> rem = applyDestroy(selection.destroy(), in, cur, all, sortedAll, count, rng);
      List<RoutePlan> partial = new ArrayList<>();
      for (RoutePlan r : cur) {
        List<Integer> s = r.customers.stream().filter(c -> !rem.contains(c)).toList();
        if (s.isEmpty()) continue;
        Depot d = bestDepot(in, fac, s);
        if (d != null)
          partial.add(
              new RoutePlan(
                  d.id, new ArrayList<>(s), sumLoads(s, load), d.distance));
      }
      List<RoutePlan> cand =
          applyRepair(selection.repair(), in, fac, load, partial, rem, rng);
      boolean improved = false;
      double improvement = 0;
      if (cand != null) {
        cand = bundledRandomLocalSearch(in, fac, load, cand, cfg, rng, cfg.maxVnsPasses);
        double nc = cost(cand);
        improved = nc + EPS < cc;
        if (improved) {
          improvement = cc - nc;
          cur = copy(cand);
          cc = nc;
        }
        if (nc + EPS < bc) {
          best = copy(cand);
          bc = nc;
        }
      }
      adaptive.record(selection, improved, improvement, it);
    }
    // Each repair already receives local search; converge once more from the best solution.
    return bundledRandomLocalSearch(in, fac, load, best, cfg, rng, cfg.maxVnsPasses);
  }

  private static AdaptiveOperatorSelection.Parameters adaptiveParameters(Config cfg) {
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

  private static double sumLoads(Collection<Integer> customers, double[] load) {
    double total = 0;
    for (int customer : customers) total += load[customer];
    return total;
  }


}
