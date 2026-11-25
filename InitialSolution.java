package heuristic;

import static heuristic.AssignmentSearch.*;
import static heuristic.Model.*;
import static heuristic.RouteSearch.cost;

import java.util.*;

/**
 * Greedy initial-solution construction: first generates multiple candidate facility selections,
 * then jointly constructs customer coverage and vehicle routes for each fixed selection.
 */
public final class InitialSolution {
  private InitialSolution() {}

  /** Scores closed candidate facilities by coverage gain, route reachability, and opening cost. */
  static double[] facilityOpenScores(
      FacilitySearch search, Set<Integer> selected, List<Integer> facilities) {
    Instance in = search.instance;
    Scores.FuzzyScoreConfig score = search.config.score;
    int n = in.customerCount();
    Arrays.fill(search.openCovered, false);
    Arrays.fill(search.openReached, false);
    for (int f : selected) {
      for (int customer : in.facilityCover.get(f)) search.openCovered[customer] = true;
      for (int customer = 0; customer < n; customer++)
        if (in.routeReachable[f][customer]) search.openReached[customer] = true;
    }
    for (int f : facilities) {
      int directGainCount = 0, reachGainCount = 0;
      for (int customer : in.facilityCover.get(f))
        if (!search.openCovered[customer]) directGainCount++;
      for (int customer = 0; customer < n; customer++)
        if (!search.openReached[customer] && in.routeReachable[f][customer]) reachGainCount++;
      double directGain = (double) directGainCount / n,
          reachGain = (double) reachGainCount / n,
          boundedCost =
              in.facilityCost[f]
                  / Math.max(EPS, in.facilityCost[f] + search.meanFacilityCost);
      search.scoreBuffer[f] =
          score.facilityCoverGain * directGain
              + score.facilityReachGain * reachGain
              - score.facilityCostPenalty * boundedCost;
    }
    return search.scoreBuffer;
  }

  /** Builds candidate facility selections from several low-cost starts, adding the best-scored facility each time. */
  static List<List<Integer>> facilityGreedyChains(FacilitySearch search) {
    Instance in = search.instance;
    Solver.Config cfg = search.config;
    List<List<Integer>> out = new ArrayList<>();
    List<Integer> first = new ArrayList<>();
    for (int f = 0; f < in.facilityCount(); f++) first.add(f);
    first.sort(Comparator.comparingDouble(f -> in.facilityCost[f]));
    for (int st : first.subList(0, Math.min(cfg.initialGreedyStarts, first.size()))) {
      Set<Integer> selected = new HashSet<>();
      selected.add(st);
      out.add(canonicalFacilities(selected));
      while (selected.size() < Math.min(cfg.maximumOpenFacilities, in.facilityCount())) {
        List<Integer> candidates = new ArrayList<>();
        for (int f = 0; f < in.facilityCount(); f++) if (!selected.contains(f)) candidates.add(f);
        double[] scores = facilityOpenScores(search, selected, candidates);
        int best = -1;
        double bestScore = -Double.MAX_VALUE;
        for (int f : candidates)
          if (scores[f] > bestScore) {
            bestScore = scores[f];
            best = f;
          }
        selected.add(best);
        out.add(canonicalFacilities(selected));
      }
    }
    return out;
  }

  /**
   * Greedily constructs customer coverage and vehicle routes for a fixed facility selection.
   *
   * <p>Restricted customers that can neither be covered directly by a facility nor visited by a
   * vehicle are assigned first so that they can depend on reachable representatives. The remaining
   * customers are processed in nearest-neighbor order by comparing the marginal costs of four
   * choices: direct facility coverage, coverage by an existing representative, insertion as a new
   * representative into an existing route, and creation of a new route for the representative.
   * Each decision updates the coverage assignment, representative-group load, and vehicle routes
   * together, producing a complete initial feasible solution.
   */
  static Result jointAssignment(AssignmentSearch search) {
    // Initialize pending customers, representatives, both coverage maps, group loads, and routes.
    Set<Integer> pending = new TreeSet<>(search.all), selected = new HashSet<>();
    Map<Integer, Integer> assignment = new HashMap<>(), depots = new HashMap<>();
    Map<Integer, Double> groupLoad = new HashMap<>();
    Map<Integer, RoutePlan> representativeRoute = new HashMap<>();
    List<RoutePlan> routes = new ArrayList<>();

    // Stage 1: handle restricted customers first. Try an existing representative; otherwise choose
    // a new representative and either insert it into a route or create a new route.
    List<Integer> forcedIndirect =
        pending.stream()
            .filter(t -> search.nearestWarehouse[t] < 0 && !search.reachable.contains(t))
            .sorted(
                Comparator.comparingInt((Integer t) -> search.candidates.get(t).size())
                    .thenComparingInt(t -> t))
            .toList();
    for (int target : forcedIndirect) {
      if (!pending.contains(target)) continue;

      int existing = nearestExistingRepresentative(search, target, selected, representativeRoute);
      if (existing >= 0) {
        assignToRepresentative(search, target, existing, assignment, groupLoad,
            representativeRoute);
        pending.remove(target);
        continue;
      }

      int representative = nearestNewRepresentative(search, target, selected, assignment);
      if (representative < 0) return null;
      double load = search.in.demand[representative] + search.in.demand[target];
      PathOptions options = fastPathOptions(search, routes, representative, load);
      GreedyInsertion insertion = cheaper(options.existing, options.fresh);
      if (insertion == null) return null;

      RoutePlan route = applyInsertion(routes, representative, load, insertion);
      selected.add(representative);
      assignment.put(representative, representative);
      assignment.put(target, representative);
      groupLoad.put(representative, load);
      representativeRoute.put(representative, route);
      pending.remove(representative);
      pending.remove(target);
    }

    // Stage 2: choose the first customer randomly, then greedily follow nearest-neighbor order.
    if (!pending.isEmpty()) {
      int current = search.rng.choice(new ArrayList<>(pending));
      while (!pending.isEmpty()) {
        int kind = 0, provider = -1;
        double bestDelta = Double.POSITIVE_INFINITY;
        GreedyInsertion chosenInsertion = null;

        // Choice 1: direct service by the nearest covering facility.
        if (search.nearestWarehouse[current] >= 0) {
          provider = search.nearestWarehouse[current];
          bestDelta = search.serviceLinkCost(current, -1 - provider);
          kind = 1;
        }

        // Choice 2: coverage by the nearest capacity-feasible existing representative.
        int representative =
            nearestExistingRepresentative(search, current, selected, representativeRoute);
        if (representative >= 0) {
          double delta = search.serviceLinkCost(current, representative);
          if (delta < bestDelta) {
            bestDelta = delta;
            provider = representative;
            kind = 2;
          }
        }

        // Choices 3 and 4: make the customer a representative and insert or create a route.
        double ownLoad = search.in.demand[current];
        if (ownLoad <= search.in.capacity + EPS) {
          PathOptions options = fastPathOptions(search, routes, current, ownLoad);
          if (options.existing != null) {
            double delta = weightedRouteDelta(options.existing);
            if (delta < bestDelta) {
              bestDelta = delta;
              chosenInsertion = options.existing;
              kind = 3;
            }
          }
          if (options.fresh != null) {
            double delta = weightedRouteDelta(options.fresh);
            if (delta < bestDelta) {
              bestDelta = delta;
              chosenInsertion = options.fresh;
              kind = 4;
            }
          }
        }

        if (kind == 0) return null;
        if (kind == 1) {
          depots.put(current, provider);
        } else if (kind == 2) {
          assignToRepresentative(search, current, provider, assignment, groupLoad,
              representativeRoute);
        } else {
          RoutePlan route = applyInsertion(routes, current, ownLoad, chosenInsertion);
          selected.add(current);
          assignment.put(current, current);
          groupLoad.put(current, ownLoad);
          representativeRoute.put(current, route);
        }

        int previous = current;
        pending.remove(current);
        if (!pending.isEmpty()) current = nearestPending(search.in, previous, pending);
      }
    }
    // Combine representative coverage, direct facility coverage, and routes into the joint result.
    return search.result(
        new CoveragePlan(selected, assignment, groupLoad, depots), routes, true);
  }

  private record GreedyInsertion(
      double delta, int routeIndex, int position, int depot, double distance) {}

  private record PathOptions(GreedyInsertion existing, GreedyInsertion fresh) {}

  /**
   * Quickly compares insertion before or after the nearest node of an existing route with creating
   * a new route from a feasible open facility.
   */
  private static PathOptions fastPathOptions(
      AssignmentSearch search, List<RoutePlan> routes, int customer, double load) {
    GreedyInsertion existing = null;
    for (int i = 0; i < routes.size(); i++) {
      RoutePlan route = routes.get(i);
      if (route.load + load > search.in.capacity + EPS) continue;
      int nearest = 0;
      for (int p = 1; p < route.customers.size(); p++)
        if (search.in.customerDistance[customer][route.customers.get(p)]
            < search.in.customerDistance[customer][route.customers.get(nearest)]) nearest = p;
      for (int position : new int[] {nearest, nearest + 1}) {
        List<Integer> sequence = new ArrayList<>(route.customers);
        sequence.add(position, customer);
        double distance = routeDistance(search.in, route.depot, sequence);
        if (distance > search.in.routeLimit + EPS) continue;
        GreedyInsertion candidate =
            new GreedyInsertion(distance - route.distance, i, position, route.depot, distance);
        if (existing == null || compareInsertion(candidate, existing) < 0) existing = candidate;
      }
    }

    // For a new route, choose the feasible open facility with the shortest round trip.
    GreedyInsertion fresh = null;
    for (int depot : search.fac) {
      double distance = 2 * search.in.facilityDistance[depot][customer];
      if (distance > search.in.routeLimit + EPS) continue;
      GreedyInsertion candidate =
          new GreedyInsertion(distance, routes.size(), 0, depot, distance);
      if (fresh == null || compareInsertion(candidate, fresh) < 0) fresh = candidate;
    }
    return new PathOptions(existing, fresh);
  }

  private static double weightedRouteDelta(GreedyInsertion insertion) {
    return insertion.delta * ExperimentParameters.W_R * ExperimentParameters.S;
  }

  private static GreedyInsertion cheaper(GreedyInsertion a, GreedyInsertion b) {
    if (a == null) return b;
    if (b == null) return a;
    return compareInsertion(a, b) <= 0 ? a : b;
  }

  private static int nearestExistingRepresentative(
      AssignmentSearch search,
      int customer,
      Set<Integer> selected,
      Map<Integer, RoutePlan> representativeRoute) {
    int best = -1;
    double bestDistance = Double.POSITIVE_INFINITY;
    for (int representative : new TreeSet<>(selected)) {
      RoutePlan route = representativeRoute.get(representative);
      if (!search.in.customerCoverMembership[customer][representative]
          || route.load + search.in.demand[customer] > search.in.capacity + EPS) continue;
      double distance = search.in.customerDistance[customer][representative];
      if (distance < bestDistance) {
        best = representative;
        bestDistance = distance;
      }
    }
    return best;
  }

  private static int nearestNewRepresentative(
      AssignmentSearch search,
      int target,
      Set<Integer> selected,
      Map<Integer, Integer> assignment) {
    int best = -1;
    double bestDistance = Double.POSITIVE_INFINITY;
    for (int representative : search.candidates.get(target)) {
      if (selected.contains(representative) || assignment.containsKey(representative)) continue;
      if (search.in.demand[representative] + search.in.demand[target]
          > search.in.capacity + EPS) continue;
      double distance = search.in.customerDistance[target][representative];
      if (distance < bestDistance) {
        best = representative;
        bestDistance = distance;
      }
    }
    return best;
  }

  private static void assignToRepresentative(
      AssignmentSearch search,
      int customer,
      int representative,
      Map<Integer, Integer> assignment,
      Map<Integer, Double> groupLoad,
      Map<Integer, RoutePlan> representativeRoute) {
    assignment.put(customer, representative);
    groupLoad.put(representative, groupLoad.get(representative) + search.in.demand[customer]);
    representativeRoute.get(representative).load += search.in.demand[customer];
  }

  private static int nearestPending(Instance in, int previous, Set<Integer> pending) {
    int best = -1;
    double bestDistance = Double.POSITIVE_INFINITY;
    for (int customer : pending) {
      double distance = in.customerDistance[previous][customer];
      if (distance < bestDistance) {
        best = customer;
        bestDistance = distance;
      }
    }
    return best;
  }

  private static int compareInsertion(GreedyInsertion a, GreedyInsertion b) {
    int c = Double.compare(a.delta, b.delta);
    if (c == 0) c = Integer.compare(a.routeIndex, b.routeIndex);
    if (c == 0) c = Integer.compare(a.position, b.position);
    return c != 0 ? c : Integer.compare(a.depot, b.depot);
  }

  private static RoutePlan applyInsertion(
      List<RoutePlan> routes, int customer, double load, GreedyInsertion insertion) {
    if (insertion.routeIndex == routes.size()) {
      RoutePlan route = new RoutePlan(insertion.depot,
          new ArrayList<>(List.of(customer)), load, insertion.distance);
      routes.add(route);
      return route;
    }
    RoutePlan route = routes.get(insertion.routeIndex);
    route.customers.add(insertion.position, customer);
    route.depot = insertion.depot;
    route.distance = insertion.distance;
    route.load += load;
    return route;
  }
}
