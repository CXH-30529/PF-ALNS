package heuristic;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/** Domain models and shared calculations. */
public final class Model {
  private Model() {}

  /** Experiment parameters. */
  public static final class ExperimentParameters {
    public static final double Q = 120.0, T = 150.0, R1 = 20.0, R2 = 10.0;
    public static final double W_F = 1.0, W_R = 1.0, W_A = 1.0, W_C = 1.0;
    public static final double S = 1.0;
    public static final double L = nonnegativeProperty("experiment.l", 0.1);
    public static final double C = nonnegativeProperty("experiment.c", 0.1);
    public static final boolean DISTANCE_NORMALIZATION = false;

    private ExperimentParameters() {}

    private static double nonnegativeProperty(String name, double defaultValue) {
      String raw = System.getProperty(name);
      if (raw == null || raw.isBlank()) return defaultValue;
      double value;
      try {
        value = Double.parseDouble(raw);
      } catch (NumberFormatException error) {
        throw new IllegalArgumentException("Invalid numeric system property " + name + "=" + raw, error);
      }
      if (!Double.isFinite(value) || value < 0.0)
        throw new IllegalArgumentException(name + " must be finite and nonnegative: " + raw);
      return value;
    }
  }

  /** Shared floating-point tolerance. */
  public static final double EPS = 1.0e-8;

  /** Instance data and precomputed geometry. */
  public static final class Instance {
    // Input data.
    public final String name;
    public final double[][] customerXY;
    public final double[] demand;
    public final double[][] facilityXY;
    public final double[] facilityCost;
    // Constraint parameters.
    public final double capacity, routeLimit, facilityRadius, customerRadius;
    // Precomputed matrices and coverage sets.
    public final double[][] customerDistance, facilityDistance, polarAngle;
    public final boolean[][] facilityCoverMembership, customerCoverMembership, routeReachable;
    public final List<Set<Integer>> facilityCover, customerCover;

    /** Creates an instance with the default experiment parameters. */
    public Instance(
        String name,
        double[][] customerXY,
        double[] demand,
        double[][] facilityXY,
        double[] facilityCost) {
      this(
          name,
          customerXY,
          demand,
          facilityXY,
          facilityCost,
          ExperimentParameters.Q,
          ExperimentParameters.T,
          ExperimentParameters.R1,
          ExperimentParameters.R2);
    }

    /** Creates an instance with explicit constraints. */
    public Instance(
        String name,
        double[][] customerXY,
        double[] demand,
        double[][] facilityXY,
        double[] facilityCost,
        double capacity,
        double routeLimit,
        double facilityRadius,
        double customerRadius) {
      if (customerXY.length != demand.length)
        throw new IllegalArgumentException(
            "Customer coordinates and demands have different lengths");
      if (facilityXY.length != facilityCost.length)
        throw new IllegalArgumentException("Facility coordinates and costs have different lengths");
      this.name = name;
      this.customerXY = customerXY;
      this.demand = demand;
      this.facilityXY = facilityXY;
      this.facilityCost = facilityCost;
      this.capacity = capacity;
      this.routeLimit = routeLimit;
      this.facilityRadius = facilityRadius;
      this.customerRadius = customerRadius;
      customerDistance = new double[customerCount()][customerCount()];
      for (int i = 0; i < customerCount(); i++)
        for (int j = 0; j < customerCount(); j++)
          customerDistance[i][j] = dist(customerXY[i], customerXY[j]);
      facilityDistance = new double[facilityCount()][customerCount()];
      for (int f = 0; f < facilityCount(); f++)
        for (int j = 0; j < customerCount(); j++)
          facilityDistance[f][j] = dist(facilityXY[f], customerXY[j]);
      polarAngle = new double[facilityCount()][customerCount()];
      routeReachable = new boolean[facilityCount()][customerCount()];
      facilityCoverMembership = new boolean[facilityCount()][customerCount()];
      customerCoverMembership = new boolean[customerCount()][customerCount()];
      for (int f = 0; f < facilityCount(); f++)
        for (int j = 0; j < customerCount(); j++) {
          polarAngle[f][j] =
              Math.atan2(customerXY[j][1] - facilityXY[f][1], customerXY[j][0] - facilityXY[f][0]);
          routeReachable[f][j] = 2 * facilityDistance[f][j] <= routeLimit + EPS;
          facilityCoverMembership[f][j] = facilityDistance[f][j] <= facilityRadius + EPS;
        }
      for (int i = 0; i < customerCount(); i++)
        for (int j = 0; j < customerCount(); j++)
          customerCoverMembership[i][j] = customerDistance[i][j] <= customerRadius + EPS;
      facilityCover = new ArrayList<>();
      for (int f = 0; f < facilityCount(); f++) {
        Set<Integer> s = new HashSet<>();
        for (int j = 0; j < customerCount(); j++) if (facilityCoverMembership[f][j]) s.add(j);
        facilityCover.add(Collections.unmodifiableSet(s));
      }
      customerCover = new ArrayList<>();
      for (int i = 0; i < customerCount(); i++) {
        Set<Integer> s = new HashSet<>();
        for (int j = 0; j < customerCount(); j++) if (customerCoverMembership[i][j]) s.add(j);
        customerCover.add(Collections.unmodifiableSet(s));
      }
    }

    /** Returns the number of customers. */
    public int customerCount() {
      return customerXY.length;
    }

    /** Returns the number of facilities. */
    public int facilityCount() {
      return facilityXY.length;
    }

    /** Returns the Euclidean distance between two points. */
    private static double dist(double[] a, double[] b) {
      return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
  }

  /** Customer coverage and assignment plan. */
  public static final class CoveragePlan {
    public final Set<Integer> selected;
    public final Map<Integer, Integer> assignment;
    public final Map<Integer, Double> groupLoad;
    public final Map<Integer, Integer> directAssignment;

    public CoveragePlan(Set<Integer> s, Map<Integer, Integer> a, Map<Integer, Double> g) {
      this(s, a, g, new HashMap<>());
    }

    public CoveragePlan(Set<Integer> s, Map<Integer, Integer> a, Map<Integer, Double> g,
        Map<Integer, Integer> d) {
      selected = s;
      assignment = a;
      groupLoad = g;
      directAssignment = d;
    }

    /** Returns a copy with independent mutable collections. */
    public CoveragePlan copy() {
      return new CoveragePlan(
          new HashSet<>(selected), new HashMap<>(assignment), new HashMap<>(groupLoad),
          new HashMap<>(directAssignment));
    }
  }

  /** Objective cost breakdown. */
  public record CostBreakdown(double facility, double route, double warehouseCoverage,
      double customerCoverage) {
    public double service() { return route + warehouseCoverage + customerCoverage; }
    public double total() { return facility + service(); }
  }

  public static CostBreakdown costs(Instance in, List<Integer> facilities,
      CoveragePlan plan, List<RoutePlan> routes) {
    double f = facilities.stream().mapToDouble(k -> ExperimentParameters.W_F * in.facilityCost[k]).sum();
    double r = routes.stream().mapToDouble(p -> ExperimentParameters.W_R * ExperimentParameters.S
        * routeDistance(in, p.depot, p.customers)).sum();
    double a = plan.directAssignment.entrySet().stream().mapToDouble(e ->
        ExperimentParameters.W_A * ExperimentParameters.L * in.facilityDistance[e.getValue()][e.getKey()]).sum();
    double c = plan.assignment.entrySet().stream().mapToDouble(e ->
        ExperimentParameters.W_C * ExperimentParameters.C * in.customerDistance[e.getValue()][e.getKey()]).sum();
    return new CostBreakdown(f, r, a, c);
  }

  /** Route assigned to one facility. */
  public static final class RoutePlan {
    public int depot;
    public List<Integer> customers;
    public double load, distance;

    public RoutePlan(int d, List<Integer> c, double l, double x) {
      depot = d;
      customers = c;
      load = l;
      distance = x;
    }

    /** Returns a copy with an independent customer sequence. */
    public RoutePlan copy() {
      return new RoutePlan(depot, new ArrayList<>(customers), load, distance);
    }
  }

  /** Final heuristic solution. */
  public static final class HeuristicSolution {
    public final String instanceName;
    public final List<Integer> openFacilities;
    public final CoveragePlan coverage;
    public final List<RoutePlan> routes;
    public final Map<Integer, Integer> directAssignment;
    public final double objective, facilityCost, routeCost;
    public final double warehouseCoverageCost, customerCoverageCost;
    public final boolean feasible;
    public final String violation;
    public final int seed;

    public HeuristicSolution(
        String n,
        List<Integer> o,
        CoveragePlan c,
        List<RoutePlan> r,
        Map<Integer, Integer> d,
        double z,
        double f,
        double rc,
        double wc,
        double cc,
        boolean ok,
        String v,
        int seed) {
      instanceName = n;
      openFacilities = o;
      coverage = c;
      routes = r;
      directAssignment = d;
      objective = z;
      facilityCost = f;
      routeCost = rc;
      warehouseCoverageCost = wc;
      customerCoverageCost = cc;
      feasible = ok;
      violation = v;
      this.seed = seed;
    }

    /** Returns an independent copy of this solution. */
    public HeuristicSolution copy() {
      return new HeuristicSolution(
          instanceName,
          List.copyOf(openFacilities),
          coverage.copy(),
          routes.stream().map(RoutePlan::copy).toList(),
          new HashMap<>(directAssignment),
          objective,
          facilityCost,
          routeCost,
          warehouseCoverageCost,
          customerCoverageCost,
          feasible,
          violation,
          seed);
    }
  }

  // Instance and reference-data loading.

  /** Loads an instance from the standard text format. */
  public static Instance loadGeneratedInstance(Path path) throws IOException {
    List<String> raw = Files.readAllLines(path, StandardCharsets.UTF_8), lines = new ArrayList<>();
    for (String s : raw) {
      s = s.replace("\ufeff", "").trim();
      if (!s.isEmpty()) lines.add(s);
    }
    if (lines.size() < 7 || !lines.get(0).startsWith("INSTANCE "))
      throw new IllegalArgumentException("Unsupported instance format: " + path);
    String name = lines.get(0).split("\\s+", 2)[1];
    int n = Integer.parseInt(lines.get(1).split("\\s+")[1]);
    int m = Integer.parseInt(lines.get(2).split("\\s+")[1]);
    int ch = lines.indexOf("CUSTOMER") + 2, fh = lines.indexOf("WAREHOUSE");
    if (fh - ch != n || lines.size() - (fh + 2) != m)
      throw new IllegalArgumentException("Declared dimensions do not match data rows in " + path);
    double[][] cxy = new double[n][2];
    double[] dem = new double[n];
    for (int i = 0; i < n; i++) {
      String[] x = lines.get(ch + i).split("\\s+");
      if (x.length != 4 || Integer.parseInt(x[0]) != i + 1)
        throw new IllegalArgumentException("Malformed customer row");
      cxy[i][0] = Double.parseDouble(x[1]);
      cxy[i][1] = Double.parseDouble(x[2]);
      dem[i] = Double.parseDouble(x[3]);
    }
    double[][] fxy = new double[m][2];
    double[] cost = new double[m];
    for (int i = 0; i < m; i++) {
      String[] x = lines.get(fh + 2 + i).split("\\s+");
      if (x.length != 4 || Integer.parseInt(x[0]) != i + 1)
        throw new IllegalArgumentException("Malformed warehouse row");
      fxy[i][0] = Double.parseDouble(x[1]);
      fxy[i][1] = Double.parseDouble(x[2]);
      cost[i] = Double.parseDouble(x[3]);
    }
    return new Instance(name, cxy, dem, fxy, cost);
  }

  /** Selects and orders experiment instance paths. */
  public static List<Path> selectedExperimentPaths(Path dir, int firstK) throws IOException {
    Pattern p = Pattern.compile("(r|c|rc)-(\\d+)-(\\d+)(?:-L(\\d+))?\\.txt");
    Map<String, Integer> d = Map.of("r", 0, "c", 1, "rc", 2);
    List<Path> x = new ArrayList<>();
    try (var s = Files.walk(dir)) {
      s.filter(Files::isRegularFile)
          .forEach(
              q -> {
                Matcher m = p.matcher(q.getFileName().toString());
                if (m.matches() && Integer.parseInt(m.group(3)) <= firstK) x.add(q);
              });
    }
    x.sort(
        Comparator.comparingInt(
                (Path q) -> {
                  Matcher m = p.matcher(q.getFileName().toString());
                  m.matches();
                  return Integer.parseInt(m.group(2));
                })
            .thenComparingInt(
                q -> {
                  Matcher m = p.matcher(q.getFileName().toString());
                  m.matches();
                  return d.get(m.group(1));
                })
            .thenComparingInt(
                q -> {
                  Matcher m = p.matcher(q.getFileName().toString());
                  m.matches();
                  return Integer.parseInt(m.group(3));
                })
            .thenComparingInt(q -> {
              Matcher m = p.matcher(q.getFileName().toString());
              m.matches();
              return m.group(4) == null ? 0 : Integer.parseInt(m.group(4));
            })
            .thenComparing(Path::toString));
    return x;
  }

  /** Loads reference objective values from CSV. */
  public static Map<String, Double> loadReferenceObjectives(Path path) throws IOException {
    List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
    Map<String, Double> result = new LinkedHashMap<>();
    if (lines.isEmpty()) return result;
    List<String> header = parseCsvRow(lines.get(0).replace("\ufeff", ""));
    int instanceColumn = header.indexOf("instance"), objectiveColumn = header.indexOf("objective");
    if (instanceColumn < 0 || objectiveColumn < 0)
      throw new IllegalArgumentException(
          "CSV must contain instance and objective columns: " + path);
    for (int i = 1; i < lines.size(); i++) {
      if (lines.get(i).isBlank()) continue;
      List<String> row = parseCsvRow(lines.get(i));
      if (row.size() <= Math.max(instanceColumn, objectiveColumn)) continue;
      String name = row.get(instanceColumn).trim(), value = row.get(objectiveColumn).trim();
      if (!name.isEmpty() && !value.isEmpty()) result.put(name, Double.parseDouble(value));
    }
    return result;
  }

  /** Parses one CSV row. */
  private static List<String> parseCsvRow(String line) {
    List<String> fields = new ArrayList<>();
    StringBuilder field = new StringBuilder();
    boolean quoted = false;
    for (int i = 0; i < line.length(); i++) {
      char ch = line.charAt(i);
      if (ch == '"') {
        if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') {
          field.append('"');
          i++;
        } else quoted = !quoted;
      } else if (ch == ',' && !quoted) {
        fields.add(field.toString());
        field.setLength(0);
      } else field.append(ch);
    }
    fields.add(field.toString());
    return fields;
  }

  // Shared calculations and normalization.

  /** Calculates the round-trip distance for a route. */
  public static double routeDistance(Instance in, int depot, Collection<Integer> customers) {
    if (customers.isEmpty()) return 0;
    List<Integer> s =
        customers instanceof List<?> ? (List<Integer>) customers : new ArrayList<>(customers);
    double total = in.facilityDistance[depot][s.get(0)];
    for (int i = 1; i < s.size(); i++) total += in.customerDistance[s.get(i - 1)][s.get(i)];
    return total + in.facilityDistance[depot][s.get(s.size() - 1)];
  }

  /** Applies Python-compatible rounding to a nonnegative value. */
  public static int pythonRound(double value) {
    double floor = java.lang.Math.floor(value), fraction = value - floor;
    if (fraction < 0.5) return (int) floor;
    if (fraction > 0.5) return (int) floor + 1;
    long lower = (long) floor;
    return (int) ((lower & 1L) == 0L ? lower : lower + 1L);
  }

  /** Returns sorted distinct facility indices. */
  public static List<Integer> canonicalFacilities(Collection<Integer> f) {
    return f.stream().distinct().sorted().toList();
  }
}
