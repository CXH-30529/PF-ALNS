package heuristic;

import static heuristic.Model.*;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Command-line entry point for a single instance. */
public final class Main {
  /** Runs one instance with the quality configuration. */
  public static void main(String[] args) throws Exception {
    System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
    if (args.length == 0) {
      System.err.println("Usage: java heuristic.Main <instance.txt> [seed] [solution.json]");
      System.exit(2);
    }
    Path p = Path.of(args[0]);
    int seed = args.length > 1 ? Integer.parseInt(args[1]) : 20260803;
    ReferenceRunner.Run run = ReferenceRunner.solve(p, seed);
    HeuristicSolution s = run.solution();
    System.out.printf(
        Locale.ROOT,
        "instance=%s objective=%.10f facility_cost=%.10f route_cost=%.10f warehouse_coverage_cost=%.10f customer_coverage_cost=%.10f runtime=%.3f%n",
        s.instanceName,
        s.objective,
        s.facilityCost,
        s.routeCost,
        s.warehouseCoverageCost,
        s.customerCoverageCost,
        run.diagnostics().runtimeSeconds());
    System.out.println(
        "open_facilities_1based=" + s.openFacilities.stream().map(x -> x + 1).toList());
    System.out.println("coverage_assignment_0based=" + new TreeMap<>(s.coverage.assignment));
    System.out.println("direct_assignment_0based=" + new TreeMap<>(s.coverage.directAssignment));
    System.out.println("group_load_0based=" + new TreeMap<>(s.coverage.groupLoad));
    System.out.println(
        "operator_statistics=" + new TreeMap<>(run.diagnostics().operatorStatistics()));
    for (RoutePlan r : s.routes)
      System.out.printf(
          Locale.ROOT,
          "depot=%d customers=%s load=%.6f distance=%.10f%n",
          r.depot + 1,
          r.customers.stream().map(x -> x + 1).toList(),
          r.load,
          r.distance);
    if (args.length > 2) ReferenceRunner.writeJson(Path.of(args[2]), run);
  }
}


/** Shared facade for solving instances and writing results. */
final class ReferenceRunner {
  private ReferenceRunner() {}

  /** Loads and solves one instance with the quality configuration. */
  public static Run solve(Path instancePath, int seed) throws IOException {
    Instance instance = loadGeneratedInstance(instancePath);
    Solver.Config qualityConfig = Solver.Config.quality(seed, instance.facilityCount());
    Solver qualitySolver = new Solver(qualityConfig);
    HeuristicSolution solution = qualitySolver.solve(instance);
    return new Run(instance, solution, qualitySolver.lastDiagnostics);
  }

  /** Writes a run as UTF-8 JSON. */
  public static void writeJson(Path path, Run run) throws IOException {
    Path parent = path.toAbsolutePath().getParent();
    if (parent != null) Files.createDirectories(parent);
    Files.writeString(path, toJson(run), StandardCharsets.UTF_8);
  }

  /** Serializes a run with stable field ordering and one-based indices. */
  public static String toJson(Run run) {
    HeuristicSolution solution = run.solution;
    StringBuilder out = new StringBuilder(4096);
    out.append("{\n  \"instance\": \"").append(escape(solution.instanceName)).append("\",");
    number(out, "objective", solution.objective);
    number(out, "facility_cost", solution.facilityCost);
    number(out, "route_cost", solution.routeCost);
    number(out, "warehouse_coverage_cost", solution.warehouseCoverageCost);
    number(out, "customer_coverage_cost", solution.customerCoverageCost);
    out.append("\n  \"parameters\": {")
        .append("\"Q\": ").append(run.instance.capacity)
        .append(", \"T\": ").append(run.instance.routeLimit)
        .append(", \"r1\": ").append(run.instance.facilityRadius)
        .append(", \"r2\": ").append(run.instance.customerRadius)
        .append(", \"W_F\": ").append(ExperimentParameters.W_F)
        .append(", \"W_R\": ").append(ExperimentParameters.W_R)
        .append(", \"W_A\": ").append(ExperimentParameters.W_A)
        .append(", \"W_C\": ").append(ExperimentParameters.W_C)
        .append(", \"s\": ").append(ExperimentParameters.S)
        .append(", \"l\": ").append(ExperimentParameters.L)
        .append(", \"c\": ").append(ExperimentParameters.C)
        .append(", \"distance_normalization\": false},");
    out.append("\n  \"seed\": ").append(solution.seed).append(',');
    out.append("\n  \"open_facilities_1based\": ")
        .append(intList(solution.openFacilities.stream().map(x -> x + 1).toList()))
        .append(',');
    List<Integer> selected = solution.coverage.selected.stream().sorted().map(x -> x + 1).toList();
    out.append("\n  \"route_customers_1based\": ").append(intList(selected)).append(',');
    out.append("\n  \"direct_assignment_1based\": ")
        .append(mapJson(solution.directAssignment, true, true))
        .append(',');
    Map<Integer, Integer> covered = new TreeMap<>();
    for (Map.Entry<Integer, Integer> entry : solution.coverage.assignment.entrySet())
      if (!entry.getKey().equals(entry.getValue())) covered.put(entry.getKey(), entry.getValue());
    out.append("\n  \"customer_cover_assignment_1based\": ")
        .append(mapJson(covered, true, true))
        .append(',');
    out.append("\n  \"routes\": [");
    for (int i = 0; i < solution.routes.size(); i++) {
      RoutePlan route = solution.routes.get(i);
      if (i > 0) out.append(',');
      out.append("\n    {\"depot_1based\": ")
          .append(route.depot + 1)
          .append(", \"customers_1based\": ")
          .append(intList(route.customers.stream().map(x -> x + 1).toList()));
      out.append(
          String.format(
              Locale.ROOT, ", \"load\": %.10f, \"distance\": %.10f}", route.load, route.distance));
    }
    out.append("\n  ],\n  \"diagnostics\": {");
    Solver.Diagnostics d = run.diagnostics;
    out.append(String.format(Locale.ROOT, "\n    \"runtime_seconds\": %.6f,", d.runtimeSeconds()));
    out.append("\n    \"unique_facility_selections\": ")
        .append(d.uniqueFacilitySelections())
        .append(',');
    out.append("\n    \"feasible_facility_selections\": ")
        .append(d.feasibleFacilitySelections())
        .append(',');
    out.append("\n    \"cache_hits\": ").append(d.cacheHits()).append(',');
    // Legacy field name for full-budget facility evaluations.
    out.append("\n    \"intensified_elites\": ").append(d.deepFacilityEvaluations()).append(',');
    out.append("\n    \"search_profiles_evaluated\": 1,");
    out.append("\n    \"selected_profile\": \"quality\",");
    out.append("\n    \"operator_statistics\": ")
        .append(mapJson(d.operatorStatistics(), false, false));
    out.append("\n  }\n}\n");
    return out.toString();
  }

  /** Formats a cost field with ten decimal places. */
  private static void number(StringBuilder out, String name, double value) {
    out.append(String.format(Locale.ROOT, "\n  \"%s\": %.10f,", name, value));
  }

  /** Serializes an integer list. */
  private static String intList(List<Integer> values) {
    return values.toString();
  }

  /** Serializes a map with sorted keys and optional one-based indices. */
  private static String mapJson(Map<?, ?> source, boolean incrementKey, boolean incrementValue) {
    Map<String, Object> sorted = new TreeMap<>();
    for (Map.Entry<?, ?> e : source.entrySet()) {
      Object key = e.getKey(), value = e.getValue();
      if (incrementKey && key instanceof Integer i) key = i + 1;
      if (incrementValue && value instanceof Integer i) value = i + 1;
      sorted.put(String.valueOf(key), value);
    }
    List<String> parts = new ArrayList<>();
    for (Map.Entry<String, Object> e : sorted.entrySet())
      parts.add("\"" + escape(e.getKey()) + "\": " + e.getValue());
    return "{" + String.join(", ", parts) + "}";
  }

  /** Escapes a JSON string. */
  private static String escape(String value) {
    return value
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r");
  }

  /** Input, solution, and diagnostics for one run. */
  public record Run(
      Instance instance, HeuristicSolution solution, Solver.Diagnostics diagnostics) {}
}


/** Batch experiment entry point. */
final class BatchMain {
  /** Default experiment seeds. */
  private static final int[] DEFAULT_SEEDS = {630090, 313000, 747476, 846698, 785920};
  /** Pattern for standard instance names. */
  private static final Pattern INSTANCE = Pattern.compile("(r|c|rc)-(\\d+)-(\\d+)(?:-L(\\d+))?");

  private BatchMain() {}

  // Batch workflow.

  /** Runs the batch and writes reports. */
  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println(
          "Usage: java heuristic.BatchMain <instances-dir> <output-dir> [seed,seed,...]"
              + " [gurobi-summary.csv] [workers]");
      System.exit(2);
    }
    Path instances = Path.of(args[0]),
        output = Path.of(args[1]),
        solutions = output.resolve("solutions");
    Files.createDirectories(solutions);
    int[] seeds = args.length > 2 ? parseSeeds(args[2]) : DEFAULT_SEEDS;
    if (seeds.length == 0 || Arrays.stream(seeds).distinct().count() != seeds.length)
      throw new IllegalArgumentException("at least one seed is required and seeds must be distinct");
    Map<String, Reference> references =
        args.length > 3 ? loadReferences(Path.of(args[3])) : Map.of();
    int workers =
        args.length > 4
            ? Integer.parseInt(args[4])
            : java.lang.Math.min(4, Runtime.getRuntime().availableProcessors());
    if (workers < 1) throw new IllegalArgumentException("workers must be positive");
    List<Path> paths = selectedExperimentPaths(instances, 5);
    List<RunRow> runs = new ArrayList<>();
    long wallStarted = System.nanoTime();
    ExecutorService executor = Executors.newFixedThreadPool(workers);
    CompletionService<RunRow> completion = new ExecutorCompletionService<>(executor);
    int submitted = 0;
    for (Path path : paths)
      for (int seed : seeds) {
        String name = stem(path);
        Reference reference = references.get(name);
        submitted++;
        completion.submit(() -> solveOne(path, seed, reference, solutions));
      }
    try {
      for (int completed = 1; completed <= submitted; completed++) {
        RunRow row = completion.take().get();
        runs.add(row);
        writeSeedRuns(output.resolve("seed_runs.csv"), runs);
        System.out.printf(
            Locale.ROOT,
            "[%d/%d] %s seed=%d status=%s objective=%s runtime=%s%n",
            completed,
            submitted,
            row.instance,
            row.seed,
            row.success ? "SUCCESS" : "FAILED",
            row.success ? format2(row.objective) : "",
            row.success ? format2(row.runtime) : "");
      }
    } finally {
      executor.shutdownNow();
    }
    List<InstanceRow> summaries =
        writeInstanceSummary(output.resolve("instance_summary.csv"), paths, runs, seeds);
    writeReferenceBestComparison(
        output.resolve("comparison_with_reference_best.csv"), paths, runs);
    writeOverallSummary(
        output.resolve("overall_summary.json"),
        paths,
        runs,
        summaries,
        seeds,
        (System.nanoTime() - wallStarted) / 1.0e9);
  }

  /** Writes the comparison with Gurobi reference values. */
  private static void writeReferenceBestComparison(
      Path path, List<Path> paths, List<RunRow> runs) throws Exception {
    Map<String, List<RunRow>> grouped = new HashMap<>();
    for (RunRow row : runs)
      grouped.computeIfAbsent(row.instance, key -> new ArrayList<>()).add(row);
    List<String> lines = new ArrayList<>();
    lines.add(
        "instance,distribution,size,reference_best_objective,best_heuristic_objective,best_heuristic_vs_reference_best_gap_percent,best_seed,comparison");
    for (Path instancePath : paths) {
      String name = stem(instancePath);
      Parts parts = parts(name);
      List<RunRow> successful =
          grouped.getOrDefault(name, List.of()).stream().filter(r -> r.success).toList();
      if (successful.isEmpty()) continue;
      RunRow best =
          successful.stream().min(Comparator.comparingDouble(r -> r.objective)).orElseThrow();
      Reference ref =
          successful.stream()
              .map(r -> r.reference)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
      double comparisonGap = ref == null ? Double.NaN : gap(best.objective, ref.upper);
      String comparison =
          ref == null
              ? "NO_REFERENCE"
              : comparisonGap < 0 ? "BETTER" : comparisonGap > 0 ? "WORSE" : "EQUAL";
      lines.add(
          csv(name)
              + ","
              + parts.distribution
              + ","
              + parts.size
              + ","
              + value(ref != null, ref == null ? 0 : ref.upper)
              + ","
              + format2(best.objective)
              + ","
              + value(ref != null, comparisonGap)
              + ","
              + best.seed
              + ","
              + comparison);
    }
    Files.write(path, lines, StandardCharsets.UTF_8);
  }

  /** Runs one instance and seed, converting failures into result rows. */
  private static RunRow solveOne(Path path, int seed, Reference reference, Path solutions) {
    String name = stem(path);
    try {
      ReferenceRunner.Run run = ReferenceRunner.solve(path, seed);
      HeuristicSolution solution = run.solution();
      ReferenceRunner.writeJson(solutions.resolve(name + "__seed_" + seed + ".json"), run);
      return new RunRow(
          solution.instanceName,
          seed,
          true,
          "",
          solution.objective,
          reference,
          run.diagnostics().runtimeSeconds(),
          solution.facilityCost,
          solution.routeCost,
          solution.warehouseCoverageCost,
          solution.customerCoverageCost);
    } catch (Exception exception) {
      return new RunRow(
          name,
          seed,
          false,
          exception.toString(),
          Double.NaN,
          reference,
          Double.NaN,
          Double.NaN,
          Double.NaN,
          Double.NaN,
          Double.NaN);
    }
  }

  // Experiment reports.

  /** Writes one result row per seed. */
  private static void writeSeedRuns(Path path, List<RunRow> runs) throws Exception {
    List<String> lines = new ArrayList<>();
    lines.add(
        "instance,distribution,size,sequence,profile,seed,status,error,heuristic_objective,gurobi_upper_bound,gurobi_lower_bound,heuristic_vs_gurobi_upper_gap_percent,gurobi_upper_vs_lower_gap_percent,runtime_seconds,facility_cost,route_cost,warehouse_coverage_cost,customer_coverage_cost");
    List<RunRow> ordered =
        runs.stream()
            .sorted(Comparator.comparing((RunRow r) -> r.instance).thenComparingInt(r -> r.seed))
            .toList();
    for (RunRow row : ordered) {
      Parts parts = parts(row.instance);
      Reference ref = row.reference;
      lines.add(
          csv(row.instance)
              + ","
              + parts.distribution
              + ","
              + parts.size
              + ","
              + parts.sequence
              + ",quality,"
              + row.seed
              + ","
              + (row.success ? "SUCCESS" : "FAILED")
              + ","
              + csv(row.error)
              + ","
              + value(row.success, row.objective)
              + ","
              + value(ref != null, ref == null ? 0 : ref.upper)
              + ","
              + value(ref != null, ref == null ? 0 : ref.lower)
              + ","
              + value(row.success && ref != null, gap(row.objective, ref == null ? 1 : ref.upper))
              + ","
              + value(
                  ref != null, gap(ref == null ? 0 : ref.upper, ref == null ? 1 : ref.lower, true))
              + ","
              + value(row.success, row.runtime)
              + ","
              + value(row.success, row.facilityCost)
              + ","
              + value(row.success, row.routeCost)
              + ","
              + value(row.success, row.warehouseCoverageCost)
              + ","
              + value(row.success, row.customerCoverageCost));
    }
    Files.write(path, lines, StandardCharsets.UTF_8);
  }

  /** Writes per-instance aggregate statistics. */
  private static List<InstanceRow> writeInstanceSummary(
      Path path, List<Path> paths, List<RunRow> runs, int[] seeds) throws Exception {
    Map<String, List<RunRow>> grouped = new LinkedHashMap<>();
    for (RunRow row : runs)
      grouped.computeIfAbsent(row.instance, key -> new ArrayList<>()).add(row);
    List<String> lines = new ArrayList<>();
    List<InstanceRow> result = new ArrayList<>();
    lines.add(
        "instance,distribution,size,gurobi_status,gurobi_upper_bound,gurobi_lower_bound,gurobi_upper_vs_lower_gap_percent,gurobi_solve_time_seconds,gurobi_total_time_seconds,successful_runs,failed_runs,seeds,best_seed,best_heuristic_objective,mean_heuristic_objective,heuristic_vs_gurobi_upper_gap_percent,mean_runtime_seconds_per_seed,total_runtime_seconds_all_seeds,min_runtime_seconds,max_runtime_seconds");
    for (Path instancePath : paths) {
      String name = stem(instancePath);
      Parts parts = parts(name);
      Reference ref =
          grouped.getOrDefault(name, List.of()).stream()
              .map(r -> r.reference)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
      List<RunRow> successful =
          grouped.getOrDefault(name, List.of()).stream()
              .filter(r -> r.success)
              .sorted(Comparator.comparingInt(r -> r.seed))
              .toList();
      if (successful.isEmpty()) continue;
      RunRow best =
          successful.stream().min(Comparator.comparingDouble(r -> r.objective)).orElseThrow();
      double meanObjective =
          successful.stream().mapToDouble(r -> round2(r.objective)).average().orElseThrow();
      double totalRuntime = successful.stream().mapToDouble(r -> round2(r.runtime)).sum();
      InstanceRow summary = new InstanceRow(name, best.objective, ref, totalRuntime);
      result.add(summary);
      lines.add(
          csv(name)
              + ","
              + parts.distribution
              + ","
              + parts.size
              + ","
              + (ref == null ? "" : ref.status)
              + ","
              + value(ref != null, ref == null ? 0 : ref.upper)
              + ","
              + value(ref != null, ref == null ? 0 : ref.lower)
              + ","
              + value(
                  ref != null, gap(ref == null ? 0 : ref.upper, ref == null ? 1 : ref.lower, true))
              + ","
              + value(ref != null, ref == null ? 0 : ref.solveTime)
              + ","
              + value(ref != null, ref == null ? 0 : ref.totalTime)
              + ","
              + successful.size()
              + ","
              + (seeds.length - successful.size())
              + ","
              + csv(join(seeds))
              + ","
              + best.seed
              + ","
              + format2(best.objective)
              + ","
              + format2(meanObjective)
              + ","
              + value(ref != null, gap(best.objective, ref == null ? 1 : ref.upper))
              + ","
              + format2(totalRuntime / successful.size())
              + ","
              + format2(totalRuntime)
              + ","
              + format2(successful.stream().mapToDouble(r -> r.runtime).min().orElseThrow())
              + ","
              + format2(successful.stream().mapToDouble(r -> r.runtime).max().orElseThrow()));
    }
    Files.write(path, lines, StandardCharsets.UTF_8);
    return result;
  }

  /** Writes the overall JSON summary. */
  private static void writeOverallSummary(
      Path path,
      List<Path> paths,
      List<RunRow> runs,
      List<InstanceRow> summaries,
      int[] seeds,
      double wall)
      throws Exception {
    long successful = runs.stream().filter(r -> r.success).count();
    double cpu = runs.stream().filter(r -> r.success).mapToDouble(r -> round2(r.runtime)).sum();
    double meanHeuristic =
        summaries.stream()
            .filter(r -> r.reference != null)
            .mapToDouble(r -> gap(r.bestObjective, r.reference.upper))
            .average()
            .orElse(Double.NaN);
    double meanGurobi =
        summaries.stream()
            .filter(r -> r.reference != null)
            .mapToDouble(r -> gap(r.reference.upper, r.reference.lower, true))
            .average()
            .orElse(Double.NaN);
    String json =
        "{\n"
            + "  \"profile\": \"quality\",\n"
            + "  \"search_profiles_per_seed\": [\"quality\"],\n"
            + "  \"seeds\": "
            + Arrays.toString(seeds)
            + ",\n  \"instances\": "
            + paths.size()
            + ",\n  \"successful_seed_runs\": "
            + successful
            + ",\n  \"failed_seed_runs\": "
            + (runs.size() - successful)
            + ",\n  \"mean_heuristic_vs_gurobi_upper_gap_percent\": "
            + jsonNumber(meanHeuristic)
            + ",\n  \"mean_gurobi_upper_vs_lower_gap_percent\": "
            + jsonNumber(meanGurobi)
            + ",\n  \"total_heuristic_cpu_seconds\": "
            + format2(cpu)
            + ",\n  \"experiment_wall_time_seconds_this_session\": "
            + format2(wall)
            + ",\n"
            + "  \"rounding_policy\": \"ROUND_HALF_UP every objective and bound to 2 decimals"
            + " before calculating gaps; round each gap to 2 decimals\"\n"
            + "}\n";
    Files.writeString(path, json, StandardCharsets.UTF_8);
  }

  // Gurobi reference loading.

  /** Loads Gurobi reference results. */
  private static Map<String, Reference> loadReferences(Path path) throws Exception {
    List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
    Map<String, Reference> result = new HashMap<>();
    if (lines.isEmpty()) return result;
    List<String> header = parseCsv(lines.get(0).replace("\ufeff", ""));
    Map<String, Integer> column = new HashMap<>();
    for (int i = 0; i < header.size(); i++) column.put(header.get(i), i);
    for (int i = 1; i < lines.size(); i++) {
      if (lines.get(i).isBlank()) continue;
      List<String> row = parseCsv(lines.get(i));
      String name = get(row, column, "instance");
      String upper = get(row, column, "objective"), lower = get(row, column, "best_bound");
      if (name.isBlank() || upper.isBlank() || lower.isBlank()) continue;
      result.put(
          name,
          new Reference(
              get(row, column, "status"),
              Double.parseDouble(upper),
              Double.parseDouble(lower),
              number(get(row, column, "solve_time_seconds")),
              number(get(row, column, "total_time_seconds"))));
    }
    return result;
  }

  /** Parses one CSV row. */
  private static List<String> parseCsv(String line) {
    List<String> fields = new ArrayList<>();
    StringBuilder field = new StringBuilder();
    boolean quoted = false;
    for (int i = 0; i < line.length(); i++) {
      char ch = line.charAt(i);
      if (ch == '\"') {
        if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '\"') {
          field.append('\"');
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

  /** Returns a CSV field by column name. */
  private static String get(List<String> row, Map<String, Integer> column, String name) {
    Integer index = column.get(name);
    return index == null || index >= row.size() ? "" : row.get(index).trim();
  }

  /** Parses a number or returns NaN for an empty value. */
  private static double number(String value) {
    return value.isBlank() ? Double.NaN : Double.parseDouble(value);
  }

  // Naming, rounding, and serialization.

  /** Parses a comma-separated seed list. */
  private static int[] parseSeeds(String text) {
    return Arrays.stream(text.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
  }

  /** Returns the instance name without the .txt extension. */
  private static String stem(Path path) {
    return path.getFileName().toString().replaceFirst("\\.txt$", "");
  }

  /** Parses a standard instance name. */
  private static Parts parts(String name) {
    Matcher matcher = INSTANCE.matcher(name);
    if (!matcher.matches()) throw new IllegalArgumentException("invalid instance name: " + name);
    return new Parts(
        matcher.group(1), Integer.parseInt(matcher.group(2)), Integer.parseInt(matcher.group(3)));
  }

  /** Rounds to two decimal places using HALF_UP. */
  private static double round2(double value) {
    if (!Double.isFinite(value)) return value;
    return BigDecimal.valueOf(value).setScale(2, RoundingMode.HALF_UP).doubleValue();
  }

  /** Calculates the percentage difference from the Gurobi upper bound. */
  private static double gap(double objective, double upper) {
    if (!Double.isFinite(objective) || !Double.isFinite(upper) || Math.abs(upper) < 1e-12)
      return Double.NaN;
    double u = round2(upper);
    return round2(100.0 * (round2(objective) - u) / Math.abs(u));
  }

  /** Calculates the Gurobi bound gap. */
  private static double gap(double upper, double lower, boolean bounds) {
    if (!Double.isFinite(upper) || !Double.isFinite(lower) || Math.abs(upper) < 1e-12)
      return Double.NaN;
    double u = round2(upper);
    return round2(100.0 * (u - round2(lower)) / Math.abs(u));
  }

  /** Formats an optional value with two decimal places. */
  private static String value(boolean present, double value) {
    return present && Double.isFinite(value) ? format2(value) : "";
  }

  /** Formats a value with two decimal places. */
  private static String format2(double value) {
    return String.format(Locale.ROOT, "%.2f", round2(value));
  }

  /** Formats a JSON number, using null for non-finite values. */
  private static String jsonNumber(double value) {
    return Double.isFinite(value) ? format2(value) : "null";
  }

  /** Escapes a CSV field. */
  private static String csv(String value) {
    return "\"" + (value == null ? "" : value.replace("\"", "\"\"")) + "\"";
  }

  /** Joins integers with commas. */
  private static String join(int[] values) {
    return String.join(",", Arrays.stream(values).mapToObj(String::valueOf).toList());
  }

  // Report records.

  /** Parsed instance name. */
  private record Parts(String distribution, int size, int sequence) {}

  /** Gurobi reference data for one instance. */
  private record Reference(
      String status, double upper, double lower, double solveTime, double totalTime) {}

  /** Result for one seed. */
  private record RunRow(
      String instance,
      int seed,
      boolean success,
      String error,
      double objective,
      Reference reference,
      double runtime,
      double facilityCost,
      double routeCost,
      double warehouseCoverageCost,
      double customerCoverageCost) {}

  /** Aggregate result for one instance. */
  private record InstanceRow(
      String instance, double bestObjective, Reference reference, double totalRuntime) {}
}


/** Entry point for ordered regression runs. */
final class RegressionBatchMain {
  private RegressionBatchMain() {}

  /** Runs the regression manifest and writes results. */
  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println(
          "Usage: java heuristic.RegressionBatchMain <instances-dir> <manifest.tsv> <output-dir>");
      System.exit(2);
    }
    Path instances = Path.of(args[0]);
    Path manifest = Path.of(args[1]);
    Path output = Path.of(args[2]);
    Files.createDirectories(output);
    List<String> lines = Files.readAllLines(manifest, StandardCharsets.UTF_8);
    int total = (int) lines.stream().skip(1).filter(line -> !line.isBlank()).count();
    int completed = 0;
    double runtime = 0;
    for (String line : lines.subList(1, lines.size())) {
      if (line.isBlank()) continue;
      String[] fields = line.split("\\t");
      String instance = fields[0];
      int seed = Integer.parseInt(fields[1]);
      String outputName = fields[2];
      ReferenceRunner.Run run = ReferenceRunner.solve(instances.resolve(instance + ".txt"), seed);
      ReferenceRunner.writeJson(output.resolve(outputName), run);
      runtime += run.diagnostics().runtimeSeconds();
      completed++;
      System.out.printf(
          Locale.ROOT,
          "[%d/%d] %s seed=%d objective=%.10f runtime=%.3f%n",
          completed,
          total,
          instance,
          seed,
          run.solution().objective,
          run.diagnostics().runtimeSeconds());
      System.out.flush();
    }
    System.out.printf(Locale.ROOT, "TOTAL_SOLVER_SECONDS=%.6f%n", runtime);
  }
}

/** Development entry point for profiling shallow construction. */
final class ProfileShallowConstruction {
  private ProfileShallowConstruction() {}

  public static void main(String[] args) throws Exception {
    Instance in = loadGeneratedInstance(Path.of(args[0]));
    Solver.Config quality = Solver.Config.quality(630090, in.facilityCount());
    AssignmentSearch.Config coverage = quality.coverage.copy();
    RouteSearch.Config route = quality.route.copy();
    route.ruinIterations = 0;
    if (in.customerCount() >= 60) {
      coverage.alnsIterations = 0;
      route.maxVnsPasses = 2;
      route.enableSwap = false;
      route.sweepRotations = 3;
    }
    int max = Math.min(6, in.facilityCount());
    for (int size = 1; size <= max; size++) {
      long total = 0;
      int count = 0, feasible = 0;
      for (int mask = 1; mask < (1 << in.facilityCount()); mask++) {
        if (Integer.bitCount(mask) != size) continue;
        List<Integer> facilities = new ArrayList<>();
        for (int f = 0; f < in.facilityCount(); f++)
          if ((mask & (1 << f)) != 0) facilities.add(f);
        long started = System.nanoTime();
        AssignmentSearch.Result result =
            new AssignmentSearch(
                    in, facilities, coverage, route, 12345L + mask)
                .solve();
        total += System.nanoTime() - started;
        count++;
        if (result.feasible) feasible++;
      }
      System.out.printf(
          "size=%d count=%d feasible=%d total_ms=%.3f avg_ms=%.3f%n",
          size, count, feasible, total / 1e6, total / 1e6 / count);
    }
  }
}
