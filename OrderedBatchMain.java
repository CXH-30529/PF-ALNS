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

/** Ordered, resumable experiment runner keyed by the outer instance-folder name. */
public final class OrderedBatchMain {
  private static final int[] DEFAULT_SEEDS = {630090, 313000, 747476, 846698, 785920};
  private static final Pattern FOLDER = Pattern.compile("(r|c|rc)-(\\d+)-(\\d+)-L(\\d+)");
  private static final Pattern SOLUTION_FILE =
      Pattern.compile("(.+)__seed_(-?\\d+)\\.json");

  private OrderedBatchMain() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.err.println(
          "Usage: java heuristic.OrderedBatchMain <data-root> <output-dir> <size>"
              + " [workers] [seed,seed,...]");
      System.exit(2);
    }
    Path dataRoot = Path.of(args[0]).toAbsolutePath().normalize();
    Path output = Path.of(args[1]).toAbsolutePath().normalize();
    int size = Integer.parseInt(args[2]);
    int workers = args.length > 3 ? Integer.parseInt(args[3]) : 6;
    int[] seeds = args.length > 4 ? parseSeeds(args[4]) : DEFAULT_SEEDS;
    if (workers < 1) throw new IllegalArgumentException("workers must be positive");
    if (seeds.length == 0 || Arrays.stream(seeds).distinct().count() != seeds.length)
      throw new IllegalArgumentException("seeds must be non-empty and distinct");

    Files.createDirectories(output.resolve("solutions"));
    Files.createDirectories(output.resolve("logs"));
    List<InstanceSpec> instances = discover(dataRoot, size);
    if (instances.isEmpty()) throw new IllegalArgumentException("no folders found for size " + size);

    List<TaskSpec> pending = new ArrayList<>();
    int resumed = 0;
    for (InstanceSpec instance : instances) {
      for (int seed : seeds) {
        Path result = solutionPath(output, instance.folderName, seed);
        if (Files.isRegularFile(result) && isCompleteSolution(result)) resumed++;
        else pending.add(new TaskSpec(instance, seed));
      }
    }

    rebuildReports(dataRoot, output, seeds);
    int total = instances.size() * seeds.length;
    System.out.printf(
        Locale.ROOT,
        "SIZE_START size=%d folders=%d runs=%d resumed=%d pending=%d workers=%d%n",
        size, instances.size(), total, resumed, pending.size(), workers);
    System.out.flush();

    ExecutorService pool = Executors.newFixedThreadPool(workers);
    CompletionService<RunResult> completion = new ExecutorCompletionService<>(pool);
    for (TaskSpec task : pending) completion.submit(() -> solveOne(task, output));
    int succeeded = resumed;
    int failed = 0;
    try {
      for (int i = 1; i <= pending.size(); i++) {
        RunResult result = completion.take().get();
        if (result.success) succeeded++;
        else {
          failed++;
          appendFailure(output.resolve("failed_runs.log"), result);
        }
        if (i % 30 == 0 || i == pending.size()) rebuildReports(dataRoot, output, seeds);
        System.out.printf(
            Locale.ROOT,
            "[%d/%d] size=%d folder=%s input=%s seed=%d status=%s objective=%s runtime=%s%n",
            resumed + i,
            total,
            size,
            result.task.instance.folderName,
            result.task.instance.input.getFileName(),
            result.task.seed,
            result.success ? "SUCCESS" : "FAILED",
            result.success ? f6(result.objective) : "",
            result.success ? f3(result.runtime) : "");
        if (!result.success) System.out.println("  error=" + result.error);
        System.out.flush();
      }
    } finally {
      pool.shutdownNow();
    }
    rebuildReports(dataRoot, output, seeds);
    updateCompletedSizes(output, size, failed == 0 && succeeded == total);
    rebuildReports(dataRoot, output, seeds);
    System.out.printf(
        Locale.ROOT,
        "SIZE_DONE size=%d successful=%d failed=%d expected=%d%n",
        size, succeeded, failed, total);
  }

  private static RunResult solveOne(TaskSpec task, Path output) {
    long started = System.nanoTime();
    try {
      ReferenceRunner.Run run = ReferenceRunner.solve(task.instance.input, task.seed);
      String base = ReferenceRunner.toJson(run);
      String json =
          "{\n  \"instance_folder\": \""
              + escape(task.instance.folderName)
              + "\",\n  \"input_file\": \""
              + escape(task.instance.input.getFileName().toString())
              + "\"," + base.substring(1);
      writeAtomic(solutionPath(output, task.instance.folderName, task.seed), json);
      return new RunResult(
          task, true, "", run.solution().objective, run.diagnostics().runtimeSeconds());
    } catch (Throwable error) {
      return new RunResult(
          task,
          false,
          error.toString(),
          Double.NaN,
          (System.nanoTime() - started) / 1.0e9);
    }
  }

  private static List<InstanceSpec> discover(Path dataRoot, int requestedSize) throws IOException {
    List<InstanceSpec> result = new ArrayList<>();
    try (var stream = Files.list(dataRoot)) {
      for (Path folder : stream.filter(Files::isDirectory).toList()) {
        Matcher m = FOLDER.matcher(folder.getFileName().toString());
        if (!m.matches() || Integer.parseInt(m.group(2)) != requestedSize) continue;
        List<Path> txt;
        try (var files = Files.list(folder)) {
          txt = files.filter(p -> Files.isRegularFile(p) && p.getFileName().toString().endsWith(".txt"))
              .sorted().toList();
        }
        if (txt.size() != 1)
          throw new IllegalArgumentException(
              "expected exactly one txt in " + folder + ", found " + txt.size());
        result.add(
            new InstanceSpec(
                folder.getFileName().toString(),
                m.group(1),
                requestedSize,
                Integer.parseInt(m.group(3)),
                Integer.parseInt(m.group(4)),
                folder,
                txt.get(0)));
      }
    }
    result.sort(instanceComparator());
    return result;
  }

  private static Comparator<InstanceSpec> instanceComparator() {
    Map<String, Integer> order = Map.of("c", 0, "r", 1, "rc", 2);
    return Comparator.comparingInt((InstanceSpec x) -> x.size)
        .thenComparingInt(x -> order.get(x.distribution))
        .thenComparingInt(x -> x.sequence)
        .thenComparingInt(x -> x.level)
        .thenComparing(x -> x.folderName);
  }

  private static void rebuildReports(Path dataRoot, Path output, int[] seeds) throws Exception {
    Map<String, InstanceSpec> allInstances = discoverAll(dataRoot);
    Map<String, Reference> references = new HashMap<>();
    for (InstanceSpec instance : allInstances.values())
      references.put(instance.folderName, readReference(instance.folder));

    List<CompletedRun> runs = new ArrayList<>();
    Path solutions = output.resolve("solutions");
    if (Files.isDirectory(solutions)) {
      try (var stream = Files.list(solutions)) {
        for (Path path : stream.filter(Files::isRegularFile).toList()) {
          Matcher m = SOLUTION_FILE.matcher(path.getFileName().toString());
          if (!m.matches() || !allInstances.containsKey(m.group(1))) continue;
          CompletedRun run = readCompletedRun(path, allInstances.get(m.group(1)), Integer.parseInt(m.group(2)));
          if (run != null) runs.add(run);
        }
      }
    }
    runs.sort(
        Comparator.comparing((CompletedRun r) -> r.instance, instanceComparator())
            .thenComparingInt(r -> r.seed));
    writeSeedRuns(output.resolve("seed_runs.csv"), runs, references);
    writeInstanceSummary(output.resolve("instance_summary.csv"), runs, references, seeds);
    writeProgress(output.resolve("progress_by_size.csv"), allInstances.values(), runs, seeds);
  }

  private static Map<String, InstanceSpec> discoverAll(Path dataRoot) throws IOException {
    Map<String, InstanceSpec> result = new LinkedHashMap<>();
    int minSize = Integer.getInteger("experiment.minSize", 20);
    int maxSize = Integer.getInteger("experiment.maxSize", 100);
    for (int size = minSize; size <= maxSize; size += 10)
      for (InstanceSpec instance : discover(dataRoot, size)) result.put(instance.folderName, instance);
    return result;
  }

  private static CompletedRun readCompletedRun(Path path, InstanceSpec instance, int seed) {
    try {
      String json = Files.readString(path, StandardCharsets.UTF_8);
      double objective = jsonNumber(json, "objective");
      double runtime = jsonNumber(json, "runtime_seconds");
      if (!Double.isFinite(objective) || !Double.isFinite(runtime)) return null;
      return new CompletedRun(
          instance,
          seed,
          objective,
          runtime,
          jsonNumber(json, "facility_cost"),
          jsonNumber(json, "route_cost"),
          jsonNumber(json, "warehouse_coverage_cost"),
          jsonNumber(json, "customer_coverage_cost"));
    } catch (Exception ignored) {
      return null;
    }
  }

  private static Reference readReference(Path folder) {
    try {
      String json = Files.readString(folder.resolve("solution.json"), StandardCharsets.UTF_8);
      String status = jsonString(json, "status");
      double upper = jsonNumber(json, "objective");
      double lower = jsonNumber(json, "best_bound");
      double relativeGap = jsonNumber(json, "relative_gap");
      String log = Files.readString(folder.resolve("run.log"), StandardCharsets.UTF_8);
      double solveTime = firstNumber(log, "Gurobi求解时间\\s*:\\s*([0-9.Ee+-]+)");
      double totalTime = firstNumber(log, "总运行时间\\s*:\\s*([0-9.Ee+-]+)");
      return new Reference(status, upper, lower, relativeGap, solveTime, totalTime);
    } catch (Exception error) {
      return new Reference("REFERENCE_ERROR", Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
    }
  }

  private static void writeSeedRuns(
      Path path, List<CompletedRun> runs, Map<String, Reference> references) throws IOException {
    List<String> lines = new ArrayList<>();
    lines.add(
        "instance_folder,input_file,distribution,size,sequence,L,seed,status,gurobi_upper_bound,gurobi_lower_bound,gurobi_solve_time_seconds,gurobi_total_time_seconds,gurobi_gap_ratio,gurobi_gap_percent,heuristic_objective,heuristic_runtime_seconds,heuristic_gap_ratio,heuristic_gap_percent,facility_cost,route_cost,warehouse_coverage_cost,customer_coverage_cost");
    for (CompletedRun run : runs) {
      InstanceSpec x = run.instance;
      Reference ref = references.get(x.folderName);
      lines.add(
          csv(x.folderName) + "," + csv(x.input.getFileName().toString()) + "," + x.distribution
              + "," + x.size + "," + x.sequence + "," + x.level + "," + run.seed + ",SUCCESS,"
              + val(ref.upper) + "," + val(ref.lower) + "," + val(ref.solveTime) + ","
              + val(ref.totalTime) + "," + val(gurobiGapRatio(ref)) + ","
              + val(100.0 * gurobiGapRatio(ref)) + "," + val(run.objective) + ","
              + val(run.runtime) + "," + val(heuristicGapRatio(run.objective, ref.upper)) + ","
              + val(100.0 * heuristicGapRatio(run.objective, ref.upper)) + ","
              + val(run.facilityCost) + "," + val(run.routeCost) + ","
              + val(run.warehouseCoverageCost) + "," + val(run.customerCoverageCost));
    }
    writeAtomic(path, String.join(System.lineSeparator(), lines) + System.lineSeparator());
  }

  private static void writeInstanceSummary(
      Path path, List<CompletedRun> runs, Map<String, Reference> references, int[] seeds)
      throws IOException {
    Map<String, List<CompletedRun>> grouped = new LinkedHashMap<>();
    for (CompletedRun run : runs)
      grouped.computeIfAbsent(run.instance.folderName, key -> new ArrayList<>()).add(run);
    List<String> lines = new ArrayList<>();
    lines.add(
        "instance_folder,input_file,distribution,size,sequence,L,gurobi_status,gurobi_upper_bound,gurobi_lower_bound,gurobi_solve_time_seconds,gurobi_total_time_seconds,gurobi_gap_ratio,gurobi_gap_percent,completed_runs,expected_runs,seeds,best_seed,best_heuristic_objective,mean_heuristic_objective,best_heuristic_gap_ratio,best_heuristic_gap_percent,mean_heuristic_gap_ratio,mean_heuristic_gap_percent,mean_runtime_seconds,min_runtime_seconds,max_runtime_seconds,total_runtime_seconds");
    List<InstanceSpec> ordered = grouped.values().stream().map(x -> x.get(0).instance)
        .sorted(instanceComparator()).toList();
    for (InstanceSpec x : ordered) {
      List<CompletedRun> xs = grouped.get(x.folderName);
      CompletedRun best = xs.stream().min(Comparator.comparingDouble(r -> r.objective)).orElseThrow();
      double meanObjective = xs.stream().mapToDouble(r -> r.objective).average().orElse(Double.NaN);
      double meanRuntime = xs.stream().mapToDouble(r -> r.runtime).average().orElse(Double.NaN);
      double minRuntime = xs.stream().mapToDouble(r -> r.runtime).min().orElse(Double.NaN);
      double maxRuntime = xs.stream().mapToDouble(r -> r.runtime).max().orElse(Double.NaN);
      double totalRuntime = xs.stream().mapToDouble(r -> r.runtime).sum();
      Reference ref = references.get(x.folderName);
      lines.add(
          csv(x.folderName) + "," + csv(x.input.getFileName().toString()) + "," + x.distribution
              + "," + x.size + "," + x.sequence + "," + x.level + "," + csv(ref.status) + ","
              + val(ref.upper) + "," + val(ref.lower) + "," + val(ref.solveTime) + ","
              + val(ref.totalTime) + "," + val(gurobiGapRatio(ref)) + ","
              + val(100.0 * gurobiGapRatio(ref)) + "," + xs.size() + "," + seeds.length + ","
              + csv(join(seeds)) + "," + best.seed + "," + val(best.objective) + ","
              + val(meanObjective) + "," + val(heuristicGapRatio(best.objective, ref.upper)) + ","
              + val(100.0 * heuristicGapRatio(best.objective, ref.upper)) + ","
              + val(heuristicGapRatio(meanObjective, ref.upper)) + ","
              + val(100.0 * heuristicGapRatio(meanObjective, ref.upper)) + ","
              + val(meanRuntime) + "," + val(minRuntime) + "," + val(maxRuntime) + ","
              + val(totalRuntime));
    }
    writeAtomic(path, String.join(System.lineSeparator(), lines) + System.lineSeparator());
  }

  private static void writeProgress(
      Path path, Collection<InstanceSpec> instances, List<CompletedRun> runs, int[] seeds)
      throws IOException {
    Map<Integer, Long> folders = new TreeMap<>();
    for (InstanceSpec x : instances) folders.merge(x.size, 1L, Long::sum);
    Map<Integer, Long> completed = new TreeMap<>();
    for (CompletedRun run : runs) completed.merge(run.instance.size, 1L, Long::sum);
    Set<Integer> done = readCompletedSizes(path.getParent().resolve("completed_sizes.txt"));
    List<String> lines = new ArrayList<>();
    lines.add("size,folders,expected_runs,completed_runs,remaining_runs,completion_percent,status");
    for (var entry : folders.entrySet()) {
      int size = entry.getKey();
      long expected = entry.getValue() * seeds.length;
      long actual = completed.getOrDefault(size, 0L);
      String status = done.contains(size) ? "COMPLETE" : actual == 0 ? "PENDING" : "RUNNING";
      lines.add(size + "," + entry.getValue() + "," + expected + "," + actual + ","
          + (expected - actual) + "," + f2(100.0 * actual / expected) + "," + status);
    }
    writeAtomic(path, String.join(System.lineSeparator(), lines) + System.lineSeparator());
  }

  private static void updateCompletedSizes(Path output, int size, boolean complete) throws IOException {
    Path path = output.resolve("completed_sizes.txt");
    Set<Integer> sizes = readCompletedSizes(path);
    if (complete) sizes.add(size); else sizes.remove(size);
    String text = String.join(System.lineSeparator(), sizes.stream().sorted().map(String::valueOf).toList());
    writeAtomic(path, text.isEmpty() ? "" : text + System.lineSeparator());
  }

  private static Set<Integer> readCompletedSizes(Path path) {
    Set<Integer> result = new TreeSet<>();
    try {
      if (Files.isRegularFile(path))
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8))
          if (!line.isBlank()) result.add(Integer.parseInt(line.trim()));
    } catch (Exception ignored) {}
    return result;
  }

  private static void appendFailure(Path path, RunResult result) throws IOException {
    String line = new Date() + "\t" + result.task.instance.folderName + "\t"
        + result.task.instance.input.getFileName() + "\t" + result.task.seed + "\t" + result.error
        + System.lineSeparator();
    Files.writeString(path, line, StandardCharsets.UTF_8,
        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
  }

  private static boolean isCompleteSolution(Path path) {
    try {
      String json = Files.readString(path, StandardCharsets.UTF_8);
      return Double.isFinite(jsonNumber(json, "objective"))
          && Double.isFinite(jsonNumber(json, "runtime_seconds"));
    } catch (Exception ignored) {
      return false;
    }
  }

  private static Path solutionPath(Path output, String folder, int seed) {
    return output.resolve("solutions").resolve(folder + "__seed_" + seed + ".json");
  }

  private static double jsonNumber(String json, String key) {
    Pattern p = Pattern.compile("\\\"" + Pattern.quote(key)
        + "\\\"\\s*:\\s*(-?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][+-]?\\d+)?)");
    Matcher m = p.matcher(json);
    return m.find() ? Double.parseDouble(m.group(1)) : Double.NaN;
  }

  private static String jsonString(String json, String key) {
    Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    Matcher m = p.matcher(json);
    return m.find() ? m.group(1) : "";
  }

  private static double firstNumber(String text, String regex) {
    Matcher m = Pattern.compile(regex).matcher(text);
    return m.find() ? Double.parseDouble(m.group(1)) : Double.NaN;
  }

  private static double gurobiGapRatio(Reference ref) {
    if (Double.isFinite(ref.relativeGap)) return ref.relativeGap;
    return heuristicGapRatio(ref.upper, ref.lower);
  }

  private static double heuristicGapRatio(double objective, double upper) {
    if (!Double.isFinite(objective) || !Double.isFinite(upper) || Math.abs(upper) < 1e-12)
      return Double.NaN;
    return (objective - upper) / Math.abs(upper);
  }

  private static int[] parseSeeds(String text) {
    return Arrays.stream(text.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
  }

  private static String join(int[] values) {
    return String.join(";", Arrays.stream(values).mapToObj(String::valueOf).toList());
  }

  private static String csv(String value) {
    return "\"" + (value == null ? "" : value.replace("\"", "\"\"")) + "\"";
  }

  private static String val(double value) {
    return Double.isFinite(value) ? String.format(Locale.ROOT, "%.10f", value) : "";
  }

  private static String f2(double value) {
    return String.format(Locale.ROOT, "%.2f", value);
  }

  private static String f3(double value) {
    return String.format(Locale.ROOT, "%.3f", value);
  }

  private static String f6(double value) {
    return String.format(Locale.ROOT, "%.6f", value);
  }

  private static String escape(String value) {
    return value.replace("\\", "\\\\").replace("\"", "\\\"");
  }

  private static void writeAtomic(Path path, String text) throws IOException {
    Path parent = path.toAbsolutePath().getParent();
    if (parent != null) Files.createDirectories(parent);
    Path tmp = path.resolveSibling(
        path.getFileName() + "." + ProcessHandle.current().pid() + "."
            + Thread.currentThread().threadId() + ".tmp");
    Files.writeString(tmp, text, StandardCharsets.UTF_8);
    try {
      Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    } catch (AtomicMoveNotSupportedException ignored) {
      Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  private record InstanceSpec(
      String folderName,
      String distribution,
      int size,
      int sequence,
      int level,
      Path folder,
      Path input) {}

  private record TaskSpec(InstanceSpec instance, int seed) {}

  private record RunResult(
      TaskSpec task, boolean success, String error, double objective, double runtime) {}

  private record CompletedRun(
      InstanceSpec instance,
      int seed,
      double objective,
      double runtime,
      double facilityCost,
      double routeCost,
      double warehouseCoverageCost,
      double customerCoverageCost) {}

  private record Reference(
      String status,
      double upper,
      double lower,
      double relativeGap,
      double solveTime,
      double totalTime) {}
}
