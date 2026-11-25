package heuristic;

import java.util.*;

/** CPython-compatible MT19937 random number generator. */
final class PythonRandom {
  // MT19937 state and current position.
  private final int[] mt = new int[624];
  private int index = 624;

  public PythonRandom(long seed) {
    seed(seed);
  }

  /** Initializes the state using CPython's integer seeding rules. */
  public void seed(long seed) {
    long x = Math.abs(seed);
    int[] key;
    if (x == 0) key = new int[] {0};
    else {
      ArrayList<Integer> a = new ArrayList<>();
      while (x != 0) {
        a.add((int) x);
        x >>>= 32;
      }
      key = a.stream().mapToInt(i -> i).toArray();
    }
    initByArray(key);
  }

  /** Initializes the base MT19937 state. */
  private void initGen(int s) {
    mt[0] = s;
    for (int i = 1; i < 624; i++)
      mt[i] = (int) (1812433253L * ((mt[i - 1] ^ (mt[i - 1] >>> 30)) & 0xffffffffL) + i);
    index = 624;
  }

  /** Expands the state from the seed words. */
  private void initByArray(int[] key) {
    initGen(19650218);
    int i = 1, j = 0, k = Math.max(624, key.length);
    for (; k > 0; k--) {
      mt[i] =
          (int)
              (((mt[i] ^ (((mt[i - 1] ^ (mt[i - 1] >>> 30)) & 0xffffffffL) * 1664525L))
                      + key[j]
                      + j)
                  & 0xffffffffL);
      i++;
      j++;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (k = 623; k > 0; k--) {
      mt[i] =
          (int)
              (((mt[i] ^ (((mt[i - 1] ^ (mt[i - 1] >>> 30)) & 0xffffffffL) * 1566083941L)) - i)
                  & 0xffffffffL);
      i++;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
    }
    mt[0] = 0x80000000;
  }

  /** Returns the next 32 random bits. */
  private int nextInt() {
    if (index >= 624) {
      for (int i = 0; i < 624; i++) {
        int y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) == 0 ? 0 : 0x9908b0df);
      }
      index = 0;
    }
    int y = mt[index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y;
  }

  /** Returns a value in [0, 1) using CPython's 53-bit construction. */
  public double random() {
    long a = (nextInt() >>> 5) & 0x7ffffffL, b = (nextInt() >>> 6) & 0x3ffffffL;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  /** Returns an integer containing k random bits. */
  private long getrandbits(int k) {
    if (k <= 0) return 0;
    if (k <= 32) return Integer.toUnsignedLong(nextInt()) >>> (32 - k);
    throw new IllegalArgumentException("k>32");
  }

  /** Returns an unbiased integer in [0, stop). */
  public int randrange(int stop) {
    if (stop <= 0) throw new IllegalArgumentException();
    int k = 32 - Integer.numberOfLeadingZeros(stop);
    long r;
    do {
      r = getrandbits(k);
    } while (r >= stop);
    return (int) r;
  }

  /** Returns an integer in [a, b]. */
  public int randint(int a, int b) {
    return a + randrange(b - a + 1);
  }

  /** Returns a floating-point value between a and b. */
  public double uniform(double a, double b) {
    return a + (b - a) * random();
  }

  /** Chooses an element from a non-empty list. */
  public <T> T choice(List<T> x) {
    return x.get(randrange(x.size()));
  }

  /** Shuffles a list in place. */
  public <T> void shuffle(List<T> x) {
    for (int i = x.size() - 1; i > 0; i--) {
      int j = randrange(i + 1);
      Collections.swap(x, i, j);
    }
  }

  /** Samples k distinct elements using CPython's call pattern. */
  public <T> List<T> sample(List<T> pop, int k) {
    int n = pop.size();
    if (k < 0 || k > n) throw new IllegalArgumentException("sample larger than population");
    List<T> out = new ArrayList<>(k);
    int setsize = 21;
    if (k > 5) {
      int power = 1, target = k * 3;
      while (power < target) power *= 4;
      setsize += power;
    }
    if (n <= setsize) {
      List<T> pool = new ArrayList<>(pop);
      for (int i = 0; i < k; i++) {
        int j = randrange(n - i);
        out.add(pool.get(j));
        pool.set(j, pool.get(n - i - 1));
      }
    } else {
      Set<Integer> selected = new HashSet<>();
      for (int i = 0; i < k; i++) {
        int j = randrange(n);
        while (selected.contains(j)) j = randrange(n);
        selected.add(j);
        out.add(pop.get(j));
      }
    }
    return out;
  }
}
