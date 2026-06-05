# Project Writeup

All tasks in the project proposal were completed. Additionally, the main stretch goal of the bracketed turtle fully on the GPU is complete, speeding up both the `plant` and `bush` examples. The cool new result is the on-device bracket resolver. We originally argued that bracketed systems break the simple linear scan and rule out per-thread stacks, and
just proposed a CPU turtle to resolve brackets. That CPU path exists only as the test oracle now, and the GPU resolves the turtle stack itself by turning the bracket structure into a tree and reducing over it.

## Architecture

- Expand: The grammar becomes a branch-free 256-entry rule table, so kernels look up
`rule_len[symbol]` with no branching. Each step is some kernels around a Thrust scan, where
`length_kernel` writes replacement lengths, `exclusive_scan` makes write offsets, and
`scatter_kernel` copies each replacement into place.

- Interpret: We fill a per-symbol world-transform array, then `copy_if` keeps the frame
of every drawn `F`. We change behavior depending on if there is a bracket or not:

    - non-bracketed, `scan_world`: Each symbol is a local `SE(3)` transform, and an
      `exclusive_scan` with `compose` turns them into world transforms (a matrix prefix scan)
    - bracketed, `scan_world_bracketed`: resolves the turtle stack on-device by treating
      brackets as a tree (details below)

Device allocations use a size-bucketed pool (`gpu/pool.cu`), and freed blocks
are reused by exact size, so per-frame allocation is a vector pop instead of a
`cudaMalloc`/`cudaFree`.

The CPU writes PPMs (`cpu/image.cpp`), while the playground draws antialiased
`GL_LINES` into a multisampled framebuffer.

## The Bracketed-System Kernels

We can think of a bracketed string is a tree, since `[` opens a branch and `]` returns to the parent. Symbol `i`'s
world transform is everything composed from the root down to `i`. Split that at `i`'s innermost bracket:

```
W[i] = entry[branch(i)] . P[i]
```

- `branch(i)`: innermost `[` enclosing `i` (`ROOT = -1` for the trunk).
- `P[i]`: Within-branch prefix, the compose-scan from that `[` up to `i`.
- `entry[b]`: The world transform at branch `b`'s `[`.

We build this in the following stages (all in `gpu/bracket_resolve.cu`):

1. `resolve_branches`: Find each symbol's enclosing `[`, where a depth scan (`[` is +1, `]` is -1)
   gives bracket depth. Then, we pack `(level, position)` keys, sort the `[` positions, and a
   `lower_bound` per symbol lands on its same-level predecessor, which is the enclosing bracket.
   A `[` is treated one level shallower so it inherits its parent.
2. `resolve_prefix`: Within-branch prefixes, Stable-sort symbols by `branch` to group
   them, `exclusive_scan_by_key` with `compose`, then scatter back to string order
3. `resolve_entries`: Push each branch's entry transform up the tree by pointer
   doubling over `parent = branch[b]`
4. `world_b_kernel`: one thread per symbol assembles
   `W[i] = compose(init, entry[branch[i]], P[i])`, and the downstream `copy_if` keeps the `F` frames.

Scratch buffers are reused, so re-renders allocate nothing!

## CPU vs GPU Performance Analysis 

`src/app/bench.cpp` times *expand*, *transform*, and the full *pipeline* for each
example, median of 5 rep. `gpu ms` is on-device cost, `e2e ms` adds the
H2D/D2H copies, `speedup` is `cpu / gpu`. Measured on an RTX A5000.

```
L-system CPU vs GPU benchmark, median of 5 reps

[expand] string rewriting
system      iters  symbols   cpu ms   gpu ms   e2e ms  speedup
------      -----    -----   ------   ------   ------  -------
koch           12   117.4M    607.0      2.9     70.7   211.5x
plant          12   103.5M    567.9      4.3     63.7   131.7x
dragon         24    67.1M    870.2      3.7     41.8   237.7x
hilbert        12    55.9M    289.5      1.9     10.3   149.7x
sierpinski     16   129.1M    926.1      3.9     79.3   238.6x

[transform] turtle interpretation
system      iters segments   cpu ms   gpu ms   e2e ms  speedup
------      -----    -----   ------   ------   ------  -------
koch           10     3.1M    309.4     31.8    113.5     9.7x
plant          11     6.3M    615.9    150.8    314.1     4.1x
dragon         22     4.2M    459.6     73.0    180.5     6.3x
hilbert        11     4.2M    380.6     60.8    171.0     6.3x
sierpinski     13     1.6M    132.2     20.9     62.9     6.3x

[pipeline] expand + interpret
system      iters segments   cpu ms   gpu ms   e2e ms  speedup
------      -----    -----   ------   ------   ------  -------
koch           10     3.1M    374.9     31.7    115.6    11.8x
plant          11     6.3M    758.0    149.9    313.4     5.1x
dragon         22     4.2M    672.1     72.7    180.8     9.2x
hilbert        11     4.2M    475.2     60.3    170.8     7.9x
sierpinski     13     1.6M    163.1     21.0     62.6     7.8x
```

We can see that *expand* is where the GPU dominates (130–240x). String length grows exponentially, so this is
the work that matters, and on-device it runs in single-digit milliseconds for ~100M symbols. End-to-end is much slower because the huge result string has to be copied back over PCIe, which is exactly the copy the device layer exists to
avoid. The turtle stage is more modest (5-12x), and the bracketed `plant` is the lowest at 5x because about half its time is the two resolver sorts, whereas `dragon` is bound by the expand kernels instead.

## Nsight Compute

I profiled `plant 11` (bracketed) and `dragon 24` (non-bracketed) per-kernel with Nsight
Compute, and we noted that the bottleneck depends on the system. For `plant`, the bracket
resolver is 84% of the total, and the two radix sorts (`brk-branches` at 44.6% and
`brk-prefix` at 38.9%) account for 50.6% on their own. For `dragon` though, the expand step
dominates at 88.4% while the transform scan is only 8%. So plants are bound by the resolver's
sorts and linear curves by expansion.

A full bandwidth pass is spent just to pick a path. `count('[')` scans the whole string
(88.6% DRAM) only to choose between the bracketed and flat routines, which could be folded
into the depth or expand pass. As expected, the bracket path is sort-bound (50.6%) and the
doubling and world-assembly kernels are negligible at under 1% each.

## Potential Improvements

The biggest issue here is the bracketed turtle, its two sorts are ~50% of `plant`, and since
the keys are `(depth, position)` with small depths, a counting or segmented sort would beat a
general radix sort. I'm not entirely sure what would be the best approach in the future.
The obvious next feature is wiring CUDA/OpenGL interop end-to-end so device
frames feed instanced drawing with no host round-trip, which the device layer already supports.
This would improve the performance of the playground.
