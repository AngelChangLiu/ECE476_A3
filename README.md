# Assignment 3: A Simple CUDA Renderer

**100 points total**

![Assignment teaser](handout/teaser.jpg?raw=true)

## Overview

In this assignment, you will complete three CUDA tasks:

1. `saxpy`: implement GPU SAXPY in `saxpy/saxpy.cu`
2. `scan`: implement `exclusive_scan` and `find_repeats` in `scan/scan.cu`
3. `render`: replace the incorrect CUDA renderer in `render/cudaRenderer.cu`

This assignment is both conceptually challenging and performance-sensitive. **Start early!**

## Files You Should Expect To Edit

- Part 1: `saxpy/saxpy.cu`
- Part 2: `scan/scan.cu`
- Part 3: `render/cudaRenderer.cu`

You may edit other files for debugging, but only changes in the required CUDA files (**`*.cu`**) are being graded.

## Environment Setup

### 1. Login Node

Use the visualization/login node with GPU:

```shell
ssh <username>@adroit-vis.princeton.edu
```

`adroit-vis` is a good place to edit, build, and do light debugging, but it is a shared machine. For actual CUDA execution, especially performance testing, submit the job to a GPU compute node.

### 2. Clone Your Private Repository

Create a private copy of the release repository:

- Go to GitHub import page: <https://github.com/new/import>
- Import from: <https://github.com/princeton-ece476/assignment3>
- Set the new repository to **Private**
- Add your partner in GitHub repository settings under: *Settings → Collaborators*

Then clone your repository and add the release repo as a remote:

```shell
git clone git@github.com:<your-github-name>/<your-repo-name>.git
cd <your-repo-name>
git remote add release https://github.com/princeton-ece476/assignment3
```

### 3. Load the Course Environment

From the project root:

```shell
source ./setup.sh
```

This script must be **sourced**, not executed. It loads:

- `gcc/15.2.0`
- `cudatoolkit/13.0`
- `freeglut/3.6.0`
- the helper command `ece476-srun`

If you are inside a subdirectory, use:

```shell
source ../setup.sh
```

### 4. Request a GPU with Slurm

Any command that actually tests CUDA code for performance, should be launched with `ece476-srun` or an equivalent `srun` command. Default `ece476-srun` settings come from `setup.sh`:

```shell
-p gpu --gres=gpu:3g.20gb:1 --time=00:15:00 --cpus-per-task=6 --mem=8G
```

Examples:

```shell
# Recommended wrapper
ece476-srun <command>

# Override defaults if needed
ece476-srun --mem=12G <command>
ece476-srun --time=00:02:00 nvidia-smi -L

# Equivalent raw srun command
srun -p gpu --gres=gpu:3g.20gb:1 --time=00:15:00 --cpus-per-task=6 --mem=8G <command>
```

Notes:

- `3g.20gb` is an A100 MIG partition.
- Queue delays are normal. If a job sits in `queued and waiting for resources`, wait for allocation.
- Build with `make` on `adroit-vis` or `adroit` should be fine. Do **not** run `make` in `srun`.

## Top-Level Commands

From the project root:

```shell
make build
make clean
make rebuild
```

What they do:

- `make build`: build `saxpy`, `scan`, and `render`
- `make clean`: remove generated build products
- `make rebuild`: clean, then build again

> [!TIP]
> If you encounter runtime errors, try a clean rebuild:
>
> ```shell
> make clean
> make
> ```

## Relevant Readings

Use official NVIDIA documentation first:

- CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- CUDA intro and explicit device memory management: <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html>
- CUDA asynchronous execution: <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>
- Compute capability tables: <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- Thrust documentation: <https://nvidia.github.io/cccl/thrust/api/namespace_thrust.html>
- NVIDIA A100 product page: <https://www.nvidia.com/en-us/data-center/a100/>

## Part 1: CUDA Warm-Up 1 - SAXPY (5 pts)

This warm-up helps you get familiar with basic CUDA programming by re-implementing `SAXPY (Y = αX + Y)` on the GPU.
Starter code is in the `/saxpy` directory.

Relevant files:

- `saxpy/saxpy.cu`
- `saxpy/main.cpp`

### Build and Run Saxpy

```shell
cd saxpy
make

# Test on adroit-vis
./cudaSaxpy
# Run on a GPU node
ece476-srun ./cudaSaxpy

# Optional: change input size
ece476-srun ./cudaSaxpy -n 1000000

# Equivalent make commands
make run
make run ARGS="-n 1000000"
```

### What You Need To Implement

Complete `saxpyCuda(...)` in [`saxpy/saxpy.cu`](saxpy/saxpy.cu). Your implementation should:

1. Allocate device (GPU) memory for input/output arrays
2. Copy data from host (CPU) → device (GPU)
3. Launch the CUDA kernel
4. Copy results back from device → host
5. Free device memory

### Timing Requirement

Measure performance in two ways:

1. End-to-End Time (Provided)
  Already implemented in the starter code: includes memory transfers + kernel execution

2. Kernel-Only Time (You Implement)
  Add timing around only the kernel execution
  You might find that the measured execution time seems amazingly fast! Because CUDA kernel launches are **asynchronous** and you are only timing the cost of the API call itself, not the cost of the actual API executing time on the GPU.

```cpp
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(...);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

> [!TIP]
> **Why `cudaDeviceSynchronize()` matters:**
>
> - CUDA kernel launches are asynchronous.
> - Without synchronization, you may only measure launch overhead, not actual GPU work.

### Questions To Answer

> [!NOTE]
> **Question 1**
>
> - Compare performance with your CPU SAXPY implementation from Assignment 1.
> - How much speedup (or slowdown) do you observe? Briefly explain why.

> [!NOTE]
> **Question 2**  
> **Compare:**
>
> - Kernel-only time
> - End-to-end time (including data transfer)
>
> **Explain:**
>
> - Is the GPU computation itself fast?
> - How significant is the data transfer overhead?
>
> **Relate to hardware limits:**
>
> - Look up the A100 memory bandwidth
> - Compare with CPU ↔ GPU transfer bandwidth (PCIe)
> You do not need exact matches, but your results should be roughly consistent with expected bandwidth constraints.

## Part 2: CUDA Warm-Up 2 - Parallel Prefix-Sum (10 pts)

In this part, you will implement:

- `exclusive_scan`
- `find_repeats`

Both functions live in `scan/scan.cu`.

Relevant files:

- `scan/scan.cu`
- `scan/main.cpp`
- `scan/checker.py`
- `scan/cudaScan_ref`

### Problem: `find_repeats`

Given an integer array `A`, `find_repeats` returns all indices `i` such that `A[i] == A[i+1]`.

Example:

```text
Input:  {1,2,2,1,1,1,3,5,3,3}
Output: {1,3,4,8}
Explain: A[1] == A[2], A[3] == A[4], A[4] == A[5], A[8] == A[9]
```

### Step 1: Implement `exclusive_scan`

You will first implement a parallel exclusive prefix sum (scan), which is the key building block for this part. An **exclusive scan** produces an output array where each element is the sum of all previous elements:

```python
Input:  {1,4,6,8,2}
Output: {0,1,5,11,19}
Explain: output[i] = output[i - 1] + input[i - 1], output[0] = 0
```

We provide a reference iterative algorithm (upsweep + downsweep):

```cpp
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

**Your Task:** Implement `exclusive_scan` in `scan/scan.cu` using CUDA.

- You will need both host and device code
- Each `parallel_for` corresponds to a CUDA kernel launch
- The implementation involves multiple kernel launches (one per iteration)

**Note:** In the starter code, the reference solution scan implementation above assumes that the input array's length (`N`) is a power of 2. In the `cudaScan` function, we solve this problem by rounding the input array length to the next power of 2 when allocating the corresponding buffers on the GPU. However, the code only copies back `N` elements from the GPU buffer back to the CPU buffer. This fact should simplify your CUDA implementation.

### Step 2: Implement `find_repeats`

**Your Task:** Implement `find_repeats` in `scan/scan.cu`.
Your implementation should:

- Identify positions where `A[i] == A[i+1]`
- Write matching indices to an output array (in device memory)

A standard approach is:

1. Build a 0/1 flag array
2. Run `exclusive_scan` on that flag array
3. Scatter the matching indices into the output array
4. Return the number of matches

### Build and Run find_repeats

```shell
cd scan
make

# Run scan
./cudaScan -m scan -n 200 # on adroit-vis
ece476-srun ./cudaScan -m scan -n 200 # run on a GPU node
make run ARGS="-m scan -n 200" # same effect as above

# Run find_repeats
./cudaScan -m find_repeats -n 200 # on adroit-vis
ece476-srun ./cudaScan -m find_repeats -n 200 # run on a GPU node
make run ARGS="-m find_repeats -n 200" # same effect as above
```

Useful debug cases:

```shell
./cudaScan -m scan -i ones -n 16
./cudaScan -m find_repeats -i ones -n 16
```

Note: a tiny random `find_repeats` run can pass by luck. Use `-i ones` and the checker when debugging.

### Checkers

Use python checker for performance testing (with srun)

```shell
ece476-srun python3 checker.py scan
make check_scan # same effect as above

ece476-srun python3 checker.py find_repeats
make check_find_repeats # same effect as above
```

We evaluate both correctness and performance. Full credit requires performance within 20% of the reference solution

Sample output for scoring (after running `python3 checker.py scan` and `python3 checker.py find_repeats`)

```shell
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

This part of the assignment is largely about getting more practice with writing CUDA and thinking in a data parallel manner, and not about performance tuning code. Getting full performance points on this part of the assignment should not require much (or really any) performance tuning, just a direct port of the algorithm pseudocode to CUDA. However, there's one trick: a naive implementation of scan might launch N CUDA threads for each iteration of the parallel loops in the pseudocode, and using conditional execution in the kernel to determine which threads actually need to do work. Such a solution will not be performant! (Consider the last outer-most loop iteration of the upsweep phase, where only two threads would do work!). A full credit solution will only launch one CUDA thread for each iteration of the innermost parallel loops.

**Test Harness:** By default, the test harness runs on a pseudo-randomly generated array that is the same every time
the program is run, in order to aid in debugging. You can pass the argument `-i random` to run on a random array - we
will do this when grading. We encourage you to come up with alternate inputs to your program to help you evaluate it.
You can also use the `-n <size>` option to change the length of the input array.

The argument `--thrust` will use the [Thrust Library's](http://thrust.github.io/) implementation of [exclusive scan](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums). **Up to two points of extra credit for anyone that can create an implementation is competitive with Thrust.**

## Part 3: A Simple Circle Renderer (85 pts)

This is the **main** part of the assignment.

Relevant files:

- `render/cudaRenderer.cu`
- `render/refRenderer.cpp`
- `render/main.cpp`
- `render/checker.py`
- `render/circleBoxTest.cu_inl`
- `render/exclusiveScan.cu_inl`
- `render/render_ref`
- `render/render_ref_super`

The starter code contains:

- a correct sequential CPU renderer in `render/refRenderer.cpp`
- an incorrect CUDA renderer in `render/cudaRenderer.cu`

The directory `/render` of the assignment starter code contains an implementation of renderer that draws colored circles. Build the code, and run the render with the following command line: `./render -r cpuref rgb`. The program will output an image `output_0000.ppm` containing three circles. Now run the renderer with the command line `./render -r cpuref snow`. Now the output image will be falling snow. If you use VSCode, an extension is available: <https://marketplace.visualstudio.com/items?itemName=ngtystr.ppm-pgm-viewer-for-vscode>

### Build Options

```shell
cd render
make
```

`render/Makefile` supports:

- `make`: build with `USE_OPENGL=auto`
- `make USE_OPENGL=1`: force OpenGL/interactive support
- `make USE_OPENGL=0`: build a headless binary

(Optional) Note: you can also use the `-i` option to send renderer output to the display instead of a file, if you build with `USE_OPENGL=1` (or auto). (In the case of snow, you'll see an animation of falling snow.) On Princeton systems, OpenGL/GLUT development headers are available on the `-vis` nodes. To run interactive mode successfully, you'll still need X11 forwarding to your local machine. ([This reference](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) or [this reference](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) may help.)

### Renderer Overview

We encourage you to familiarize yourself with the structure of the renderer codebase by inspecting the reference implementation in `refRenderer.cpp`. The method `setup` is called prior to rendering the first frame. In your CUDA-accelerated renderer, this method will likely contain all your renderer initialization code (allocating buffers, etc). `render` is called each frame and is responsible for drawing all circles into the output image. The other main function of the renderer, `advanceAnimation`, is also invoked once per frame. It updates circle positions and velocities.
You will not need to modify `advanceAnimation` in this assignment.
The renderer accepts an array of circles (3D position, velocity, radius, color) as input. The basic sequential algorithm for rendering each frame is:

   ```python
   Clear image
   for each circle
      update position and velocity
   for each circle
      compute screen bounding box
      for all pixels in bounding box
         compute pixel center point
         if center point is within the circle
             compute color of circle at point
             blend contribution of circle into image for this pixel
   ```

The figure below illustrates the basic algorithm for computing circle-pixel coverage using point-in-circle tests. Notice that a circle contributes color to an output pixel only if the pixel's center lies within the circle.

![Point in circle test](handout/point_in_circle.jpg?raw=true "A simple algorithm for computing the contribution of a circle to the output image: All pixels within the circle's bounding box are tested for coverage. For each pixel in the bounding box, the pixel is considered to be covered by the circle if its center point (black dots) is contained within the circle. Pixel centers that are inside the circle are colored red. The circle's contribution to the image will be computed only for covered pixels.")

An important detail of the renderer is that it renders **semi-transparent** circles. Therefore, the color of any one pixel is not the color of a single circle, but the result of blending the contributions of all the semi-transparent circles overlapping the pixel (note the "blend contribution" part of the pseudocode above). The renderer represents the color of a circle via a 4-tuple of red (R), green (G), blue (B), and opacity (alpha) values (RGBA). Alpha = 1 corresponds to a fully opaque circle. Alpha = 0 corresponds to a fully transparent circle. To draw a semi-transparent circle with color `(C_r, C_g, C_b, C_alpha)` on top of a pixel with color `(P_r, P_g, P_b)`, the renderer uses the following math:

```cpp
result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
```

Notice that composition is not commutative (object X over Y does not look the same as object Y over X), so it's important that the render draw circles in a manner that follows the order they are provided by the application. (You can assume the application provides the circles in depth order.) For example, consider the two images below where a blue circle is drawn OVER a green circle which is drawn OVER a red circle. In the image on the left, the circles are drawn into the output image in the correct order. In the image on the right, the circles are drawn in a different order, and the output image does not look correct.

![Ordering](handout/order.jpg?raw=true "The renderer must be careful to generate output that is the same as what is generated when sequentially drawing all circles in the order provided by the application.")

### CUDA Renderer

After familiarizing yourself with the circle rendering algorithm as implemented in the reference code, now study the CUDA implementation of the renderer provided in `cudaRenderer.cu`. You can run the CUDA implementation of the renderer using the `--renderer cuda (or -r cuda)` cuda program option.
The provided CUDA implementation parallelizes computation across all input circles, assigning one circle to each CUDA thread. While this CUDA implementation is a complete implementation of the mathematics of a circle renderer, it contains several major errors that you will fix in this assignment. Specifically: the current implementation does not ensure image update is an atomic operation and it does not preserve the required order of image updates (the ordering requirement will be described below).

### What Is Wrong with the Starter CUDA Renderer?

The starter CUDA renderer violates two correctness requirements:

1. **Atomicity**
   Updating a pixel is a read-blend-write operation on RGBA values. That operation must be logically atomic.
2. **Order**
   If two circles affect the same pixel, their contributions must be applied in increasing circle index order.

The order rule matters because the circles are semi-transparent, and alpha blending is not commutative.

![Dependencies](handout/dependencies.jpg?raw=true "The contributions of circles 1, 2, and 3 must be applied to overlapped pixels in the order the circles are provided to the renderer.")

Since the provided CUDA implementation does not satisfy either of these requirements, the result of not correctly respecting order or atomicity can be seen by running the CUDA renderer implementation on the rgb and circles scenes. You will see horizontal streaks through the resulting images, as shown below. These streaks will change with each frame.

![Order_errors](handout/bug_example.jpg?raw=true "Errors in the output due to lack of atomicity in frame-buffer update (notice streaks in bottom of image).")

### What You Need To Do

**Your job is to write the fastest, correct CUDA renderer implementation you can**. You may take any approach you see fit, but your renderer must adhere to the atomicity and order requirements specified above. A solution that does not meet both requirements will be given no more than 12 points on part 3 of the assignment. We have already given you such a solution!

A good place to start would be to read through `cudaRenderer.cu` and convince yourself that it *does not* meet the correctness requirement. In particular, look at how `CudaRenderer:render` launches the CUDA kernel `kernelRenderCircles`. (`kernelRenderCircles` is where all the work happens.) To visually see the effect of violation of above two requirements, compile the program with `make` (or `make USE_OPENGL=0` on nodes without GLUT headers). Then run `./render -r cuda rand10k"` which should generate a PPM image with 10K circles, shown in the bottom row of the image above. Compare this (incorrect) image with the image generated by sequential code by running `./render -r cpuref rand10k"`.

We recommend that you:

1. First rewrite the CUDA starter code implementation so that it is logically correct when running in parallel (we recommend an approach that does not require locks or synchronization)
2. Then determine what performance problem is with your solution.
3. At this point the real thinking on the assignment begins... (Hint: the circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.)

### Part 3 Implementation Checklist (What To Do)

1. Read `render/refRenderer.cpp` and `render/cudaRenderer.cu` end-to-end before changing code. Confirm where ordering and atomicity are violated in `kernelRenderCircles`.
2. Reproduce the bug first:
   - `make clean` then run `./render -r cuda rand10k -f cuda_rand10k`, check figure `cuda_rand10k_0000.ppm`
   - Compare with `./render -r cpuref rand10k -f cpu_rand10k` with figure `cpu_rand10k_0000.ppm`
   - `-f` is optional to set the prefix of figure file names, so we can keep outputs from multiple runs and compare.
3. Redesign the CUDA work decomposition so a pixel's blending sequence is deterministic and follows circle input order.
4. Enforce both correctness invariants for every pixel:
   - Atomicity of read-blend-write for RGBA updates.
   - Strict application of contributing circles in increasing circle index order.
5. Use `circleBoxTest.cu_inl` to cull circles that cannot affect a tile/pixel before shading work.
6. Tune after correctness passes:
   - Choose block/tile sizes carefully.
   - Think of how to arrange computations more efficiently. (think of previous parts)
   - Re-check correctness after every optimization.

### Part 3 Testing Checklist (How To Test)

Run these from `/render`:

1. Build:
   - `make` (or `make USE_OPENGL=0` on headless nodes)
2. Quick visual sanity checks:
   - `./render -r cpuref rgb -f cpu_rgb`
   - `./render -r cuda rgb -f cuda_rgb`
   - `./render -r cpuref rand10k -f cpu_rand10k`
   - `./render -r cuda rand10k -f cuda_rand10k`
3. Per-scene correctness checks with CPU reference (use GPU compute note for tests beyond `rgb` and `rand10k`)
   - `./render -r cuda -c rgb -f cuda_rgb`
   - `./render -r cuda -c rand10k -f cuda_rand10k`
   - `ece476-srun ./render -r cuda -c rand100k -f cuda_rand100k`
   - `ece476-srun ./render -r cuda -c pattern -f cuda_pattern`
   - `ece476-srun ./render -r cuda -c snowsingle -f cuda_snowsingle`
   - `ece476-srun ./render -r cuda -c biglittle -f cuda_biglittle`
   - `ece476-srun ./render -r cuda -c rand1M -f cuda_rand1M`
   - `ece476-srun ./render -r cuda -c micro2M -f cuda_micro2M`
4. Full grading-style run:
   - `ece476-srun python3 checker.py` or `make check`
5. Optional stability/perf checks before submission:
   - Repeat `./checker.py` a few more times or increase the `test_times=3` to a large number in the python to catch nondeterministic failures.
   - Use bench frames for timing trends, e.g. `./render -r cuda -b 0:10 rand100k`.
6, Extra credit for performance
   - `ece476-srun python3 checker.py --super` or `make check_super`

Following are commandline options to `./render`:

```shell
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -S  --Seed  <INT>             Random seed for scene generation (default=0)
  -?  --help                    This message
```

In a headless (`USE_OPENGL=0`) build, the `-i/--interactive` option is unavailable by design.

**Checker code:** To detect correctness of the program, `render` has a convenient `--check` option. This option runs the sequential version of the reference CPU renderer along with your CUDA renderer and then compares the resulting images to ensure correctness. The time taken by your CUDA renderer implementation is also printed.

We provide a total of eight circle datasets you will be graded on. However, in order to receive full credit, your code must pass all of our correctness-tests. To check the correctness and performance score of your code, run **`python3 ./checker.py`** (notice the .py extension) in the `/render` directory. If you run it on the starter code, the program will print a table like the following, along with the results of our entire test set:

```shell
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

Note: on some runs, you *may* receive credit for some of these scenes, since the provided renderer's runtime is non-deterministic sometimes it might be correct. This doesn't change the fact that the current CUDA renderer is in general incorrect.

"Ref time" is the performance of our reference solution on your current machine (in the provided `render_ref` executable). "Your time" is the performance of your current CUDA renderer solution, where an `(F)` indicates an incorrect solution. Your grade will depend on the performance of your implementation compared to these reference implementations (see Grading Guidelines).

> [!NOTE]
> **Writing**  
> Along with your code, we would like you to hand in a clear, high-level description of how your implementation works as well as a brief description of how you arrived at this solution. Specifically address approaches you tried along the way, and how you went about determining how to optimize your code (For example, what measurements did you perform to guide your optimization efforts?).
>
> Aspects of your work that you should mention in the write-up include:
>
> 1. Include both partners' names and NetIDs at the top of your write-up.
> 2. Replicate the score table generated for your solution and specify which machine you ran your code on.
> 3. Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
> 4. Describe where synchronization occurs in your solution.
> 5. What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
> 6. Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?

### Grading Guidelines

- The write-up for the assignment is worth 18 points.
- Your parallel prefix implementation is worth 10 points.
- Your render implementation is worth 72 points. These are equally divided into 9 points per scene as follows:
  - 2 correctness points per scene. We will only test your program with image sizes that are multiples of 256.
  - 7 performance points per scene (only obtainable if the solution is correct). Your performance will be graded with respect to the performance of a provided benchmark reference renderer, T<sub>ref</sub>:
    - No performance points will be given for solutions having time (T) 10 times the magnitude of T<sub>ref</sub>.
    - Full performance points will be given for solutions within 20% of the optimized solution ( T <= 1.20 \* T<sub>ref</sub> )
    - For other values of T (for 1.20 T<sub>ref</sub> < T < 10 _ T<sub>ref</sub>), your performance score on a scale 1 to 7 will be calculated as: `7 _ T_ref / T`.

- Up to five points extra credit: For the score you get with `python3 checker.py --super`, one point each is awarded if your program reaches `[68, 69, 70, 71, 72]` points. **Your write-up must clearly explain your approach thoroughly.**

So the total points for this project is as follows:

- part 1 (5 points)
- part 2 (10 points)
- part 3 write up (13 points)
- part 3 implementation (72 points)
- potential **extra** credit (up to 5 points)

## Assignment Tips and Hints

Below are a set of tips and hints compiled from previous years. Note that there are various ways to implement your renderer and not all hints may apply to your approach.

- There are two potential axes of parallelism in this assignment. One axis is *parallelism across pixels* another is *parallelism across circles* (provided the ordering requirement is respected for overlapping circles). Solutions will need to exploit both types of parallelism, potentially at different parts of the computation.
- The circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.
- The shared-memory prefix-sum operation provided in `exclusiveScan.cu_inl` may be valuable to you on this assignment (not all solutions may choose to use it). See the simple description of a [prefix-sum](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums). We have provided an implementation of an exclusive prefix-sum on a **power-of-two-sized** arrays in shared memory. **The provided code does not work on non-power-of-two inputs and IT ALSO REQUIRES THAT THE NUMBER OF THREADS IN THE THREAD BLOCK BE THE SIZE OF THE ARRAY. PLEASE READ THE COMMENTS IN THE CODE.**
- Take a look at the `shadePixel` method that is being called. Notice how it is doing many global memory operations to update the color of a pixel. It might be wise to instead use a local accumulator in your `kernelRenderCircles` method. You can then perform the accumulation of a pixel value in a register, and once the final pixel value is accumulated you can then just perform a single write to global memory.
- You are allowed to use the [Thrust library](http://thrust.github.io/) in your implementation if you so choose. Thrust is not necessary to achieve the performance of the optimized CUDA reference implementations. There is one popular way of solving the problem that uses the shared memory prefix-sum implementation that we give you. There another popular way that uses the prefix-sum routines in the Thrust library. Both are valid solution strategies.
- Is there data reuse in the renderer? What can be done to exploit this reuse?
- How will you ensure atomicity of image update since there is no CUDA language primitive that performs the logic of the image update operation atomically? Constructing a lock out of global memory atomic operations is one solution, but keep in mind that even if your image update is atomic, the updates must be performed in the required order. **We suggest that you think about ensuring order in your parallel solution first, and only then consider the atomicity problem (if it still exists at all) in your solution.**
- For the tests which contain a larger number of circles - `rand1M` and `micro2M` - you should be careful about allocating temporary structures in global memory. If you allocate too much global memory, you will have used up all the memory on the device. If you are not checking the `cudaError_t` value that is returned from a call to `cudaMalloc`, then the program will still execute but you will not know that you ran out of device memory. Instead, you will fail the correctness check because you were not able to make your temporary structures. This is why we suggest you to use the CUDA API call wrapper below so you can wrap your `cudaMalloc` calls and produce an error when you run out of device memory.
- If you find yourself with free time, have fun making your own scenes!

### Catching CUDA Errors

By default, if you access an array out of bounds, allocate too much memory, or otherwise cause an error, CUDA won't normally inform you; instead it will just fail silently and return an error code. You can use the following macro (feel free to modify it) to wrap CUDA calls:

```cpp
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

Note that you can undefine DEBUG to disable error checking once your code is correct for improved performance.

You can then wrap CUDA API calls to process their returned errors as such:

```cpp
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

Note that you can't wrap kernel launches directly. Instead, their errors will be caught on the next CUDA call you wrap:

```cpp
kernel<<<1,1>>>(a); // suppose kernel causes an error!
cudaCheckError( cudaDeviceSynchronize() ); // error is printed on this line
```

All CUDA API functions, `cudaDeviceSynchronize`, `cudaMemcpy`, `cudaMemset`, and so on can be wrapped.

**IMPORTANT:** if a CUDA function error'd previously, but wasn't caught, that error will show up in the next error check, even if that wraps a different function. For example:

```shell
...
line 742: cudaMalloc(&a, -1); // executes, then continues
line 743: cudaCheckError(cudaMemcpy(a,b)); // prints "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

Therefore, while debugging, it's recommended that you wrap **all** CUDA API calls (at least in code that you wrote).

(Credit: adapted from [this Stack Overflow post](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api))

## 3.4 Hand-in Instructions

Please submit your work using Gradescope. If you are working with a partner please remember to tag your partner on gradescope.

1. **Please submit your writeup as the file `writeup.pdf`.**
2. **Download ZIP from GitHub or submit on GradeScope directly.**

Our grading scripts will rerun the checker code allowing us to verify your score matches what you submitted in the `writeup.pdf`. We might also try to run your code on other datasets to further examine its correctness.
