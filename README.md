# Roots of High-degree Polynomials Visualisation

https://github.com/user-attachments/assets/4ea02bf0-57c1-4661-a799-bf9e19f520c1

Visualisations of complex roots of n-th degree polynomials written in Python with Jax and Matplotlib.

This is inspired by:
- Work by [Simone Conradi](https://x.com/S_Conradi) which made me interested in this mathematical visualisation in the first place.
- [The Beauty of Roots](https://math.ucr.edu/home/baez/roots/) by John Baez that made the reasoning click.

I make no guarantees about the correctness or efficiency of this implementation, as this was mostly just an experiment for me to learn how the visualisations I had seen work! I may rewrite this in a "proper" graphics API in the future. Expect sharp edges too, in particular the equation parser is pretty fragile! 😬

### Installation

Simply create a python environment via your preferred method, then run `pip install -r requirements.txt`

### Explanation

My explanation of what is going on:
- Take a function `f(tf1, ..., tv1, ...)` that generates a `n`-th degree polynomial, with coefficients that are a function of fixed, potentially complex scalars `tf1, tf2, ...` and similarly complex scalars that are intended to be varied `tv1, tv2, ...`
- Randomly generate a large number `N` of "fixed" scalars (these represent the particles in our system)
- Define a function that takes in the current frame index and returns `tv1, tv2, ...`.
- For each frame index calculate the varying scalars and solve all polynomials from `f` that are generated by all the randomly sampled fixed scalars.
- This yields at most `n` complex roots for each of the `N` particles, which can have real components plotted on the x-axis and imaginary components on the y-axis.
- Repeating this process for each frame yields the animation.

Colouring the animation could be implemented in many ways, but so far we just support simple colour-mapping (aka Matplotlib's `cmap`) and another method which sums the fixed scalars at a given point, computes the angle of the resulting complex number, and uses that to determine the hue at a given point.

The root finding is implemented in Jax and makes uses of the `shard_map` (or "shmap") parallelism API for calculating roots across many processes. Unfortunately, certain operations are not supported on Jax with a GPU backend, so we do all computation on CPU.

### Usage

`main.py` is the main entrypoint.
```
usage: main.py [-h] [-N N] [--threads THREADS] [--framerate FRAMERATE] [--length LENGTH] [--seed SEED] [--output-path OUTPUT_PATH]
               [--disable-cache] [--equation EQUATION] [--varying-indices VARYING_INDICES [VARYING_INDICES ...]]
               [--fixed-indices [FIXED_INDICES ...]] [--resolution RESOLUTION] [--max-filter-size MAX_FILTER_SIZE]
               [--color-mode {cmap,t_fixed_hue}] [--cmap CMAP] [--preset-equation PRESET_EQUATION]

options:
  -h, --help            show this help message and exit
  -N N                  number of particles (default: 100000)
  --threads THREADS     number of threads (default: 10)
  --framerate FRAMERATE
                        frames per second (default: 24)
  --length LENGTH       number of frames (default: 5.0)
  --seed SEED           random seed (default: 12648430)
  --output-path OUTPUT_PATH
                        output path (default: animation-{date}.{time}.mp4)
  --disable-cache       disable compilation cache (default: False)
  --equation EQUATION   string representation of equation to plot.
  --varying-indices VARYING_INDICES [VARYING_INDICES ...]
  --fixed-indices [FIXED_INDICES ...]
  --resolution RESOLUTION
                        resolution of output animation. (default: 1000)
  --max-filter-size MAX_FILTER_SIZE
                        maximum filter size (default: 3)
  --color-mode {cmap,t_fixed_hue}
                        color mode to use in output animation (default: t_fixed_hue)
  --cmap CMAP           matplotlib colormap to use if color_mode is `cmap` (default: gray)
  --preset-equation PRESET_EQUATION
                        use a zero-indexed preset equation, see preset_equations.py (default: None)
```

Most of the above is explained by the above comments, however the format for custom equations is somewhat complex and fragile.
- The equation is a space separated list of terms which must consist of:
    - the current degree, specified by `x^n` where `x` is literally `x` and `n` is the degree
    - optionally, a leading set of space separated coefficients enclosed in brackets
    - optionally only if not the first term, a leading `+` or `-`.
- The space separated coefficients are basically a summation of terms, where each term has:
    - optionally only if not the first term, a leading `+` or `-`
    - an optional numeric scalar
    - an optional `j` to indicate a complex scalar
    - an optional `[n]` indicating a (time) varying or fixed (randomly generated) coefficient, which may also be optionally raised to a power with `^p` where `p` is the power. The `n` must be an integer which will be used later to specify if it is fixed or varying.
- In the command line, specify a list of space separated integers to `--fixed-indices` and `--varying-indices` to specify which `[n]` was fixed and which were varying.

See `preset_equations.py` for examples, which can be selected by passing `--preset-equation` into `main.py`. The parser is not very high-effort, so I'd recommend sticking to the format in this file.

`main-numpy.py` is a numpy-only version used for early experimentation, which is more concise and easier to understand, but less flexible for general use.
