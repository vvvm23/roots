import argparse
import math
import os
import time
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import tqdm
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from equation_parser import string_to_equation


def complex_at_angle(x):
    if isinstance(x, np.ndarray):
        return (np.cos(x) + 1j * np.sin(x)).astype(np.complex64)
    if isinstance(x, jnp.ndarray):
        return (jnp.cos(x) + 1j * jnp.sin(x)).astype(jnp.complex64)

    return math.cos(x) + 1j * math.sin(x)


# TODO: can this be... faster?
def jax_polyroots(x):
    n = len(x) - 1
    mat = jnp.diag(jnp.ones((n - 1,), dtype=x.dtype), k=-1)
    mat = mat.at[:, -1].subtract(x[:-1] / x[-1])
    r = jnp.linalg.eigvals(mat[::-1, ::-1])
    return jnp.sort(r)


def roots_one_frame(
    coefficients: jax.Array,
    coefficient_overrides_index: Sequence[int],
    coefficient_varying_index: int,
    coefficient_varying: jax.Array,
    *coefficient_overrides: Sequence[jax.Array],
):
    assert coefficient_overrides_index is not None, "specify the coefficient index to override"
    assert coefficient_varying_index is not None, "specify the coefficient index to vary"

    coefficients = coefficients.at[coefficient_varying_index].set(coefficient_varying)

    # TODO: try vmap instead?
    def scan_fn(carry, xs):
        coefficients = carry
        coefficient_overrides = xs

        for override_index, coefficient_index in enumerate(coefficient_overrides_index):
            coefficients = coefficients.at[coefficient_index].set(coefficient_overrides[override_index])

        # roots = jnp.roots(coefficients, strip_zeros=False).astype(jnp.complex64)
        roots = jax_polyroots(coefficients).astype(jnp.complex64)
        roots = jnp.reshape(roots, (-1,))
        roots = jnp.nan_to_num(roots, copy=False, nan=0.0)

        return coefficients, roots

    _, roots = jax.lax.scan(scan_fn, coefficients, coefficient_overrides, unroll=4)

    return roots


def roots_to_histogram(
    roots_real: jax.Array,
    roots_imag: jax.Array,
    bins: int = 200,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
):
    def histogram_fn(roots_real, roots_imag):
        return jnp.histogram2d(
            roots_real,
            roots_imag,
            bins=(
                jnp.linspace(x_min, x_max, bins + 1),
                jnp.linspace(y_min, y_max, bins + 1),
            ),
            range=jnp.array([[x_min, x_max], [y_min, y_max]]),
            density=False,
        )[0]

    return histogram_fn(roots_real, roots_imag)


def main(
    args: argparse.Namespace,
):
    equation = string_to_equation(args.equation)

    devices = jax.devices("cpu")
    n_devices = len(devices)
    mesh = jax.make_mesh((n_devices,), ("data",))
    n_frames = int(args.length * args.framerate)

    # degree = 11
    # TODO: can probably alloc this inside the jit function rather than outside
    # TODO: integrate equation parser
    degree = max([t.degree for t in equation.terms])
    coefficients = np.zeros((degree + 1,), dtype=np.complex64)

    rng = np.random.default_rng(args.seed)
    ts = {i: rng.random(args.N) * 2 * math.pi for i in args.fixed_indices}
    fixed_coeffs = {}
    partial_varying_coeffs = {}
    for term in equation.terms:
        if all([t.t is None for t in term.coefficient_parts]):
            # if term has a constant coefficient
            coefficients[term.degree] = term.coefficient_parts[0].constant
        elif all([t.t[0] in args.fixed_indices for t in term.coefficient_parts]):
            # if term has all fixed indices, they can be precomputed
            fixed_coeffs[term.degree] = 0  # np.zeros((args.N,), dtype=np.complex64)
            for coeff_part in term.coefficient_parts:
                if coeff_part.t is not None:
                    fixed_coeffs[term.degree] += coeff_part.constant * ts[coeff_part.t[0]] ** coeff_part.t[1]
                else:
                    fixed_coeffs[term.degree] += coeff_part.constant
        else:
            # either all varying, or a mix of varying a fixed, we can precompute some
            # TODO: we can probably merge this and the above branch
            partial_varying_coeffs[term.degree] = 0  # np.zeros((args.N,), dtype=np.complex64)
            for coeff_part in term.coefficient_parts:
                if coeff_part.t is not None and coeff_part.t[0] in args.fixed_indices:
                    partial_varying_coeffs[term.degree] += coeff_part.constant * ts[coeff_part.t[0]] ** coeff_part.t[1]
                else:
                    partial_varying_coeffs[term.degree] += coeff_part.constant

    # coefficients[11] = 1
    # coefficients[10] = -1

    # t1s = rng.random(args.N) * 2 * math.pi
    # t2s = rng.random(args.N) * 2 * math.pi

    # # project timesteps on unit circle
    # t1s = complex_at_angle(t1s)
    # t2s = complex_at_angle(t2s)

    # coeffs1 = 30j * t1s * t1s - 30 * t1s - 30
    # coeffs2 = 30j * t2s * t2s + 30j * t2s - 30

    # coeffs1 = jnp.asarray(coeffs1)
    # coeffs2 = jnp.asarray(coeffs2)

    coefficients = jnp.asarray(coefficients)
    fixed_coeffs = jax.tree_util.tree_map(jnp.asarray, fixed_coeffs)
    partial_varying_coeffs = jax.tree_util.tree_map(jnp.asarray, partial_varying_coeffs)

    jit_roots_to_histogram = jax.jit(roots_to_histogram, static_argnums=(2,), backend="cpu")

    in_specs = (
        *[None] * 4,
        P("data"),
        P("data"),
    )
    shmap_roots_one_frame = shard_map(
        roots_one_frame,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P("data"),
    )

    start_time = time.time()

    frame_roots = []
    for fi in tqdm.tqdm(range(n_frames)):
        tv = complex_at_angle(2 * math.pi * (fi / n_frames))

        coeff_varying = -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30

        # cool settings
        # original (8, 5) 6
        # (8, 2) 6
        # (8, 2) 7
        # (8, 2) 4
        roots = shmap_roots_one_frame(
            coefficients,
            (args.coeff1_index, args.coeff2_index),
            args.coeff_varying_index,
            coeff_varying,
            coeffs1,
            coeffs2,
        )

        # TODO: we filter out zeros outside jit, is there a way to do this inside?
        roots = np.asarray(roots)
        roots = roots[~np.isclose(roots, 0.0 + 0.0j, atol=1e-4, rtol=1e-4)]
        roots = np.unique(roots)
        roots = np.ascontiguousarray(roots)

        frame_roots.append(roots)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to generate all roots")

    hists = []
    x_min, x_max = np.min([np.min(roots.real) for roots in frame_roots]), np.max(
        [np.max(roots.real) for roots in frame_roots]
    )
    y_min, y_max = np.min([np.min(roots.imag) for roots in frame_roots]), np.max(
        [np.max(roots.imag) for roots in frame_roots]
    )

    # TODO: parallelise this too
    start_time = time.time()
    for roots in tqdm.tqdm(frame_roots):
        hist = jit_roots_to_histogram(
            roots.real.astype(np.float64),
            roots.imag.astype(np.float64),
            bins=args.hist_bins,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        hists.append(hist)

    hists = np.array(hists)

    hists = scipy.ndimage.maximum_filter(hists, size=(1, args.max_filter_size, args.max_filter_size))
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to generate all histograms")

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(4, 4, True)
    ax.set_axis_off()
    im = plt.imshow(hists[0], cmap=args.colourmap, origin="lower")

    def animate(i):
        im.set_data(hists[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(hists), interval=int(1000 / args.framerate), blit=False)
    FFwriter = animation.FFMpegWriter(fps=args.framerate)
    anim.save(args.output_path, writer=FFwriter)


def get_default_equation(args):
    if args.equation is None:
        args.equation = "x^11 - x^10 + (30j[0]^2 -30[0] - 30) x^8 + (-30[1]^5 - 30j[1]^3 + 30j[1]^2 - 30j[1] + 30) x^6 + (30j[2]^2 + 30j[2] - 30) x^5"
        args.varying_indices = [0, 2]
        args.fixed_indices = [1]

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", type=int, default=100000, help="number of particles")
    parser.add_argument("--threads", type=int, default=10, help="number of threads")
    parser.add_argument("--framerate", type=int, default=24, help="frames per second")
    parser.add_argument("--length", type=float, default=5.0, help="number of frames")
    parser.add_argument("--seed", type=int, default=0xC0FFEE, help="random seed")
    parser.add_argument("--output-path", type=str, default="animation.mp4", help="output path")
    parser.add_argument("--disable-cache", action="store_true", help="disable compilation cache")
    # parser.add_argument("--coeff1-index", type=int, default=8)
    # parser.add_argument("--coeff2-index", type=int, default=5)
    # parser.add_argument("--coeff-varying-index", type=int, default=6)
    parser.add_argument("--equation", type=str, default=None)  # set to default outside parser, as it is long
    parser.add_argument("--varying-indices", nargs="+", type=int, default=None)
    parser.add_argument("--fixed-indices", nargs="*", type=int, default=None)
    parser.add_argument("--hist-bins", type=int, default=1000)
    parser.add_argument("--max-filter-size", type=int, default=3)
    parser.add_argument("--colourmap", type=str, default="gray")
    # TODO: add way to serialize the polynomial so we don't have to hardcode one
    """
    kinda like
    \[
        x^11
        - x^10
        + (30j[0]^2 -30[0] - 30) x^8
        + (-30[1]^5 - 30j[1]^3 + 30j[1]^2 - 30j[1] + 30) x^6
        + (30j[2]^2 + 30j[2] - 30) x^5
    \]

    where x is the roots
    [a] is the ath coefficient, specified by coeff-index or varying-coeff-index
    j is sqrt(-1)

    """
    args = parser.parse_args()
    args = get_default_equation(args)

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(args.threads)

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not args.disable_cache:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    main(args)
