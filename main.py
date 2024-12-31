import argparse
import colorsys
import math
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import tqdm
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from equation_parser import string_to_equation
from preset_equations import FUN_EQUATIONS
from utils import complex_at_angle, jax_polyroots

COLOR_MODE_CHOICES = ["cmap", "t_fixed_hue"]


def roots_one_frame(
    coefficients: jax.Array,
    fixed_coefficients: jax.Array,
    varying_coefficients: jax.Array,
):
    """
    Calculate the roots of the polynomial defined by a set of fixed and varying coefficients.
    """
    flat_varying = {k: v.astype(jnp.complex64) for k, v in varying_coefficients.items() if v.ndim == 0}
    for i in flat_varying:
        del varying_coefficients[i]

    coefficient_overrides = fixed_coefficients | varying_coefficients

    for k, v in flat_varying.items():
        coefficients = coefficients.at[k].set(v)

    def scan_fn(carry, xs):
        coefficients = carry
        coeff_overrides = xs

        for k, v in coeff_overrides.items():
            coefficients = coefficients.at[k].set(v)

        roots = jax_polyroots(coefficients).astype(jnp.complex64)
        roots = jnp.reshape(roots, (-1,))
        roots = jnp.nan_to_num(roots, copy=False, nan=0.0)

        return coefficients, roots

    _, roots = jax.lax.scan(scan_fn, coefficients, coefficient_overrides, unroll=4)

    return roots


def main(
    args: argparse.Namespace,
):
    equation = string_to_equation(args.equation)
    print("\n".join(map(str, equation.terms.values())))

    devices = jax.devices("cpu")
    n_devices = len(devices)
    mesh = jax.make_mesh((n_devices,), ("data",))
    n_frames = int(args.length * args.framerate)

    # TODO: can probably alloc this inside the jit function rather than outside
    degree = max(equation.terms)
    coefficients = np.zeros((degree + 1,), dtype=np.complex64)

    rng = np.random.default_rng(args.seed)
    ts = {i: complex_at_angle(rng.random(args.N) * 2 * math.pi) for i in args.fixed_indices}
    fixed_coeffs = {}
    partial_varying_coeffs = {}
    for term_degree, term in equation.terms.items():
        if all([t.t is None for t in term.coefficient_parts]):
            # if term has a constant coefficient
            coefficients[term_degree] = term.coefficient_parts[0].constant
        elif all([t.t is None or t.t[0] in args.fixed_indices for t in term.coefficient_parts]):
            # if term has all fixed indices, they can be precomputed
            fixed_coeffs[term_degree] = np.zeros((args.N,), dtype=np.complex64)
            for coeff_part in term.coefficient_parts:
                if coeff_part.t is not None:
                    fixed_coeffs[term_degree] += coeff_part.constant * ts[coeff_part.t[0]] ** coeff_part.t[1]
                else:
                    fixed_coeffs[term_degree] += coeff_part.constant
        else:
            # either all varying, or a mix of varying a fixed, we can precompute some
            # TODO: we can probably merge this and the above branch
            partial_varying_coeffs[term_degree] = 0  # np.zeros((args.N,), dtype=np.complex64)
            for coeff_part in term.coefficient_parts:
                if coeff_part.t is not None and coeff_part.t[0] in args.fixed_indices:
                    partial_varying_coeffs[term_degree] += coeff_part.constant * (
                        ts[coeff_part.t[0]] ** coeff_part.t[1]
                    )
                else:
                    partial_varying_coeffs[term_degree] += coeff_part.constant

    coefficients = jnp.asarray(coefficients)
    fixed_coeffs = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.complex64), fixed_coeffs)

    varying_in_specs = {
        i: P()
        if isinstance(v, complex)
        else P(
            "data",
        )
        for i, v in partial_varying_coeffs.items()
    }

    in_specs = (None, P("data"), varying_in_specs)

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

        def _resolve_partial_varying(degree, v, ts):
            # partially resolve coefficients with varying components
            # we enforce coefficients that are simply sums of components, so can
            # do this by computing the partial sum.
            v = np.copy(v)
            coefficient_parts = equation.terms[degree].coefficient_parts
            for coeff_part in coefficient_parts:
                if coeff_part.t is not None and coeff_part.t[0] in args.varying_indices:
                    v += coeff_part.constant * (ts ** coeff_part.t[1])

            return v

        varying_coeffs = {
            i: jnp.asarray(_resolve_partial_varying(i, v, tv), dtype=jnp.complex64)
            for i, v in partial_varying_coeffs.items()
        }

        roots = shmap_roots_one_frame(
            coefficients,
            fixed_coeffs,
            varying_coeffs,
        )

        frame_roots.append(roots)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to generate all roots")

    frame_roots = np.stack(frame_roots, axis=0)
    frame_roots_real = frame_roots.real
    frame_roots_imag = frame_roots.imag

    roots_min_x, roots_max_x = np.min(frame_roots_real), np.max(frame_roots_real)
    roots_min_y, roots_max_y = np.min(frame_roots_imag), np.max(frame_roots_imag)

    # scale along both axes by the same size, taking the larger of the two
    roots_diff_x, roots_diff_y = roots_max_x - roots_min_x, roots_max_y - roots_min_y
    if roots_diff_x < roots_diff_y:
        frame_roots_real = (frame_roots_real - roots_min_x) / roots_diff_y
        frame_roots_imag = (frame_roots_imag - roots_min_y) / roots_diff_y
    else:
        frame_roots_real = (frame_roots_real - roots_min_x) / roots_diff_x
        frame_roots_imag = (frame_roots_imag - roots_min_y) / roots_diff_x

    frame_roots_real = (frame_roots_real * (args.resolution - 1)).astype(np.int32)
    frame_roots_imag = (frame_roots_imag * (args.resolution - 1)).astype(np.int32)

    # we compute intensities and optional colouring in numpy rather than jax as
    # the speed up is not significant.
    intensities = np.zeros((n_frames, args.resolution, args.resolution), dtype=np.int32)
    # TODO: any way to vectorise this loop?
    for fi in range(n_frames):
        np.add.at(intensities[fi], (frame_roots_real[fi], frame_roots_imag[fi]), 1)
    intensities = intensities / intensities.max()

    upweight = 10.0
    threshold = 0.2
    intensities *= np.maximum(0.0, upweight - upweight * intensities / threshold) + 1
    intensities = scipy.ndimage.maximum_filter(intensities, size=(1, args.max_filter_size, args.max_filter_size))

    # hue
    if args.color_mode == "t_fixed_hue":
        # TODO: any way to vectorise this loop?
        hues_complex = np.ones((n_frames, args.resolution, args.resolution), dtype=np.complex64)
        for fi in tqdm.tqdm(range(n_frames)):
            for t_hue in ts.values():
                for i, hue_complex in enumerate(t_hue):
                    np.add.at(hues_complex[fi], (frame_roots_real[fi, i], frame_roots_imag[fi, i]), hue_complex)

        hues = (np.angle(hues_complex) + np.pi) / (2 * np.pi)
        saturation = np.full((n_frames, args.resolution, args.resolution), 1.0)

        hists = colors.hsv_to_rgb(np.stack([hues, saturation, intensities], axis=-1))
    elif args.color_mode == "cmap":
        hists = intensities

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(4, 4, True)
    ax.set_axis_off()
    im = plt.imshow(hists[0], cmap=args.cmap, origin="lower")

    def animate(i):
        im.set_data(hists[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(hists), interval=int(1000 / args.framerate), blit=False)
    FFwriter = animation.FFMpegWriter(fps=args.framerate)
    anim.save(args.output_path, writer=FFwriter)


def get_preset_equation(args):
    if args.preset_equation is None and args.equation is None:
        args.preset_equation = 1

    if args.equation is None:
        args.equation, args.varying_indices, args.fixed_indices = FUN_EQUATIONS[args.preset_equation]

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", type=int, default=100000, help="number of particles")
    parser.add_argument("--threads", type=int, default=10, help="number of threads")
    parser.add_argument("--framerate", type=int, default=24, help="frames per second")
    parser.add_argument("--length", type=float, default=5.0, help="number of frames")
    parser.add_argument("--seed", type=int, default=0xC0FFEE, help="random seed")
    parser.add_argument("--output-path", type=str, default=f"animation-{time.time()}.mp4", help="output path")
    parser.add_argument("--disable-cache", action="store_true", help="disable compilation cache")
    parser.add_argument("--equation", type=str, default=None, help="string representation of equation to plot")
    parser.add_argument("--varying-indices", nargs="+", type=int, default=None)
    parser.add_argument("--fixed-indices", nargs="*", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1000, help="resolution of output animation.")
    parser.add_argument("--max-filter-size", type=int, default=3, help="maximum filter size")
    parser.add_argument("--color-mode", type=str, default="t_fixed_hue", choices=COLOR_MODE_CHOICES, help="color mode to use in output animation")
    parser.add_argument("--cmap", type=str, default="gray", help="matplotlib colormap to use if color_mode is `cmap`")
    parser.add_argument("--preset-equation", type=int, default=None, help="use a zero-indexed preset equation, see preset_equations.py")
    args = parser.parse_args()

    args = get_preset_equation(args)

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(args.threads)

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not args.disable_cache:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print(args.equation)
    main(args)
