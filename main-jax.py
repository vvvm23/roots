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


def complex_at_angle(x):
    if isinstance(x, np.ndarray):
        return (np.cos(x) + 1j * np.sin(x)).astype(np.complex64)
    if isinstance(x, jnp.ndarray):
        return (jnp.cos(x) + 1j * jnp.sin(x)).astype(jnp.complex64)

    return math.cos(x) + 1j * math.sin(x)


# TODO: can this be... faster? and on gpu perhaps
def jax_polyroots(x):
    n = len(x) - 1
    mat = jnp.diag(jnp.ones((n - 1,), dtype=x.dtype), k=-1)
    mat = mat.at[:, -1].subtract(x[:-1] / x[-1])
    r = jnp.linalg.eigvals(mat[::-1, ::-1])
    return jnp.sort(r)


def roots_one_frame(
    coefficients: jax.Array,
    fixed_coefficients: jax.Array,
    varying_coefficients: jax.Array,
):
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
            density=True,
        )[0]

    return histogram_fn(roots_real, roots_imag)


def main(
    args: argparse.Namespace,
):
    equation = string_to_equation(args.equation)
    print("\n".join(map(str, equation.terms)))

    devices = jax.devices("cpu")
    n_devices = len(devices)
    mesh = jax.make_mesh((n_devices,), ("data",))
    n_frames = int(args.length * args.framerate)

    # TODO: can probably alloc this inside the jit function rather than outside
    degree = max([t.degree for t in equation.terms])
    coefficients = np.zeros((degree + 1,), dtype=np.complex64)

    rng = np.random.default_rng(args.seed)
    ts = {i: complex_at_angle(rng.random(args.N) * 2 * math.pi) for i in args.fixed_indices}
    fixed_coeffs = {}
    partial_varying_coeffs = {}
    for term in equation.terms:
        if all([t.t is None for t in term.coefficient_parts]):
            # if term has a constant coefficient
            coefficients[term.degree] = term.coefficient_parts[0].constant
        elif all([t.t is None or t.t[0] in args.fixed_indices for t in term.coefficient_parts]):
            # if term has all fixed indices, they can be precomputed
            fixed_coeffs[term.degree] = np.zeros((args.N,), dtype=np.complex64)
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
                    partial_varying_coeffs[term.degree] += coeff_part.constant * (
                        ts[coeff_part.t[0]] ** coeff_part.t[1]
                    )
                else:
                    partial_varying_coeffs[term.degree] += coeff_part.constant

    coefficients = jnp.asarray(coefficients)
    fixed_coeffs = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.complex64), fixed_coeffs)

    jit_roots_to_histogram = jax.jit(roots_to_histogram, static_argnums=(2,), backend="cpu")

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
            v = np.copy(v)
            for term in equation.terms:
                if term.degree == degree:
                    for coeff_part in term.coefficient_parts:
                        if coeff_part.t is not None and coeff_part.t[0] in args.varying_indices:
                            v += coeff_part.constant * (ts ** coeff_part.t[1])

                    return v
            else:
                raise ValueError("didn't find degree")

        varying_coeffs = {
            i: jnp.asarray(_resolve_partial_varying(i, v, tv), dtype=jnp.complex64)
            for i, v in partial_varying_coeffs.items()
        }

        roots = shmap_roots_one_frame(
            coefficients,
            fixed_coeffs,
            varying_coeffs,
        )

        roots = np.asarray(roots)

        # TODO: we filter out zeros outside jit, is there a way to do this inside?
        # is_close = np.isclose(roots, 0.0 + 0.0j, atol=1e-4, rtol=1e-4)
        # is_close_batch = np.all(is_close, axis=-1)
        # roots = roots[~is_close]

        # roots = np.unique(roots)
        # roots = np.ascontiguousarray(roots)

        frame_roots.append(roots)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to generate all roots")

    frame_roots = np.stack(frame_roots, axis=0)
    frame_roots_real = frame_roots.real
    frame_roots_imag = frame_roots.imag

    roots_min_x, roots_max_x = np.min(frame_roots_real), np.max(frame_roots_real)
    roots_min_y, roots_max_y = np.min(frame_roots_imag), np.max(frame_roots_imag)

    # thinking through colour step by step..
    # TODO: jaxify the colouring?
    # - scale up x and y to fit inside a [hist_bins, hist_bins] resolution window (change name later)
    # - cast to int
    # - use these as scatter add indices into an intensity bucket
    # - normalise the intensity buckets

    # scale along both axes by the same size, taking the larger of the two
    roots_diff_x, roots_diff_y = roots_max_x - roots_min_x, roots_max_y - roots_min_y
    if roots_diff_x < roots_diff_y:
        frame_roots_real = (frame_roots_real - roots_min_y) / roots_diff_y * (args.hist_bins - 1)
        frame_roots_imag = (frame_roots_imag - roots_min_y) / roots_diff_y * (args.hist_bins - 1)
    else:
        frame_roots_real = (frame_roots_real - roots_min_x) / roots_diff_x * (args.hist_bins - 1)
        frame_roots_imag = (frame_roots_imag - roots_min_x) / roots_diff_x * (args.hist_bins - 1)

    frame_roots_real = frame_roots_real.astype(np.int32)
    frame_roots_imag = frame_roots_imag.astype(np.int32)

    intensities = np.zeros((n_frames, args.hist_bins, args.hist_bins), dtype=np.int32)
    # TODO: any way to vectorise this loop?
    for fi in range(n_frames):
        np.add.at(intensities[fi], (frame_roots_real[fi], frame_roots_imag[fi]), 1)

    # TODO: how to deal with hues based on t?
    # hues_complex = np.ones((args.hist_bins, args.hist_bins), dtype=np.complex64)
    # for t_hue in ts.values():
    #     np.add.at(hues_complex, (frame_roots_real, frame_roots_imag), t_hue)

    # saturation = 0.5

    hists = intensities / intensities.max()

    # hists = []
    # x_min, x_max = np.min([np.min(roots.real) for roots in frame_roots]), np.max(
    #     [np.max(roots.real) for roots in frame_roots]
    # )
    # y_min, y_max = np.min([np.min(roots.imag) for roots in frame_roots]), np.max(
    #     [np.max(roots.imag) for roots in frame_roots]
    # )

    # # TODO: parallelise this too
    # start_time = time.time()
    # for roots in tqdm.tqdm(frame_roots):
    #     hist = jit_roots_to_histogram(
    #         roots.real.astype(np.float64),
    #         roots.imag.astype(np.float64),
    #         bins=args.hist_bins,
    #         x_min=x_min,
    #         x_max=x_max,
    #         y_min=y_min,
    #         y_max=y_max,
    #     )
    #     hists.append(hist)

    # hists = np.array(hists)
    # hists = hists / hists.max()

    upweight = 10.0
    threshold = 0.1
    hists *= np.maximum(0.0, upweight - upweight * hists / threshold) + 1

    hists = scipy.ndimage.maximum_filter(hists, size=(1, args.max_filter_size, args.max_filter_size))

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


FUN_EQUATIONS = [
    (
        "x^11 - x^10 + (30j[0]^2 -30[0] - 30) x^8 + (-30[1]^5 - 30j[1]^3 + 30j[1]^2 - 30j[1] + 30) x^6 + (30j[2]^2 + 30j[2] - 30) x^5",
        (1,),
        (0, 2),
    ),
    (
        "x^24 + (-0.1j) x^17 + (10j[1] - 10) x^16 + (-15) x^7 + (15j[0]^5 - 15[0]^4 - 15[0]^2 - 15j) x^10 + (10j[1]^6 + 10j[1]^5 - 10j[1]^4 + 10j[1]^3 - 10[1]^2 + 10j[1] + 10j) x^1 + (1.5j) x^0",
        (0,),
        (1,),
    ),
    (
        "x^24 + (-0.1j) x^17 + x^14 + (-10j[0]^6 + 10j[0]^5 + 10j[0]^4 - 10j[0]^3 + 10j[0]^2 - 10j[0] - 10j) x^11 + (10j[1]^8 - 10j[1]^4 + 10[1]^2 + 10[1] - 10j) x^4 + (-10j[1] - 10j) x^9 + (-50 - 0.1j) x^7 + (5j) x^0",
        (0,),
        (1,),
    ),
]


def get_preset_equation(args):
    if args.preset_equation is None and args.equation is None:
        args.preset_equation = 0

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
    parser.add_argument("--equation", type=str, default=None)  # set to default outside parser, as it is long
    parser.add_argument("--varying-indices", nargs="+", type=int, default=None)
    parser.add_argument("--fixed-indices", nargs="*", type=int, default=None)
    parser.add_argument("--hist-bins", type=int, default=1000)
    parser.add_argument("--max-filter-size", type=int, default=3)
    parser.add_argument("--colourmap", type=str, default="gray")
    parser.add_argument("--preset-equation", type=int, default=None)
    args = parser.parse_args()

    args = get_preset_equation(args)

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(args.threads)

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not args.disable_cache:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    main(args)
