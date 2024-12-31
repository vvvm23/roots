import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import math

import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence, Optional

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

def complex_at_angle(x):
    if isinstance(x, np.ndarray):
        return (np.cos(x) + 1j * np.sin(x)).astype(np.complex64)
    if isinstance(x, jnp.ndarray):
        return (jnp.cos(x) + 1j * jnp.sin(x)).astype(jnp.complex64)

    return math.cos(x) + 1j * math.sin(x)

# TODO: make this have configurable coefficients, for now hardcode 

def jax_polyroots(x):
    n = len(x) - 1
    mat = jnp.diag(jnp.ones((n-1,), dtype=x.dtype), k=-1)
    mat = mat.at[:, -1].subtract(x[:-1]/x[-1])
    r = jnp.linalg.eigvals(mat[::-1,::-1])
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
            bins=(jnp.linspace(x_min, x_max, bins+1), jnp.linspace(y_min, y_max, bins+1)),
            range=jnp.array([[x_min, x_max], [y_min, y_max]]),
            density=False,
        )[0]
    
    return histogram_fn(roots_real, roots_imag)

# def generate_all_frames(
#     frames: jax.Array,
#     coefficients: jax.Array,
#     coefficient_overrides_index: Sequence[int],
#     coefficient_varying_index: int,
#     n_frames: int,
#     *coefficient_overrides: Sequence[jax.Array],
# ):
#     def scan_fn(carry, xs):
#         coefficients, overrides = carry
#         fi = xs

#         tv = complex_at_angle(2 * math.pi * (fi / n_frames))
#         varying = -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30

#         roots = roots_one_frame(
#             coefficients, 
#             coefficient_overrides_index,
#             coefficient_varying_index,
#             varying, 
#             *overrides
#         )

#         return (coefficients, overrides), roots

#     _, roots = jax.lax.scan(
#         scan_fn, 
#         (coefficients, coefficient_overrides), 
#         frames, 
#     )

#     return roots

def main(
    degree: int = 11,
    N: int = 200*200,
    length: float = 5.0,
    framerate: int = 24
    # TODO: string for passing in the coefficients and such
):
    n_frames = int(length*framerate)

    sqrt_N = int(math.sqrt(N))
    N_by_sqrt_N = N // sqrt_N

    # TODO: make this more flexible
    # assert N_by_sqrt_N * sqrt_N == N

    coefficients = np.zeros((degree + 1,), dtype=np.complex64)
    coefficients[11] = 1
    coefficients[10] = -1

    # t1s = np.tile(np.linspace(0.0, 2*math.pi, sqrt_N), N_by_sqrt_N)
    # t2s = np.repeat(np.linspace(0.0, 2*math.pi, N_by_sqrt_N), sqrt_N)
    t1s = np.random.rand(N) * 2 * math.pi
    t2s = np.random.rand(N) * 2 * math.pi

    # project timesteps on unit circle
    t1s = complex_at_angle(t1s)
    t2s = complex_at_angle(t2s)

    coeffs8 = 30j * t1s * t1s - 30 * t1s - 30
    coeffs5 = 30j * t2s * t2s + 30j * t2s - 30

    jit_roots_one_frame = jax.jit(roots_one_frame, backend='cpu', static_argnums=(1, 2))
    # print(jax.devices('cpu'))
    # jit_roots_one_frame = jax.pmap(
    #     roots_one_frame,
    #     in_axes=0,
    #     devices=jax.devices('cpu'),
    #     out_axes=0
    # )
    jit_roots_to_histogram = jax.jit(roots_to_histogram, static_argnums=(2,), backend='cpu')

    start_time = time.time()

    frame_roots = []
    for fi in tqdm.tqdm(range(n_frames)):
        tv = complex_at_angle(2 * math.pi * (fi / n_frames))

        coeff6 = -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30

        # cool settings
        # original (8, 5) 6
        # (8, 2) 6
        # (8, 2) 7
        # (8, 2) 4
        roots = jit_roots_one_frame(
            coefficients, 
            (4, 5),
            9,
            coeff6, 
            coeffs8, 
            coeffs5, 
        )

        # TODO: we filter out zeros outside jit, is there a way to do this inside?
        roots = np.asarray(roots)
        roots = roots[~np.isclose(roots, 0.0 + 0.0j, atol=1e-4, rtol=1e-4)]
        roots = np.unique(roots)
        roots = np.ascontiguousarray(roots)

        frame_roots.append(roots)

    # fis = np.arange(n_frames)
    # jit_all_frames = jax.jit(generate_all_frames, backend='cpu', static_argnums=(2, 3, 4))
    
    # stacked_frame_roots = jit_all_frames(
    #     fis, 
    #     coefficients, 
    #     (8, 5),
    #     6,
    #     n_frames,
    #     coeffs8, 
    #     coeffs5, 
    # ).block_until_ready()

    end_time = time.time()
    print(f'Took {end_time - start_time} seconds to generate all roots')

    # start_time = time.time()
    # frame_roots = []
    # for roots in stacked_frame_roots:
    #     roots = np.asarray(roots)
    #     roots = roots[~np.isclose(roots, 0.0 + 0.0j, atol=1e-4, rtol=1e-4)]
    #     roots = np.unique(roots)
    #     roots = np.ascontiguousarray(roots)

    #     frame_roots.append(roots)

    # end_time = time.time()
    # print(f'Took {end_time - start_time} seconds to clean up all roots')


    hists = []
    x_min, x_max = np.min([np.min(roots.real) for roots in frame_roots]), np.max([np.max(roots.real) for roots in frame_roots])
    y_min, y_max = np.min([np.min(roots.imag) for roots in frame_roots]), np.max([np.max(roots.imag) for roots in frame_roots])

    # x_min, x_max = np.min([roots.real for roots in frame_roots]), np.max([roots.real for roots in frame_roots])
    # y_min, y_max = np.min([roots.imag for roots in frame_roots]), np.max([roots.imag for roots in frame_roots])

    
    start_time = time.time()
    for roots in tqdm.tqdm(frame_roots):
        hist = jit_roots_to_histogram(
            roots.real.astype(np.float64), 
            roots.imag.astype(np.float64),
            bins=1000,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
        )
        hists.append(hist)

    hists = np.array(hists)

    end_time = time.time()
    print(f'Took {end_time - start_time} seconds to generate all histograms')
    
    fig, ax = plt.subplots(figsize = (4,4))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(4, 4, True)
    ax.set_axis_off()
    im = plt.imshow(hists[0], cmap='gray', origin='lower')

    def animate(i):
        im.set_data(hists[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(hists), interval=int(1000 / framerate), blit=False)
    anim.save('animation.gif', dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()