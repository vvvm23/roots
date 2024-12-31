import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import math

import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

def complex_at_angle(x):
    return (jnp.cos(x) + 1j * jnp.sin(x)).astype(jnp.complex64)

# TODO: make this have configurable coefficients, for now hardcode 
# TODO: cache compilation cache to disk?
@partial(jax.jit, static_argnums=(0, 1, 2, 3), backend='cpu')
def jit_main(
    # key: jr.PRNGKey,
    degree: int,
    N: int,
    num_frames: int,
    bins: int = 200
):
    coefficients = jnp.zeros((degree + 1,), dtype=jnp.complex64)

    coefficients = coefficients.at[11].set(1)    
    coefficients = coefficients.at[10].set(-1)

    sqrt_N = int(math.sqrt(N))
    N_by_sqrt_N = N // sqrt_N

    # TODO: make this more flexible
    assert N_by_sqrt_N * sqrt_N == N

    # linearly spaced points along two time frames
    t1s = jnp.tile(jnp.linspace(0.0, 2*math.pi, sqrt_N), N_by_sqrt_N)
    t2s = jnp.repeat(jnp.linspace(0.0, 2*math.pi, N_by_sqrt_N), sqrt_N)

    # project timesteps on unit circle
    t1s = complex_at_angle(t1s)
    t2s = complex_at_angle(t2s)

    coeffs8 = 30j * t1s * t1s - 30 * t1s - 30
    coeffs5 = 30j * t2s * t2s + 30j * t2s - 30

    tvs = jnp.linspace(0.0, 2*math.pi, num_frames)
    tvs = complex_at_angle(tvs)

    # we want to carry over the coefficients and ts from one frame to the next
    # and output the roots for this frame
    def scan_fn(carry, xs):
        coefficients, coeffs8, coeffs5 = carry
        tv = xs

        coefficients = coefficients.at[6].set(
            -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30
        )

        def compute_root(coefficients, coeffs8, coeffs5):
            coefficients = coefficients.at[8].set(coeffs8)
            coefficients = coefficients.at[5].set(coeffs5)

            roots = jnp.roots(coefficients, strip_zeros=False).astype(jnp.complex64)

            return roots
        
        vmap_compute_root = jax.vmap(compute_root, in_axes=(None, 0, 0))
        frame_roots = vmap_compute_root(coefficients, coeffs8, coeffs5)
        frame_roots = jnp.reshape(frame_roots, (-1,))

        return (coefficients, coeffs8, coeffs5), frame_roots

    _, roots = jax.lax.scan(scan_fn, (coefficients, coeffs8, coeffs5), tvs, unroll=1)

    x_min, x_max = jnp.nanmin(roots.real.astype(jnp.float64)), jnp.nanmax(roots.real.astype(jnp.float64))
    y_min, y_max = jnp.nanmin(roots.imag.astype(jnp.float64)), jnp.nanmax(roots.imag.astype(jnp.float64))

    def masked_histogram(roots):
        # weird trick to set all zero and nan to nan, and else to 1
        # then mask out the nans, yielding weights
        weights = jnp.nan_to_num((roots / roots).real.astype(jnp.float64), copy=False, nan=0.0)

        return jnp.histogram2d(
            roots.real.astype(jnp.float64), 
            roots.imag.astype(jnp.float64), 
            # bins=200, 
            bins=(jnp.linspace(x_min, x_max, bins), jnp.linspace(y_min, y_max, bins)),
            range=jnp.array([[x_min, x_max], [y_min, y_max]]),
            weights=weights,
            density=False,
        )[0]


    vmapped_histogram = jax.vmap(masked_histogram)

    hists = vmapped_histogram(roots)

    return hists

def main(
    degree: int = 11,
    N: int = 10000,
    length: float = 5.0,
    framerate: int = 24
    # TODO: string for passing in the coefficients and such
):
    hists = jit_main(degree, N, int(length * framerate)).block_until_ready()
    import ipdb; ipdb.set_trace()
    
    fig, ax = plt.subplots(figsize = (8,8))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(4, 4, True)
    ax.set_axis_off()
    im = plt.imshow(hists[0], cmap='gray', origin='lower')

    def animate(i):
        im.set_data(hists[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(hists), interval=int(1000 / framerate), blit=False)
    anim.save('animation.gif', dpi=300)


if __name__ == "__main__":
    main()