import math

import jax
import jax.numpy as jnp
import numpy as np


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
