"""
This script is exploring differences between numpy polyroot, jax roots, and my
own jax polyroot.
"""
import jax
import math
import numpy as np
import random
import jax.numpy as jnp

from functools import partial
def get_complex_at_angle(angle):
    return math.cos(angle) + 1j * math.sin(angle)

@partial(jax.jit, backend='cpu')
def jax_root(coeffs):
    return jnp.roots(coeffs, strip_zeros=False)

degree = 11
x = np.zeros((degree+1,), dtype=np.complex64)
x[11] = 1. + 0j
x[10] = -1. + 0j

t1, t2 = get_complex_at_angle(2*math.pi*random.random()), get_complex_at_angle(2*math.pi*random.random())

x[8] = 30j * t1 * t1 - 30 * t1 - 30
x[5] = 30j * t2 * t2 + 30j * t2 - 30

@partial(jax.jit, backend='cpu')
def jax_polyroots(x):
    n = len(x) - 1
    mat = jnp.diag(jnp.ones((n-1,), dtype=x.dtype), k=-1)
    mat = mat.at[:, -1].subtract(x[:-1]/x[-1])
    r = jnp.linalg.eigvals(mat[::-1,::-1])
    return jnp.sort(r)

def normalise_result(x):
    x = np.nan_to_num(x)
    x = np.unique(x)
    x = x[~np.isclose(x, 0+0j)]

    return x

for fi in range(1):
    tv = get_complex_at_angle(2 * math.pi * (fi / 10000)) 
    x[6] = -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30

    np_result = np.polynomial.polynomial.polyroots(x)
    np_roots_result = np.roots(x)
    jnp_result = np.asarray(jax_root(x))
    jax_polyroot_result = np.asarray(jax_polyroots(x))

    np_result = normalise_result(np_result)
    np_roots_result = normalise_result(np_roots_result)
    jnp_result = normalise_result(jnp_result)
    jax_polyroot_result = normalise_result(jax_polyroot_result)


    print(np_result)
    print(np_roots_result)
    print(jnp_result)
    print(jax_polyroot_result)
    print()
