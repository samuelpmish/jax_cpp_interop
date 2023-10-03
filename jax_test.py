import jax
from jax import grad
from jax import numpy as np
from jax.numpy import float64 as f64
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from build.nanobind_example_ext import func, vjp

x = np.array([1.0, 1.0, 1.0], f64)

Ax = func(x)
print(Ax)

xA = vjp(x)
print(xA)

@jax.custom_jvp
def f(x):
  result_shape_dtype = jax.ShapeDtypeStruct(shape=(3,), dtype=f64)
  return jax.pure_callback(func, result_shape_dtype, x)

@f.defjvp
def _f_jvp(primals, tangents):
  x, = primals
  dx, = tangents
  return f(x), f(dx)

print(f(x))
print(jax.jacfwd(f)(x))
