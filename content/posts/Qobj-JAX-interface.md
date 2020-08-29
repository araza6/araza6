---

title: "Interfacing JAX with QuTiP"

date: "2020-06-30"

description: "Issues with interfacing JAX with QuTiP and alternate routes to make QuTiP autodifferentiable"

tags: [gsoc, code, physics, qgrad]


---

As the month of June comes to an end, so does the first phase of Google Summer of Code (GSoC) 2020. In my previous [post](https://araza6.github.io/posts/autodiff/autodiff/), I attempted to cover the theory behind auto-differentiation, but did not fully explain how am I going to use autodiff for my GSoC project. Here, I am going to discuss just that. I will explain what problems did I face to make QuTiP work with [JAX](https://jax.readthedocs.io/en/latest/index.html), the famed auto-differentitation library, and what route did I take to solve the problem.

The project started with a goal to _fully_ interface JAX with QuTiP. By that, I mean the following. To be able to differentiate any QuTiP method that returns a scalar output --  because JAX's [`grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad) only auto differentiates methods that have a scalar return value. An example of this workflow would be the following: 

```python
from qutip import basis, fidelity 
from jax import grad

grad(fidelity)(basis(2, 0), basis(2, 1))

```
This is a very smooth workflow for the user who wants to take gradients of QuTiP functions. One case where the user would want to take gradients, of say the `fidelity` function I have used in the snippet above is when `fidelity` is used as a cost function in some (quantum) machine learning routine. 

However this QuTiP-JAX integration is not that easy. The above snippet throws a Type Error 
```
Qobj data =
[[1.]
 [0.]]' of type <class 'qutip.qobj.Qobj'> is not a valid JAX type
 ```
 This is because JAX's `grad` (by default) only works with functions that accept arrays, scalars, or standard Python containers, whereas `fidelity` in our example accepts `Qobj` (see [docs](http://qutip.org/docs/4.0.2/modules/qutip/metrics.html#fidelity)) as input kets or density matrices (whose fidelity one wishes to calculate). One might try converting QuTiP's `Qobj`, using the `full` method in QuTiP, to a standard numpy array before passing it into JAX's `grad`. Something like:
 ```python
 # assuming the same imports as above
 import jax.numpy as jnp

#Converting to JAX-aware numpy arrays
 ket1 = jnp.asarray(basis(2,0).full()) 
 ket2 jnp.asarray(basis(2,0).full())

grad(fidelity)(ket1, ket2)
 ```

 This still would not work since `grad(fidelity)` returns the ``Callable`` that accepts exactly the same arguments as the original function. The original `fidelity` function in QuTiP does not accept numpy array; it accepts `Qobj` class object. Here's the bottleneck: passing on a numpy array to `grad` shall be fine with JAX, but not with QuTiP; passing `Qobj` is fine with QuTiP but not with JAX. 

 When JAX was being developed as Autograd, ``autograd.extend`` was something that could have been of use in our case. In JAX, there does not seem to be any similar option available, or at least it is not well documented. There are some related issues, [here](https://github.com/google/jax/issues/446) and [here](https://github.com/google/jax/issues/1251), but I could not extract enough insights from these issues alone to be able to make QuTiP work seamlessly with JAX. I have a strong intuition that there is a way to make it work, perhaps by defining JAX [primitives](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html) and/or [custom vjps](https://github.com/google/jax/issues/1142). But this exploration has already taxed me quite some time already. This project has a strict time-frame that pushes me to take an alternate route: Make a light version of QuTiP that is JAX-compatible. Now, how does that look like? This means re-writing all the basic QuTiP functions that our intended users may need to construct their Hamiltonians, take expectations, etc. For example, `fidelity` source code (which you don't need to understand fully) in QuTiP looks like
 ```python

 def fidelity(A, B):
    """
    Calculates the fidelity (pseudo-metric) between two density matrices.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    """
    if A.isket or A.isbra:
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        sqrtmA = ket2dm(A)
        # Check whether we have to turn B into a density operator, too.
        if B.isket or B.isbra:
            B = ket2dm(B)
    else:
        if B.isket or B.isbra:
            
            return fidelity(B, A)
       
        sqrtmA = A.sqrtm()

    if sqrtmA.dims != B.dims:
        raise TypeError('Density matrices do not have same dimensions.')

    eig_vals = (sqrtmA * B * sqrtmA).eigenenergies()
    return float(np.real(np.sqrt(eig_vals[eig_vals > 0]).sum()))

 ```

 while the `fidelity` function (for kets only) in `qgrad.qutip` looks like 

 ```python
def fidelity(a, b):
    """
    Computes fidelity between two kets.
    
    Args:
        a (`:obj:numpy.array`): State vector (ket)
        b (`:obj:numpy.array`): State vector (ket)
        
    Returns:
        float: fidelity between the two state vectors
    """
    return jnp.abs(jnp.dot(jnp.transpose(jnp.conjugate(a)), b)) ** 2
 ```

Here we completely get rid of `Qobj` constraint, which by the way is a great feature in QuTiP but became a pain for `qgrad` interfacing. Other similar operators like the Paulis, displacement and squeeze operators, etc are added, so that one can now work with QuTiP functions and also differentiate them, where appropriate.

# Future Work

Having a light version of QuTiP available through `qgrad.qutip`, we are ready to start applying it to exciting areas. I will now be working to make tutorials reconstructing [this](https://arxiv.org/abs/2001.11897) unitary learning paper and this [cavity control](https://arxiv.org/abs/2004.14256) paper. On the side, I shall be exploring ways to accomplish the ideal goal to fully interface JAX with QuTiP. In case that does not work, we would still have a version to let the users play with `qgrad`. 
