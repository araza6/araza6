---

title: "qgrad: An autodifferentiation framework for quantum physics routines"

date: "2020-08-29"

description: "Getting started with qgrad"

tags: [gsoc, code, physics, qgrad]


---



This is the last post in a series of posts I have been puttting up
for my Google Summer of Code (GSoC) 2020 project. This post shall 
serve as a "Getting Started" guide to `qgrad`, in that it will walk
you through all the major features of qgrad and how they can be possibly
used. But before that, here are two quick links in case you 
want to take a nose-dive and explore yourself

**Project GitHub**: https://github.com/qgrad/qgrad

**Documentation**: https://qgrad.readthedocs.io/en/latest/#

In essence, qgrad is a package that makes autodifferentiation of 
quantum functions easier. Here "quantum function" means any
routine that involves quantum physics operators and/or
quantum states. Differentiation of such 
quantum functions is extremely useful in quantum 
machine learning, quantum control, and the like.
You may have any quantum function whose
derivative you might want to evaluate, you can construct your 
favorite quantum routine using the functions we provide as part of
our [API](https://qgrad.readthedocs.io/en/latest/api.html).
All the functions in our API are autdiff compatible, which is to 
say that you can simply take gradients of your function

```python
from jax import grad

# Define your function
def qfunc():
    ....

gradient = grad(func)
```

Often times, one can simply use fidelity as the cost function
in tasks like quantum state preparation, this
[basic qubit rotation tutorial](https://qgrad.readthedocs.io/en/latest/Qubit_Rotation.html)
shows how
qgrad helps with moving from an initial state to a 
target state using gradient-based optimization.

With qgrad, you can differentiate commonly used quantum physics
and quantum optics 
operators like the displacement operator, the squeeze operator, 
etc. In the 
[SNAP Gates tutorial](https://qgrad.readthedocs.io/en/latest/SNAP_gates.html)
I show how qgrad can be used to differentiate a 
cost function involving SNAP gates 
and the displacement operator.


In addition, qgrad also supports hamiltonian learning type
tasks. One can
parametrize an arbitrary $N$-dimensional
unitary using the `Unitary` class
with $N^2$ parameters as follows

```python
from jax.random import PRNGKey, uniform
from qgrad.qgrad_qutip import Unitary

N = 10 # Dimension of the unitary
params = uniform(PRNGKey(0), (N**2, ),
                        minval=0.0, maxval=2 * jnp.pi)
thetas = params[:N * (N-1) // 2]
phis = params[N * (N - 1) // 2 : N * (N - 1)]
omegas = params[N * (N - 1):]
unitary = Unitary(N)(thetas, phis, omegas)
```

Full unitary learning tutorial 
can be found [here](https://qgrad.readthedocs.io/en/latest/Unitary-Learning-qgrad.html), 
where starting from a randomly parametrized unitary, we learn the desired
target unitary. 
I have also written a 
unitary learning tutorial **without** 
qgrad to juxtapose how gradient calculation
involving parameterized unitaries becomes easier
with qgrad.

Unitary matrices are also extremely crucial in quantum circuit 
learning routines. In this [Circuit Learning tutorial](https://qgrad.readthedocs.io/en/latest/Circuit-Learning.html), I show how
a two-qubit parametrized circuit can be optimized using 
qgrad. Indeed, for circuit learning with qgrad, one 
has to define the JAX-compatible representation of quantum
gates themselves. Say for rotation around 
the y-axis, $R_{Y}(\phi)$, one writes

```python
def ry(phi):
    """Rotation around y-axis

    Args:
        phi (float): rotation angle

    Returns:
        :obj:`jnp.ndarray`: Matrix
        representing rotation around
        the y-axis

    """
    return jnp.array([[jnp.cos(phi / 2), -jnp.sin(phi / 2)],
                     [jnp.sin(phi / 2), jnp.cos(phi / 2)]])
```

and this requires additional work from the user. 
However, qgrad never intended to be a full
quantum circuit simulator, so understandably this would
be a pain for qgrad users working with quantum circuits. Perhaps
in future releases, circuit simulators might be one of
our priorities, but the existing quantum machine learning 
libraries like [Pennylane](https://pennylane.ai/) already
do a great job at it.

I reckon that these tutorial examples shall serve 
as a hook to encourage quantum developers and 
quantum physcists alike to use `qgrad`. The
existing tutorials just begin to show what
can be accomplished with `qgrad`. I can't
wait to see what exciting things you will develop
with this package.

## What's next?

GSoC 2020 concludes, and so does the first phase of
`qgrad` development. Looking
forward, the goal is now to **fully** integrate
QuTiP with an autodiff framework because qgrad
only provides a subset of quantum functions 
available in QuTiP. Another possible direction
is to make tools to be able to differentiate
through the evolution of the Schrodinger's equation.
While there are other small nice-to-haves, these
are the two major future directions for `qgrad`.
I have detailed our future plans in this wiki called
[Going forward: post-GSoC for qgrad](https://github.com/qgrad/qgrad/wiki/Going-forward:-post-GSoC-for-qgrad).


## Acknowledgements

`qgard`'s development was supported by Google Summer of 
Code, so firstly a huge shout out to Google for sponsoring 
such projects and helping to keep the open-source scene
fueled up. My project, in particular, fell under
NUMFOCUS's umbrella as part of QuTiP. I extend a token
of thanks to both of these organizations to
have given me this opportunity. Finally, I would thank 
my mentors, [Nathan Shammah](https://nathanshammah.com/)
from [Unitary Fund](https://unitary.fund/)
and [Shahanawaz Ahmed](http://sahmed.in/) 
from [Chalmers University of Technology, Göteborg](https://www.chalmers.se/en/Pages/default.aspx)
for guiding 
me throughout this project.