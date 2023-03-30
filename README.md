<img align="center" height="150" src="./images/sstrax_logo.png">

----
## Description

- **sstrax** is a [jax](https://github.com/google/jax)-accelerated library designed to perform fast simulations of Milky Way stellar stream formation. It is desgined to be integrated with the [albatross](https://github.com/undark-lab/albatross) simulation-based inference code to perform complete parameter inference on stellar streams.
- **Contacts:** For questions and comments on the code, please contact either [James Alvey](mailto:j.b.g.alvey@uva.nl) or [Mathis Gerdes](mailto:m.gerdes@uva.nl). Alternatively feel free to open an issue.
- **Related paper:** The details regarding the implementation of `sstrax` and the physics modelling choices can be found in the companion paper.
- **Citation:** If you use **sstrax** in your analysis, or find it useful, we would ask that you please use the following citation.
```
@article{...}
```

----
## Installing `sstrax`

<img align="center" height="300" src="./images/sstrax_example.png">

`sstrax` is designed to be installed as a standard python package, so can be obtained using the following process
```
cd [path/to/directory]
git clone https://github.com/undark-lab/sstrax.git # for https client
[or git clone git@github.com:undark-lab/sstrax.git # for ssh client]

pip install sstrax
```
- This will install `sstrax` in the current python environment that is active on your system and will be available via `import sstrax`
- This library requires `jax` and `jaxlib` which will be installed as default from the requirements if they are not already present in your python installation
- *Note:* If you are interested in local development, it is probably advisable to run `pip install -e sstrax` instead which will then track the main directory rather than creating a copy

----
## Running `sstrax`

- Import the module with `import sstrax as st`
- To simply generate a stream with the default parameters given in [`sstrax/constants.py`](./sstrax/constants.py), run,
```
import jax, time
stream = sstrax.simulate_stream(key=jax.random.PRNGKey(time.time_ns()), 
                                params=st.Parameters())
```
- This outputs a list of stars in a jax array of shape `(6, N_stars)` where `N_stars` is a dynamically computed quantity based on the mass loss, and the columns are the phase space co-ordinates $(x, y, z, v_x, v_y, v_z)$ defined in the [`sstrax/projection.py`](./sstrax/projection.py) module. Projections to other co-ordinate frames (including the velocity jacobain transformations) are available in the same module.
- **Important:** One of the distinct features of `jax` compared to e.g. `numpy` is that you must specify and handle your own random seed generation. Most of this management is done internally, but to run an individual simulation, you must specify and generate your own key. The above example shows one way to do this simply by using the `time.time_ns()` command as a random seed. For debugging purposes, however, it might actually be preferable to fix the value e.g. `key = jax.random.PRNGKey(0)` so that the same realisation is generated every time.

----
## Current Implementation

The code is currently structured into the following files:

- [`background.py`](./sstrax/background.py)

----
## Release Details

- **v0.0.1** | *April 2023* | Initial release based on [arXiv:2004.xxxx]()
