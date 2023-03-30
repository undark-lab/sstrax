from setuptools import setup

setup(
    name="sstrax",
    version="0.0.1",
    description="sstrax: a pure jax implementation for modelling stellar streams",
    url="https://github.com/undark-lab/sstrax",
    author="James Alvey, Mathis Gerdes",
    author_email="j.b.g.alvey@uva.nl, m.gerdes@uva.nl",
    packages=["sstrax"],
    install_requires=[
        "jax==0.4.1",
        "jaxlib==0.4.1",
        "jaxtyping>=0.2.11",
        "diffrax>=0.2.2",
        "chex==0.1.6",
        "numpy",
    ],
)
