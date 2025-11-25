from setuptools import find_packages, setup

setup(
    name="latex2math",
    version="0.1.0",
    description="Physics LaTeX DSL to IR with NumPy/JAX/QuTiP backends",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "sympy",
        "numpy",
        "scipy",
        "qutip",
        "jax",
        "jaxlib",
        "antlr4-python3-runtime==4.11",
    ],
    extras_require={
        "qutip": ["qutip"],
        "jax": ["jax", "jaxlib"],
        "scqubits": ["scqubits"],
    },
    python_requires=">=3.13",
)
