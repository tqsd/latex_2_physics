==================
Installation Guide
==================

Prerequisites
=============

- **Python 3.8+** (3.10+ recommended for best performance)
- **pip** or **conda** (for package management)

Basic Installation
==================

Using pip
---------

The simplest way to install ``latex_parser`` with QuTiP backend support:

.. code-block:: bash

    pip install latex-parser

This installs the core library and QuTiP (the default backend).

Using conda
-----------

If you prefer conda:

.. code-block:: bash

    conda install -c conda-forge latex-parser

Verify Installation
-------------------

Test that everything is working:

.. code-block:: python

    from latex_parser.latex_api import compile_model
    
    # Simple test
    H = r"\sigma_z"
    model = compile_model(H_latex=H, params={}, qubits=1, backend="qutip")
    print("✅ Installation successful!")
    print(f"Backend: {model.__class__.__name__}")

Optional Dependencies
=====================

Depending on which backends you want to use, install optional packages:

JAX Backend
-----------

For automatic differentiation and GPU acceleration:

.. code-block:: bash

    pip install jax jaxlib

To use GPU:

.. code-block:: bash

    # CUDA 11
    pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Or follow: https://github.com/google/jax#installation

Verify JAX is available:

.. code-block:: python

    import jax
    print(f"JAX version: {jax.__version__}")
    
    model = compile_model(H_latex=r"\sigma_z", params={}, qubits=1, backend="jax")
    print("✅ JAX backend working")

NumPy Backend
-------------

The NumPy backend is already included with the core installation (NumPy is a dependency).
No additional setup needed:

.. code-block:: python

    model = compile_model(H_latex=r"\sigma_z", params={}, qubits=1, backend="numpy")
    print("✅ NumPy backend working")

QuTiP Backend (Default)
-----------------------

QuTiP is installed by default, but you can ensure it's up-to-date:

.. code-block:: bash

    pip install --upgrade qutip

For advanced QuTiP features:

.. code-block:: bash

    pip install qutip[ipython,graphics]  # Jupyter support, plotting tools

Development Installation
=========================

If you want to contribute or modify the source code:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/yourusername/latex_parser.git
    cd latex_parser

2. Install in editable mode with all optional dependencies:

.. code-block:: bash

    pip install -e ".[all]"

3. Install testing dependencies:

.. code-block:: bash

    pip install pytest pytest-cov tox

4. Run tests to verify:

.. code-block:: bash

    pytest tests/

Docker (Optional)
=================

If you prefer containerization:

.. code-block:: bash

    docker build -t latex-parser .
    docker run -it latex-parser python -c "from latex_parser.latex_api import compile_model; print('✅ Working')"

Docker with GPU (JAX)
---------------------

For GPU-accelerated JAX:

.. code-block:: bash

    docker build -f Dockerfile.gpu -t latex-parser:gpu .
    docker run -it --gpus all latex-parser:gpu

Troubleshooting
===============

**"ModuleNotFoundError: No module named 'qutip'"**

Install QuTiP:

.. code-block:: bash

    pip install qutip

**"AttributeError: module 'jax' has no attribute 'grad'"**

JAX is not installed. Install it:

.. code-block:: bash

    pip install jax jaxlib

**"SymPy parse error"**

Ensure SymPy is installed and up-to-date:

.. code-block:: bash

    pip install --upgrade sympy

**Compilation is slow**

This is normal for large systems. Some tips:

- Use smaller subsystems while prototyping
- Switch to JAX for automatic differentiation (it's faster for complex Hamiltonians)
- Pre-compile models and cache results if you're running many simulations

**"DSLValidationError: Missing numeric value..."**

Parameter names don't match. Check:

- Parameter spelling (e.g., ``omega_c`` vs ``omega_c1``)
- Braces and underscores (``\omega_{c}`` → ``omega_c``)
- See the :doc:`usage` guide for parameter aliasing rules

Verifying All Backends
======================

Run this script to test all installed backends:

.. code-block:: python

    from latex_parser.latex_api import compile_model
    import numpy as np
    
    H = r"\omega \sigma_z + A \sigma_x"
    params = {"omega": 1.0, "A": 0.5}
    
    backends_to_test = ["qutip", "numpy", "jax"]
    
    for backend in backends_to_test:
        try:
            model = compile_model(
                H_latex=H,
                params=params,
                qubits=1,
                backend=backend
            )
            print(f"✅ {backend:10s} backend working")
        except ImportError as e:
            print(f"❌ {backend:10s} backend not installed: {e}")
        except Exception as e:
            print(f"❌ {backend:10s} backend error: {e}")

System-Specific Notes
=====================

**macOS (M1/M2 Silicon)**

JAX on ARM may require special handling. Install with:

.. code-block:: bash

    # Use conda for better compatibility
    conda install -c conda-forge jax

**Windows**

JAX on Windows is officially supported as of JAX 0.3.0. Standard pip install should work:

.. code-block:: bash

    pip install jax jaxlib

**Linux (GPU)**

For NVIDIA GPUs with CUDA:

.. code-block:: bash

    pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    export CUDA_VISIBLE_DEVICES=0  # Select GPU 0
    export JAX_PLATFORM_NAME=gpu

**Linux (CPU-only)**

To prevent accidental GPU usage:

.. code-block:: bash

    export JAX_PLATFORM_NAME=cpu

Next Steps
==========

After installation:

1. Read the :doc:`usage` guide for basic examples
2. Explore the :doc:`examples` (17 worked examples)
3. Check the :doc:`api` reference for detailed documentation

---

**Problems?** Open an issue on GitHub or check :doc:`usage` for troubleshooting.
