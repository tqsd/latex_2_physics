=====================================
LaTeX Parser for Quantum Systems
=====================================

.. toctree::
   :hidden:
   :maxdepth: 1

   welcome
   install
   usage
   examples
   api

**Welcome to latex_parser** — a library for writing quantum models in physics-style LaTeX and compiling them to numerical backends.

👋 **New here?** Start with :doc:`welcome` for an overview and quick start.

🔧 **Ready to install?** Go to :doc:`install` for setup instructions.

📚 **Want to learn?** Check out :doc:`usage` for basics, or :doc:`examples` for 17 worked examples.

🎯 **Looking for the API?** See :doc:`api` for complete documentation.

---

.. rst-class:: center

**The Pipeline: LaTeX → IR → Backend**

.. code-block:: text

    Write in LaTeX:           Parse to IR:              Compile to Backend:
    ───────────────────────  ────────────────────────  ────────────────────
    H = ω σ_z +              HamiltonianIR             QuTiP Qobj
        A cos(ωt) σ_x    →   - terms: [...]       →   NumPy array
        + c_ops         - free_symbols: {ω, A}       JAX function
                          - time_dependent: True      Custom backend

**Key Features:**

- ✅ Physics-style LaTeX input
- ✅ Multiple backends (QuTiP, JAX, NumPy, custom)
- ✅ Time-dependent Hamiltonians & collapse operators
- ✅ Transparent intermediate representation (IR)
- ✅ Open-system support
- ✅ Boson deformations & custom subsystems

**Start with:** :doc:`welcome` or jump straight to :doc:`install`.
