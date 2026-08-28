"""Package bmopso: Binary Multiobjective Particle Swarm Optimization for pymoo.

Implements the BMOPSO algorithm originally proposed by Luciano S. de Souza,
Péricles B. C. de Miranda, Ricardo B. C. Prudêncio, and Flávia de A. Barros in:
"A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on
Functional Requirements Coverage and Execution Effort", 2011 23rd IEEE International
Conference on Tools with Artificial Intelligence (ICTAI 2011).
DOI: https://doi.org/10.1109/ICTAI.2011.45

BMOPSO was created by synthesizing:
1. Binary PSO (BPSO) (J. Kennedy and R. C. Eberhart, 1997)
2. MOPSO with Adaptive Hypercube Grid (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004)
3. Constrained-Dominance Principle (K. Deb, 2002)

In this library, BMOPSO is structured following the official pymoo framework
architecture (algorithms, operators, util).
"""

from .algorithms.bmopso import BMOPSO
from .util.archive import NonDominatedArchive
from .util.grid import AdaptiveGrid

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AdaptiveGrid",
    "BMOPSO",
    "NonDominatedArchive",
]
