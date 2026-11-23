"""lineage-check: tell whether a model is a modified copy of a listed open-weight model, from its weights alone."""
__version__ = "0.1.0"

import warnings

# Weights are read as read-only buffers and never written to; PyTorch warns about that once per process.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
