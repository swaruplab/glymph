"""Glymph: platform-general ambient RNA removal by count thinning.

Named after the glymphatic system, which clears extracellular waste from the brain.
Prototype v0: per-library fit, one library per GPU. Not yet validated.
"""

from ._version import __version__

from .api import Config, remove_background  # noqa: E402,F401
from .io import read_counts  # noqa: E402,F401
