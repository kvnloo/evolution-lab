"""Evolution Lab: experiments are the population."""

from .targets import PRODUCT_TARGET

__version__ = "0.1.0"
GOAL = (
    "Continuously discover, verify, and explain computational systems "
    "that expand the achievable frontier between capability and resources."
)

__all__ = ["GOAL", "PRODUCT_TARGET", "__version__"]
