"""Compatibility imports for Q5 callers of the shared JAX memory settings."""

from pipeline.common.memory import (
    DEFAULT_MEMORY_FRACTION as DEFAULT_MEMORY_FRACTION,
    add_memory_argument as add_memory_argument,
    configure_memory_fraction as configure_memory_fraction,
    memory_fraction as memory_fraction,
    memory_settings as memory_settings,
)
