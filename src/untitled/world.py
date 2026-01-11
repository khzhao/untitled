# Copyright (c) 2025 untitled

import os

__all__ = ["World"]

_REQUIRED_ENV_VARS = [
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
]


class World:
    """System environment for distributed training."""

    def __init__(self):
        """Keeps track of the following system variables.

        - MASTER_ADDR: The address of the master node
        - MASTER_PORT: The port of the master node
        - RANK: The global process idx
        - WORLD_SIZE: The total number of processes
        - LOCAL_RANK: The process idx on the node
        - LOCAL_WORLD_SIZE: The number of processes on the node
        """
        self._check_env_vars()

        self.MASTER_ADDR = os.environ["MASTER_ADDR"]
        self.MASTER_PORT = os.environ["MASTER_PORT"]
        self.RANK = int(os.environ["RANK"])
        self.WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        self.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        self.LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    def _check_env_vars(self):
        for var in _REQUIRED_ENV_VARS:
            if var not in os.environ:
                raise ValueError(
                    f"Environment variable {var} is not set. Was this launched with torchrun?"
                )
