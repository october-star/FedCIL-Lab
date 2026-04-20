from __future__ import annotations

import torch


def fedavg(state_dicts: list[dict], weights: list[float]) -> dict:
    """
    Weighted average of model parameters.
    """
    assert len(state_dicts) == len(weights)

    new_state = {}

    for key in state_dicts[0].keys():
        new_state[key] = sum(
            w * state_dicts[i][key] for i, w in enumerate(weights)
        )

    return new_state