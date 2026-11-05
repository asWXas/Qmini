"""Unrestricted v3 residual updates on the aligned two-source prior map."""

import math

import torch


def keys(coordinates, resolution):
    b, x, y, z = coordinates[:, :4].long().unbind(1)
    return ((b * resolution + x) * resolution + y) * resolution + z


def residual_update(
    candidate_coordinates,
    delta_logits,
    delta_offsets,
    input_coordinates,
    input_features,
    resolution,
    prior_logit=2.0,
    threshold=0.5,
    prior_source=1,
):
    history = input_coordinates[:, 4] == prior_source
    history_coordinates, history_offsets = input_coordinates[history], input_features[history]
    query = keys(candidate_coordinates, resolution)
    prior_keys, order = keys(history_coordinates, resolution).sort()
    occupied = torch.zeros(len(query), device=query.device, dtype=torch.bool)
    offsets = torch.full_like(delta_offsets, 0.5)
    if len(prior_keys):
        index = torch.searchsorted(prior_keys, query).clamp_max(len(prior_keys) - 1)
        occupied = prior_keys[index] == query
        offsets[occupied] = history_offsets[order[index[occupied]]]
    update = torch.ones(len(query), dtype=torch.bool, device=query.device)
    keep = update
    center = math.log(threshold / (1 - threshold))
    base_logits = center + torch.where(occupied[:, None], prior_logit, -prior_logit).to(delta_logits.dtype)
    logits = base_logits + torch.where(update[:, None], delta_logits, 0.0)
    corrected = (offsets + delta_offsets.tanh()).clamp(0, 1 - 1e-6)
    corrected = torch.where(update[:, None], corrected, offsets)
    return keep, logits[keep], corrected[keep], update[keep]
