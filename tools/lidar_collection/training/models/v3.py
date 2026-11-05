"""V3: observation/history (+ optional canvas), persistent GRUs, residual decoder."""

import torch
from torch import nn

from ..geometry.residual import residual_update
from ..geometry.sparse_voxel import SparsePrediction
from ..runtime.state import FeatureMemory
from .v1 import ME, SparseSceneNet4D, TemporalCollapse, convolve_at


class RecurrentFusion(nn.Module):
    def __init__(self, channels, source_planes=2):
        super().__init__()
        self.merge = TemporalCollapse(channels, source_planes)
        self.cell = nn.GRUCell(channels, channels)

    def forward(self, tensor, memory):
        merged = self.merge(tensor)
        current = merged.C[:, :4].to(merged.F.device)
        if memory is None:
            coordinates = current
            inputs = merged.F
            hidden = torch.zeros_like(inputs)
        else:
            coordinates, inverse = torch.unique(torch.cat((current, memory.coordinates)), dim=0, return_inverse=True)
            inputs = merged.F.new_zeros((len(coordinates), merged.F.shape[1])).index_add(
                0, inverse[: len(current)], merged.F
            )
            hidden = torch.zeros_like(inputs).index_add(0, inverse[len(current) :], memory.features)
        values = self.cell(inputs, hidden)
        coords4d = torch.cat((coordinates, torch.zeros_like(coordinates[:, :1])), dim=1)
        result = ME.SparseTensor(
            features=values,
            coordinates=coords4d,
            tensor_stride=merged.tensor_stride,
            coordinate_manager=merged.coordinate_manager,
        )
        return result, FeatureMemory(coordinates, values, merged.tensor_stride[0])


class RecurrentSceneNet(SparseSceneNet4D):
    def __init__(self, base_channels=16, resolution=64, threshold=0.5, relative_pose=False, canvas=False):
        # V1 encoder kernels also cover the optional third canvas source.
        super().__init__(base_channels, resolution, threshold, time_planes=2, relative_pose=relative_pose)
        self.source_planes = 3 if canvas else 2
        self.collapse = nn.ModuleList([RecurrentFusion(base_channels * 2**i, self.source_planes) for i in range(5)])
        nn.init.zeros_(self.classifiers[-1].bias)
        nn.init.zeros_(self.offset_head.kernel)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, coordinates, features, relative_pose, hidden, batch_size):
        raw_coordinates, raw_features = coordinates, features
        # Empty samples still need a coarse generative lattice and their recurrent memory.
        # Zero-feature anchors are computational support only, never prior occupied cells.
        missing = [i for i in range(batch_size) if not bool((coordinates[:, 0] == i).any())]
        if missing:
            anchors = torch.tensor([[i, 0, 0, 0, 0] for i in missing], device=coordinates.device, dtype=torch.int32)
            coordinates = torch.cat((coordinates, anchors))
            features = torch.cat((features, features.new_zeros((len(missing), 3))))
        if self.pose_encoder is not None:
            encoded = self.pose_encoder(relative_pose)
            features = torch.cat((features, encoded[coordinates[:, 0].long()]), dim=1)
        tensor = ME.SparseTensor(features=features, coordinates=coordinates.int())
        x = self.stem(tensor)
        skips = [x]
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        axis = torch.arange(0, self.resolution, 16, dtype=torch.int32, device=coordinates.device)
        coarse = torch.cartesian_prod(
            axis, axis, axis, torch.arange(self.source_planes, dtype=torch.int32, device=axis.device)
        )
        coarse = torch.cat(
            [
                torch.cat((torch.full((len(coarse), 1), i, dtype=torch.int32, device=axis.device), coarse), dim=1)
                for i in range(batch_size)
            ]
        )
        next_hidden = [None] * 5
        x, next_hidden[-1] = self.collapse[-1](convolve_at(self.coarse, x, coarse), hidden[-1])
        stages = []
        for stage, (upsample, refine, classifier) in enumerate(
            zip(self.upsample, self.refine, self.classifiers, strict=True)
        ):
            level = 3 - stage
            skip, next_hidden[level] = self.collapse[level](skips[level], hidden[level])
            x = self._bounds(upsample(x) + skip) if x is not None else skip
            x = refine(x)
            logits = classifier(x)
            cells = logits.C.to(features.device)
            if stage == 3:
                keep, logits, offsets, _ = residual_update(
                    cells,
                    logits.F,
                    self.offset_head(x).F,
                    raw_coordinates,
                    raw_features,
                    self.resolution,
                    threshold=self.threshold,
                    prior_source=1,
                )
                cells = cells[keep]
                stages.append((cells, logits, 1))
                return SparsePrediction(stages, cells, logits, offsets), next_hidden
            stages.append((cells, logits.F, 2**level))
            keep = logits.F.flatten().sigmoid() >= self.threshold
            x = self.prune(x, keep) if bool(keep.any()) else None
        raise RuntimeError("Unreachable recurrent decoder state")
