"""Paper-method reconstruction using genuine MinkowskiEngine 4D sparse operators.

The paper does not publish channel widths, skip merge or temporal reduction code.
Those choices are explicitly recorded in README.md; this is not author source code.
"""

import os

import torch
from torch import nn

from ..geometry.sparse_voxel import SparsePrediction

os.environ.setdefault("OMP_NUM_THREADS", "4")

try:
    import MinkowskiEngine as ME
except (ImportError, OSError) as error:
    raise ImportError(
        "sparse4d requires a working MinkowskiEngine build. See training/README.md; "
        "use --model dense3d only when intentionally selecting the baseline."
    ) from error


class SparseBatchNorm(ME.MinkowskiBatchNorm):
    """Keep singleton sparse clouds valid without altering normal batch statistics."""

    def forward(self, tensor):
        if self.training and len(tensor) == 1:
            features = torch.nn.functional.batch_norm(
                tensor.F,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                eps=self.bn.eps,
            )
            return ME.SparseTensor(
                features=features,
                coordinate_map_key=tensor.coordinate_map_key,
                coordinate_manager=tensor.coordinate_manager,
            )
        return super().forward(tensor)


def sparse_block(in_channels, out_channels, kernel=(3, 3, 3, 3), stride=1):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=list(kernel), stride=stride, dimension=4),
        SparseBatchNorm(out_channels),
        ME.MinkowskiELU(),
    )


def convolve_at(conv, tensor, coordinates):
    # Passing raw coordinates to ME 0.5.4's convolution loses tensor_stride.
    # Register the explicit coordinate map with its true physical stride first.
    key, _ = tensor.coordinate_manager.insert_and_map(coordinates.contiguous(), tensor_stride=tensor.tensor_stride)
    return conv(tensor, coordinates=key)


class TemporalCollapse(nn.Module):
    """Learned source projection; two-plane mode preserves legacy checkpoint shapes."""

    def __init__(self, channels, source_planes=2):
        super().__init__()
        if source_planes not in (2, 3):
            raise ValueError("Expected two or three source planes")
        self.source_planes = source_planes
        self.conv = ME.MinkowskiConvolution(
            channels, channels, kernel_size=[1, 1, 1, source_planes], stride=1, bias=True, dimension=4
        )

    def forward(self, tensor):
        coordinates = tensor.C.to(tensor.F.device).clone()
        # Odd kernels are centered: query source 1 so a size-3 kernel sees 0,1,2.
        coordinates[:, 4] = 1 if self.source_planes == 3 else 0
        coordinates = torch.unique(coordinates, dim=0)
        merged = convolve_at(self.conv, tensor, coordinates)
        if self.source_planes == 2:
            return merged
        coordinates = merged.C.to(merged.F.device).clone()
        coordinates[:, 4] = 0
        return ME.SparseTensor(
            features=merged.F,
            coordinates=coordinates,
            tensor_stride=merged.tensor_stride,
            coordinate_manager=merged.coordinate_manager,
        )


class SparseSceneNet4D(nn.Module):
    """Four spatial downsamplings, time-preserving encoder, generative pruned decoder.

    No supervision labels enter forward. The trainer may seed the initial map
    from preceding-frame GT; later history is predicted.
    """

    def __init__(
        self,
        base_channels=16,
        resolution=64,
        threshold=0.5,
        time_planes=2,
        relative_pose=False,
    ):
        super().__init__()
        if resolution < 16 or resolution % 16:
            raise ValueError("sparse4d resolution must be divisible by 16 and at least 16")
        if time_planes != 2:
            raise ValueError("Sparse encoders require two sources")
        self.time_planes = time_planes
        self.pose_encoder = (
            nn.Sequential(nn.Linear(10, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU()) if relative_pose else None
        )
        self.resolution, self.threshold = resolution, threshold
        widths = [base_channels * 2**i for i in range(5)]
        kernel = (3, 3, 3, 3)
        self.stem = sparse_block(19 if relative_pose else 3, widths[0], kernel=kernel)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    sparse_block(widths[i], widths[i + 1], kernel=(2, 2, 2, 1), stride=[2, 2, 2, 1]),
                    sparse_block(widths[i + 1], widths[i + 1], kernel=kernel),
                )
                for i in range(4)
            ]
        )
        # Query the entire small coarse lattice so unknown regions can be generated.
        self.coarse = ME.MinkowskiConvolution(widths[-1], widths[-1], kernel_size=list(kernel), bias=True, dimension=4)
        self.collapse = nn.ModuleList([TemporalCollapse(width) for width in widths])
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    ME.MinkowskiGenerativeConvolutionTranspose(
                        widths[i + 1], widths[i], kernel_size=[2, 2, 2, 1], stride=[2, 2, 2, 1], dimension=4
                    ),
                    SparseBatchNorm(widths[i]),
                    ME.MinkowskiELU(),
                )
                for i in reversed(range(4))
            ]
        )
        self.refine = nn.ModuleList(
            [sparse_block(widths[i], widths[i], kernel=(3, 3, 3, 1)) for i in reversed(range(4))]
        )
        self.classifiers = nn.ModuleList(
            [ME.MinkowskiConvolution(widths[i], 1, kernel_size=1, bias=True, dimension=4) for i in reversed(range(4))]
        )
        self.offset_head = ME.MinkowskiConvolution(widths[0], 3, kernel_size=1, bias=True, dimension=4)
        self.prune = ME.MinkowskiPruning()
        # Start with candidates retained, avoiding an untrainable all-pruned decoder.
        for classifier in self.classifiers:
            nn.init.zeros_(classifier.kernel)
            nn.init.constant_(classifier.bias, 0.4)

    def _bounds(self, tensor):
        coordinates = tensor.C
        keep = ((coordinates[:, 1:4] >= 0) & (coordinates[:, 1:4] < self.resolution)).all(1)
        keep &= coordinates[:, 4] == 0
        return self.prune(tensor, keep.to(tensor.F.device)) if not bool(keep.all()) else tensor

    def forward(self, coordinates, features, relative_pose=None):
        if not len(coordinates):
            zero = self.offset_head.kernel.sum() * 0
            return SparsePrediction([], coordinates, zero.expand(0, 1), zero.expand(0, 3))
        if self.pose_encoder is not None:
            if relative_pose is None or relative_pose.ndim != 2 or relative_pose.shape[1] != 10:
                raise ValueError("Expected relative pose codes [batch,10]")
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
            axis, axis, axis, torch.arange(self.time_planes, dtype=torch.int32, device=axis.device)
        )
        # Preserve batch IDs throughout coarse completion; never mix independent maps.
        coarse = torch.cat(
            [
                torch.cat((torch.full((len(coarse), 1), batch, dtype=torch.int32, device=axis.device), coarse), dim=1)
                for batch in torch.unique(coordinates[:, 0]).tolist()
            ]
        )
        x = self.collapse[-1](convolve_at(self.coarse, x, coarse))
        stages = []
        for stage, (upsample, refine, classifier) in enumerate(
            zip(self.upsample, self.refine, self.classifiers, strict=True)
        ):
            skip_index = 3 - stage
            skip = self.collapse[skip_index](skips[skip_index])
            # Sparse addition aligns coordinate unions, unlike an unsafe ME.cat on different maps.
            x = self._bounds(upsample(x) + skip) if x is not None else skip
            x = refine(x)
            logits = classifier(x)
            if logits.tensor_stride != [2**skip_index] * 3 + [1]:
                raise RuntimeError(f"Incorrect sparse decoder stride: {logits.tensor_stride}")
            cells = logits.C.to(features.device)
            stages.append((cells, logits.F, 2**skip_index))
            if stage == 3:
                offsets = self.offset_head(x).F.sigmoid()
                return SparsePrediction(stages, cells, logits.F, offsets)
            keep = logits.F.flatten().sigmoid() >= self.threshold
            x = self.prune(x, keep) if bool(keep.any()) else None
        raise RuntimeError("Unreachable decoder state")
