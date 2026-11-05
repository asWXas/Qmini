"""Explicit architecture selection; never substitute dense convolutions for sparse ones."""

DENSE_ARCHITECTURE = "dense_3d_unet_prediction_feedback_v1"
SPARSE_ARCHITECTURE = "minkowski_4d_scene_reconstruction_v1"
RECURRENT_ARCHITECTURE = "minkowski_recurrent_residual_v3"
ARCHITECTURES = {
    "dense3d": DENSE_ARCHITECTURE,
    "sparse4d_v1": SPARSE_ARCHITECTURE,
    "sparse4d_v3": RECURRENT_ARCHITECTURE,
}


def is_recurrent(config):
    return config.get("architecture") == RECURRENT_ARCHITECTURE


def is_sparse(config):
    return config.get("architecture", DENSE_ARCHITECTURE) in (
        SPARSE_ARCHITECTURE,
        RECURRENT_ARCHITECTURE,
    )


def make_model(config):
    architecture = config["architecture"]
    if architecture == RECURRENT_ARCHITECTURE:
        from .v3 import RecurrentSceneNet

        return RecurrentSceneNet(
            config["base_channels"],
            config["resolution"],
            config["threshold"],
            config.get("relative_pose", True),
            config.get("canvas", False),
        )
    if architecture == DENSE_ARCHITECTURE:
        from .dense3d import TerrainUNet

        return TerrainUNet(config["base_channels"])
    if architecture == SPARSE_ARCHITECTURE:
        from .v1 import SparseSceneNet4D

        return SparseSceneNet4D(config["base_channels"], config["resolution"], config["threshold"])
    if architecture == "minkowski_sparse_gru_residual_v2":
        raise ValueError("v2 has been removed; select sparse4d_v1 or sparse4d_v3 and use a matching checkpoint")
    raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
