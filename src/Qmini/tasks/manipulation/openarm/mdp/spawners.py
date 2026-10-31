"""Task-local collision geometry overrides for the OpenArm USD asset."""


def spawn_openarm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Preserve wrist concavities so link5/link7 do not collide in the nominal pose.

    The source USD uses single convex hulls for these links. Apply decomposition
    on the source before scene replication; leave the installed asset untouched.
    """
    # Import USD-backed code only after Kit has selected its USD runtime.
    from pxr import UsdPhysics

    from isaaclab.sim.spawners.from_files import spawn_from_usd

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for name in ("openarm_link5", "openarm_link7"):
        path = f"{root.GetPath()}/{name}/collisions"
        collision = root.GetStage().GetPrimAtPath(path)
        if not collision.IsValid():
            raise ValueError(f"Expected OpenArm wrist collision geometry at {path}")
        UsdPhysics.MeshCollisionAPI(collision).GetApproximationAttr().Set("convexDecomposition")
    return root
