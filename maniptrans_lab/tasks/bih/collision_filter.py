"""Disable hand-table collisions per env with FilteredPairsAPI."""

from pxr import UsdPhysics


def author_hand_table_filter(stage, num_envs: int):
    # CollisionGroup prims do not survive PhysX replication
    for i in range(num_envs):
        table = stage.GetPrimAtPath(f"/World/envs/env_{i}/table")
        assert table.IsValid(), f"table prim missing in env_{i}"
        api = UsdPhysics.FilteredPairsAPI.Apply(table)
        rel = api.CreateFilteredPairsRel()
        rel.AddTarget(f"/World/envs/env_{i}/dexhand_r")
        rel.AddTarget(f"/World/envs/env_{i}/dexhand_l")
