"""课程地形按方向分列，楼梯形状按数量配额随机混排。"""

import numpy as np

from isaaclab.terrains import TerrainGenerator


def allocate_counts(weights, total):
    weights = np.asarray(weights, dtype=float)
    quotas = weights / weights.sum() * total
    counts = np.floor(quotas).astype(int)
    remainder = total - counts.sum()
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


class MixedStairsTerrainGenerator(TerrainGenerator):
    def _generate_curriculum_terrains(self):
        groups = {}
        configs = list(self.cfg.sub_terrains.values())
        for index, (name, cfg) in enumerate(self.cfg.sub_terrains.items()):
            group = next(
                (f"stairs_{direction}" for direction in ("up", "down") if name.endswith(f"stairs_{direction}")), name
            )
            if name in {"slope_up", "slope_down"}:
                group = "slopes"
            groups.setdefault(group, []).append(index)
        column_counts = allocate_counts(
            [sum(configs[i].proportion for i in ids) for ids in groups.values()], self.cfg.num_cols
        )
        column = 0
        self.stair_shape_layout = {}
        for (group, ids), num_columns in zip(groups.items(), column_counts):
            if num_columns == 0:
                continue
            counts = allocate_counts([configs[i].proportion for i in ids], num_columns * self.cfg.num_rows)
            assignments = np.repeat(ids, counts)
            self.np_rng.shuffle(assignments)
            assignments = assignments.reshape(num_columns, self.cfg.num_rows)
            self.stair_shape_layout[group] = assignments.copy()
            slope_difficulties = self.np_rng.uniform(size=self.cfg.num_rows) if group == "slopes" else None
            for local_col in range(num_columns):
                for row in range(self.cfg.num_rows):
                    cfg = configs[assignments[local_col, row]]
                    lower, upper = self.cfg.difficulty_range
                    offset = slope_difficulties[row] if slope_difficulties is not None else self.np_rng.uniform()
                    difficulty = lower + (upper - lower) * (row + offset) / self.cfg.num_rows
                    mesh, origin = self._get_terrain_mesh(difficulty, cfg)
                    self._add_sub_terrain(mesh, origin, row, column + local_col, cfg)
            column += num_columns
