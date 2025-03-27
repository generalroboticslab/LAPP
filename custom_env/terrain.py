# terrain.py

import numpy as np
from isaacgym.terrain_utils import (
    SubTerrain, pyramid_stairs_terrain, convert_heightfield_to_trimesh
)

class Terrain:
    def __init__(self, cfg, num_robots):
        self.type = cfg.terrain_type
        if self.type in ["none", "plane"]:
            return
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        self.border_size = cfg.border_size
        self.num_envs = num_robots
        self.env_rows = cfg.num_rows
        self.env_cols = cfg.num_cols
        self.num_tiles = self.env_rows * self.env_cols
        self.env_origins = []

        self.width_per_env_pixels = int(cfg.terrain_width / self.horizontal_scale)
        self.length_per_env_pixels = int(cfg.terrain_length / self.horizontal_scale)
        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = self.env_cols * self.width_per_env_pixels + 2 * self.border
        self.tot_rows = self.env_rows * self.length_per_env_pixels + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # Generate the selected terrain
        if cfg.selected:
            self._generate_selected_terrain(cfg)
        else:
            # Default to a flat terrain if no specific terrain is selected
            self._generate_flat_terrain()

        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw,
            self.horizontal_scale,
            self.vertical_scale,
            slope_threshold=cfg.slope_threshold
        )

    def _generate_selected_terrain(self, cfg):
        terrain_params = cfg.terrain_kwargs.copy()
        terrain_type = cfg.terrain_type
        difficulty_levels = cfg.difficulty_levels

        # Calculate rows per difficulty level
        total_levels = len(difficulty_levels)
        level_rows = self.env_rows // total_levels
        remainder_rows = self.env_rows % total_levels  # Handle any remainder rows

        row_counter = 0  # Keep track of the current row index
        self.env_origins = []

        for level_idx, step_height in enumerate(difficulty_levels):
            # Determine the number of rows for this difficulty level
            num_rows = level_rows + 1 if level_idx < remainder_rows else level_rows
            print(f"Difficulty Level {level_idx}: Step Height {step_height}, Num Rows {num_rows}")

            # Update terrain parameters for this difficulty level
            terrain_params['step_height'] = step_height

            for i in range(num_rows):
                for j in range(self.env_cols):
                    terrain = SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale
                    )

                    if terrain_type == 'pyramid_stairs':
                        pyramid_stairs_terrain(terrain, **terrain_params)
                    elif terrain_type == 'plane':
                        pass  # Already flat
                    else:
                        raise ValueError(f"Unknown terrain type: {terrain_type}")

                    # Calculate the actual row index
                    row = row_counter

                    # Adjust start and end indices based on the row and column
                    start_x = self.border + row * self.width_per_env_pixels
                    end_x = start_x + self.width_per_env_pixels
                    start_y = self.border + j * self.length_per_env_pixels
                    end_y = start_y + self.length_per_env_pixels

                    # Assign the terrain to the correct location in height_field_raw
                    self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                    # Calculate environment origins
                    env_origin_x = (start_x + end_x) * self.horizontal_scale / 2.0 - self.border_size
                    env_origin_y = (start_y + end_y) * self.horizontal_scale / 2.0 - self.border_size
                    env_origin_z = 0.0  # Adjust based on terrain height if necessary
                    self.env_origins.append([env_origin_x, env_origin_y, env_origin_z])

                row_counter += 1  # Move to the next row

        self.env_origins = np.array(self.env_origins)

    def _generate_flat_terrain(self):
        """Generates a flat terrain as the default terrain."""
        terrain = SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale
        )
        # For a flat terrain, the height field remains zero

        start_x = self.border
        end_x = self.border + self.width_per_env_pixels
        start_y = self.border
        end_y = self.border + self.length_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (self.width_per_env_pixels * self.horizontal_scale) / 2.0
        env_origin_y = (self.length_per_env_pixels * self.horizontal_scale) / 2.0
        env_origin_z = 0.0  # Adjust if needed based on terrain heights
        self.env_origins = np.array([[env_origin_x, env_origin_y, env_origin_z] for _ in range(self.num_envs)])
