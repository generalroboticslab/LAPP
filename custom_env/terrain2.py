# terrain.py

import numpy as np
from isaacgym.terrain_utils import (
    SubTerrain, pyramid_stairs_terrain, convert_heightfield_to_trimesh,
    # [XW]
    # ---------------------------------------------------------------------------------
    # Import essential functions for generating terrains
    pyramid_sloped_terrain,
    discrete_obstacles_terrain,
    wave_terrain,
    stepping_stones_terrain
    # ---------------------------------------------------------------------------------
)


class Terrain2:
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

        # [XW]
        # ---------------------------------------------------------------------------------
        # These variables are used by original repo's implementation
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # ---------------------------------------------------------------------------------

        # Generate the selected terrain
        # [XW]
        # ---------------------------------------------------------------------------------
        # Include
        # pyramid_sloped,
        # discrete_obstacles,
        # wave,
        # stepping_stone into terrain_type
        if cfg.selected:
            if cfg.terrain_type == 'pyramid_sloped':
                print("@@@ Building pyramid sloped terrain")
                self._generate_pyramid_sloped_terrain(cfg)
            elif cfg.terrain_type == 'pyramid_stairs':
                print("@@@ Building pyramid stairs terrain")
                self._generate_pyramid_stairs_terrain(cfg)
            elif cfg.terrain_type == 'discrete_obstacles':
                print("@@@ Building discrete obstacles terrain")
                self._generate_discrete_obstacles_terrain(cfg)
            elif cfg.terrain_type == 'wave':
                print("@@@ Building wave terrain")
                self._generate_wave_terrain(cfg)
            elif cfg.terrain_type == 'stepping_stones':
                print("@@@ Stepping stones currently not supported")
                # self._generate_stepping_stones_terrain(cfg)

            elif cfg.terrain_type == 'curriculum_slope':
                print("@@@ Building slope with curriculum")
                self._generate_curriculum_slope_terrain(cfg)

            elif cfg.terrain_type == 'curriculum_obs':
                print("@@@ Building discrete obstacles with curriculum")
                self._generate_curriculum_obstacles_terrain(cfg)

            elif cfg.terrain_type == 'curriculum_wave':
                print("@@@ Building wave with curriculum")
                self._generate_curriculum_wave_terrain(cfg)

        # ---------------------------------------------------------------------------------
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

    def _generate_pyramid_stairs_terrain(self, cfg):
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
            print(f"@@@ Difficulty Level {level_idx}: Step Height {step_height}, Num Rows {num_rows}")

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

    # [XW]
    # ---------------------------------------------------------------------------------
    def _generate_pyramid_sloped_terrain(self, cfg):
        # [XW]
        # This function mainly generates the pyramid sloped terrain
        # Variables: slope, platform_size
        for k in range(self.num_tiles):
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))
            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )
            pyramid_sloped_terrain(terrain, slope=.5, platform_size=2.)

            self._add_terrain_to_map(terrain, i, j)

        self.env_origins = np.array(self.env_origins)

    def _generate_curriculum_slope_terrain(self, cfg):
        terrain_params = cfg.terrain_kwargs.copy()
        terrain_type = cfg.terrain_type
        difficulty_levels = cfg.difficulty_levels

        # Calculate rows per difficulty level
        total_levels = len(difficulty_levels)
        level_rows = self.env_rows // total_levels
        remainder_rows = self.env_rows % total_levels  # Handle any remainder rows

        row_counter = 0  # Keep track of the current row index
        self.env_origins = []

        for level_idx, difficulty in enumerate(difficulty_levels):
            # Determine the number of rows for this difficulty level
            num_rows = level_rows + 1 if level_idx < remainder_rows else level_rows

            slope = 0.1 + difficulty * 0.1
            platform_size = 5. - difficulty * 0.75
            print(f"@@@ Difficulty level {level_idx} === Slope {slope}, Platform Size {platform_size}")

            for i in range(num_rows):
                for j in range(self.env_cols):
                    terrain = SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale
                    )

                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)

                    self._add_terrain_to_map(terrain, row_counter, j)

                row_counter += 1  # Move to the next row

        self.env_origins = np.array(self.env_origins)

    def _generate_discrete_obstacles_terrain(self, cfg):
        # [XW]
        # This function mainly generates the discrete obstacles terrain
        # Variables: discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size
        for k in range(self.num_tiles):
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))
            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )
            # XW: temp max height 0.25 -> 0.2
            discrete_obstacles_terrain(terrain, max_height=0.14, min_size=1., max_size=2., num_rects=20,
                                       platform_size=3.)  # original max_height=0.2, then 0.14. I use 0.03 only for using it as domain randomization for real robot
            self._add_terrain_to_map(terrain, i, j)

        self.env_origins = np.array(self.env_origins)

    def _generate_curriculum_obstacles_terrain(self, cfg):
        terrain_params = cfg.terrain_kwargs.copy()
        terrain_type = cfg.terrain_type
        difficulty_levels = cfg.difficulty_levels

        # Calculate rows per difficulty level
        total_levels = len(difficulty_levels)
        level_rows = self.env_rows // total_levels
        remainder_rows = self.env_rows % total_levels  # Handle any remainder rows

        row_counter = 0  # Keep track of the current row index
        self.env_origins = []

        for level_idx, difficulty in enumerate(difficulty_levels):
            # Determine the number of rows for this difficulty level
            num_rows = level_rows + 1 if level_idx < remainder_rows else level_rows

            max_height = 0.04 + difficulty * 0.025  # originally 0.06 + difficulty * 0.025
            min_size = 0.6 + difficulty * 0.1
            max_size = 1.2 + difficulty * 0.2
            num_rects = int(12 + difficulty * 2)
            platform_size = 5. - difficulty * 0.75
            print(
                f"@@@ Difficulty level {level_idx} === max_height {max_height}, min_size {min_size}, max_size {max_size}, num_rects {num_rects}, platform_size {platform_size}")

            for i in range(num_rows):
                for j in range(self.env_cols):
                    terrain = SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale
                    )

                    discrete_obstacles_terrain(terrain, max_height=max_height,
                                               min_size=min_size, max_size=max_size,
                                               num_rects=num_rects, platform_size=platform_size)

                    self._add_terrain_to_map(terrain, row_counter, j)

                row_counter += 1  # Move to the next row

        self.env_origins = np.array(self.env_origins)

    def _generate_wave_terrain(self, cfg):
        # [XW]
        # This function mainly generates the pyramid sloped terrain
        # Variables: num_waves, amplitude
        for k in range(self.num_tiles):
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))
            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )
            # XW: temp amplitude 0.5 -> 0.35
            wave_terrain(terrain, num_waves=3, amplitude=.34)
            self._add_terrain_to_map(terrain, i, j)

        self.env_origins = np.array(self.env_origins)

    def _generate_curriculum_wave_terrain(self, cfg):
        terrain_params = cfg.terrain_kwargs.copy()
        terrain_type = cfg.terrain_type
        difficulty_levels = cfg.difficulty_levels

        # Calculate rows per difficulty level
        total_levels = len(difficulty_levels)
        level_rows = self.env_rows // total_levels
        remainder_rows = self.env_rows % total_levels  # Handle any remainder rows

        row_counter = 0  # Keep track of the current row index
        self.env_origins = []

        for level_idx, difficulty in enumerate(difficulty_levels):
            # Determine the number of rows for this difficulty level
            num_rows = level_rows + 1 if level_idx < remainder_rows else level_rows

            num_waves = 0
            if difficulty < 1.1:
                num_waves = 1
            elif difficulty < 3.1:
                num_waves = 2
            else:
                num_waves = 3
            amplitude = .2 + difficulty * .035
            print(f"@@@ Difficulty level {level_idx} === num_waves {num_waves}, amplitude {amplitude}")

            for i in range(num_rows):
                for j in range(self.env_cols):
                    terrain = SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale
                    )

                    wave_terrain(terrain, num_waves=num_waves, amplitude=amplitude)

                    self._add_terrain_to_map(terrain, row_counter, j)

                row_counter += 1  # Move to the next row

        self.env_origins = np.array(self.env_origins)

    def _generate_stepping_stones_terrain(self, cfg):
        # [XW]
        # This function mainly generates the pyramid sloped terrain
        # Variables:
        # stepping_stones_size = 1.5 * (1.05 - difficulty)
        # stone_distance = 0.05 if difficulty==0 else 0.1
        # stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        for k in range(self.num_tiles):
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))
            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )
            stepping_stones_terrain(terrain, stone_size=0.4, stone_distance=1.25, max_height=0.7, platform_size=4.,
                                    depth=-0.5)
            self._add_terrain_to_map(terrain, i, j)

        self.env_origins = np.array(self.env_origins)
        return

    def _add_terrain_to_map(self, terrain, i, j):
        # [XW]
        # This function is transplanted by original repo
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

        self.env_origins.append([env_origin_x, env_origin_y, env_origin_z])

    # ---------------------------------------------------------------------------------

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
