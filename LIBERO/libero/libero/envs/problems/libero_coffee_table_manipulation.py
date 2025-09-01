from robosuite.utils.mjcf_utils import new_site
from .base_table_manipulation import BaseTableManipulation
from libero.libero.envs.bddl_base_domain import register_problem
import numpy as np


@register_problem
class Libero_Coffee_Table_Manipulation(BaseTableManipulation):
    
    def _configure_workspace(self, **kwargs):
        """Configure coffee table specific workspace parameters"""
        self.workspace_name = "coffee_table"
        
        if "coffee_table_full_size" in kwargs:
            self.coffee_table_full_size = kwargs["coffee_table_full_size"]
        else:
            self.coffee_table_full_size = (0.70, 1.6, 0.024)
        
        self.coffee_table_offset = (0, 0, 0.41)
        self.z_offset = 0.01 - self.coffee_table_full_size[2]
        
        # Create new kwargs dictionary instead of updating the original
        workspace_kwargs = dict(kwargs)
        workspace_kwargs.update({
            "robots": [f"OnTheGround{robot_name}" for robot_name in kwargs["robots"]],
            "workspace_offset": self.coffee_table_offset,
            "arena_type": "coffee_table",
            "scene_xml": "scenes/libero_coffee_table_base_style.xml",
            "scene_properties": {
                "floor_style": "wood-plank",
                "wall_style": "light-gray-plaster",
            },
        })
        return workspace_kwargs

    def _get_table_region_name(self):
        return "coffee_table_table_region"

    def _get_excluded_fixture_category(self):
        return "coffee_table"

    def _is_table_region(self, region_name):
        return "coffee_table" in region_name

    def _get_zone_centroid(self, ranges):
        return (
            (ranges[2] + ranges[0]) / 2,
            (ranges[3] + ranges[1]) / 2,
        )

    def _get_table_z_offset(self, region_name):
        return 0.42 if 'table_region' in region_name else None

    def _append_table_site(self, mujoco_arena, target_zone):
        mujoco_arena.coffee_table_body.append(
            new_site(
                name=target_zone.name,
                pos=target_zone.pos,
                quat=target_zone.quat,
                rgba=target_zone.rgba,
                size=target_zone.size,
                type="box",
            )
        )

    def _setup_camera(self, mujoco_arena):
        """Setup coffee table specific cameras"""
        mujoco_arena.set_camera(
            camera_name="agentview", 
            pos=[1.2879197620252157, 0.0, 1.6155250360558634],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )

        # robosuite's default agentview camera configuration
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771, 0.0, 0.7903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

