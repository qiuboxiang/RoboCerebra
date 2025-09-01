from robosuite.utils.mjcf_utils import new_site

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Living_Room_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "living_room_table"
        self.visualization_sites_list = []
        self._task_progress = {}
        self._pour_logs = {} 
        self.pour_distance_threshold = 0.3 
        self.pour_quat_threshold = 0.1 
        if "living_room_table_full_size" in kwargs:
            self.living_room_table_full_size = living_room_table_full_size
        else:
            self.living_room_table_full_size = (0.70, 1.6, 0.024)
        self.living_room_table_offset = (0, 0, 0.41)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.living_room_table_full_size[2]
        self.check_inside = [
            'basket_1_contain_region',
            'microwave_1_top_side',
            'microwave_1_heating_region',
            'short_cabinet_1_middle_region',
            'short_cabinet_1_top_region',
            'short_cabinet_1_bottom_region',
            'short_fridge_1_upper_region',
            'short_fridge_1_middle_region',
            'short_fridge_1_lower_region',
            'wooden_cabinet_1_top_side',
            'wooden_cabinet_1_top_region',
            'wooden_cabinet_1_middle_region',
            'wooden_cabinet_1_bottom_region',
            'white_cabinet_1_top_side',
            'white_cabinet_1_top_region',
            'white_cabinet_1_middle_region',
            'white_cabinet_1_bottom_region',
            'white_storage_box_1_top_side',
            'white_storage_box_1_bottom_side',
            'white_storage_box_1_right_side',
            'white_storage_box_1_left_side',
            'wooden_shelf_1_top_side',
            'wooden_shelf_1_top_region',
            'wooden_shelf_1_middle_region',
            'wooden_shelf_1_bottom_region',
            'wooden_two_layer_shelf_1_top_side',
            'wooden_two_layer_shelf_1_top_region',
            'wooden_two_layer_shelf_1_bottom_region',
            'wine_rack_1_top_region',
            'bowl_drainer_1_left_region',
            'bowl_drainer_1_right_region',
            'living_room_table_table_region',
        ]
        kwargs.update(
            {"robots": [f"OnTheGround{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.living_room_table_offset})
        kwargs.update({"arena_type": "living_room"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_living_room_tabletop_base_style.xml"}
            )
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "wood-plank",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)
        self.regions = list(self.parsed_problem['regions'].keys())
        self.object_names = list(self.objects_dict.keys())
        self._has_left_table = {obj: False for obj in self.object_names}

        self._last_regions = {obj: None for obj in self.object_names}

        self._location_log = {obj: []   for obj in self.object_names}
        self.regions = [x for x in self.check_inside if x in self.regions]

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "living_room_table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "living_room_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                if 'table_region' in object_region_name:
                    target_zone = TargetZone(
                        z_offset=0.46,
                        name=object_region_name,
                        rgba=region_dict[object_region_name]["rgba"],
                        zone_size=zone_size,
                        zone_centroid_xy=zone_centroid_xy,
                    )
                else:
                    target_zone = TargetZone(
                        name=object_region_name,
                        rgba=region_dict[object_region_name]["rgba"],
                        zone_size=zone_size,
                        zone_centroid_xy=zone_centroid_xy,
                    )    
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.living_room_table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):

        mapping_inv = {}
        for k, values in self.parsed_problem["fixtures"].items():
            for v in values:
                mapping_inv[v] = k
        for k, values in self.parsed_problem["objects"].items():
            for v in values:
                mapping_inv[v] = k

        regions = self.parsed_problem["regions"]
        initial_state = self.parsed_problem["initial_state"]
        problem_name = self.parsed_problem["problem_name"]

        conditioned_initial_place_state_on_sites = []
        conditioned_initial_place_state_on_objects = []
        conditioned_initial_place_state_in_objects = []

        for state in initial_state:
            if state[0] == "on" and state[2] in self.objects_dict:
                conditioned_initial_place_state_on_objects.append(state)
                continue

            # (Yifeng) Given that an object needs to have a certain "containing" region in order to hold the relation "In", we assume that users need to specify the containing region of the object already.
            if state[0] == "in" and state[2] in regions:
                conditioned_initial_place_state_in_objects.append(state)
                continue
            # Check if the predicate is in the form of On(object, region)
            if state[0] == "on" and state[2] in regions:
                object_name = state[1]
                region_name = state[2]
                target_name = regions[region_name]["target"]
                x_ranges, y_ranges = rectangle2xyrange(regions[region_name]["ranges"])
                yaw_rotation = regions[region_name]["yaw_rotation"]
                if (
                    target_name in self.objects_dict
                    or target_name in self.fixtures_dict
                ):
                    conditioned_initial_place_state_on_sites.append(state)
                    continue
                if self.is_fixture(object_name):
                    # This is to place environment fixtures.
                    fixture_sampler = MultiRegionRandomSampler(
                        f"{object_name}_sampler",
                        mujoco_objects=self.fixtures_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=yaw_rotation,
                        rotation_axis="z",
                        z_offset=self.z_offset,  # -self.table_full_size[2],
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(fixture_sampler)
                else:
                    # This is to place movable objects.
                    region_sampler = get_region_samplers(
                        problem_name, mapping_inv[target_name]
                    )(
                        object_name,
                        self.objects_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=self.objects_dict[object_name].rotation,
                        rotation_axis=self.objects_dict[object_name].rotation_axis,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(region_sampler)
            if state[0] in ["open", "close"]:
                # If "open" is implemented, we assume "close" is also implemented
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "open":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_open_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_close_ranges"
                        ]

                    property_initializer = OpenCloseSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)
            elif state[0] in ["turnon", "turnoff"]:
                # If "turnon" is implemented, we assume "turnoff" is also implemented.
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "turnon":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnon_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnoff_ranges"
                        ]

                    property_initializer = TurnOnOffSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)

        # Place objects that are on sites
        for state in conditioned_initial_place_state_on_sites:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = SiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )
        # Place objects that are on other objects
        for state in conditioned_initial_place_state_on_objects:
            object_name = state[1]
            other_object_name = state[2]
            sampler = ObjectBasedSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[0.0, 0.0]],
                y_ranges=[[0.0, 0.0]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_on_objects_initializer.append_sampler(
                sampler, {"reference": other_object_name}
            )
        # Place objects inside some containing regions
        for state in conditioned_initial_place_state_in_objects:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]

            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = InSiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                # x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                # y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )

    def _check_success(self, monitor_dict):
        """
        Parameters
        ----------
        monitor_dict : dict
            {object_name: [[pred, obj, region], …]}
            Only monitor objects in the dictionary, evaluating states sequentially for each object.

        Returns
        -------
        dict
            {object_name: completion_rate (0-100)}
        """


        if not hasattr(self, "_state_progress"):
            self._state_progress = {obj: 0 for obj in monitor_dict}
        if not hasattr(self, "_last_regions"):
            self._last_regions  = {obj: None for obj in self.object_names}
        if not hasattr(self, "_location_log"):
            self._location_log  = {obj: []   for obj in self.object_names}

        object_locations = {obj: [] for obj in self.object_names}
        for region in self.regions:
            for obj in self.object_names:
                # print(['in', obj, region])
                if self._eval_predicate(['in', obj, region]):
                    object_locations[obj].append(region)
        for obj, regs in object_locations.items():
            if not regs:
                object_locations[obj] = ['None']
        print(object_locations['chocolate_pudding_1'])

        for obj, curr_regs in object_locations.items():
            last_regs = self._last_regions[obj]
            if last_regs is None or set(curr_regs) != set(last_regs):
                self._location_log[obj].append({
                    'time': self.sim.data.time,
                    'regions': curr_regs.copy()
                })
                self._last_regions[obj] = curr_regs.copy()

        completion = {}
        for obj, state_list in monitor_dict.items():
            curr_idx = self._state_progress[obj]
            total   = len(state_list)

            if curr_idx >= total:
                print(f"[Monitor] {obj} completed all states ({total}/{total})")
                completion[obj] = 100.0
                continue

            print(f"[Monitor] {obj} monitoring state {curr_idx + 1}/{total}: {state_list[curr_idx]}")

            pred, _, region = state_list[curr_idx]

            if region == "living_room_table_table_region": 
                in_table = self._eval_predicate(['in', obj, region])

                if not self._has_left_table[obj]:
                    if not in_table:
                        self._has_left_table[obj] = True
                        print(f"[Monitor] {obj} has left table area, will only count as completed when re-entering")
                    success = False
                else:
                    success = in_table
                    if success:
                        print(f"[Monitor] {obj} returned to table area, step judged successful")
            else:
                success = self._eval_predicate(state_list[curr_idx])

            if success:
                self._state_progress[obj] += 1
                print(f"[Monitor] {obj} step {curr_idx + 1} completed, progress {self._state_progress[obj]}/{total}")

            completion[obj] = self._state_progress[obj] / total * 100.0

        return completion


        

    def _eval_predicate(self, state):
        if len(state) == 3 and state[0] == 'pour':
            obj1_name, obj2_name = state[1], state[2]
            key = (obj1_name, obj2_name)

            geom1 = self.object_states_dict[obj1_name].get_geom_state()
            geom2 = self.object_states_dict[obj2_name].get_geom_state()
            pos1, quat1 = geom1['pos'], geom1['quat']
            pos2, quat2 = geom2['pos'], geom2['quat']

            logs = self._pour_logs.setdefault(key, [])

            logs.append({
                'pos1': pos1.copy(), 'quat1': quat1.copy(),
                'pos2': pos2.copy(), 'quat2': quat2.copy()
            })

            if len(logs) > 40:
                self._pour_logs[key] = logs = logs[-40:]

            if len(logs) == 40:
                import numpy as np

                p1 = np.stack([e['pos1'] for e in logs])
                p2 = np.stack([e['pos2'] for e in logs])
                dists = np.linalg.norm(p1 - p2, axis=1)
                min_dist = dists.min()

                q1 = np.stack([e['quat1'] for e in logs])
                quat_diff = np.linalg.norm(q1[-1] - q1[0])

                if (min_dist < self.pour_distance_threshold
                        and quat_diff > self.pour_quat_threshold):
                    return True
                if obj1_name=='chocolate_pudding_1' and obj2_name=='porcelain_mug_1':
                    print(min_dist, quat_diff)
            return False

        if len(state) == 3:
            predicate_fn_name, object_1_name, object_2_name = state
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            predicate_fn_name, object_name = state
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_name]
            )
        else:
            raise ValueError(f"Unsupported state format: {state}")

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

from robosuite.utils.mjcf_utils import new_site

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Living_Room_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "living_room_table"
        self.visualization_sites_list = []
        if "living_room_table_full_size" in kwargs:
            self.living_room_table_full_size = living_room_table_full_size
        else:
            self.living_room_table_full_size = (0.70, 1.6, 0.024)
        self.living_room_table_offset = (0, 0, 0.41)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.living_room_table_full_size[2]
        kwargs.update(
            {"robots": [f"OnTheGround{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.living_room_table_offset})
        kwargs.update({"arena_type": "living_room"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_living_room_tabletop_base_style.xml"}
            )
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "wood-plank",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "living_room_table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "living_room_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    z_offset=self.workspace_offset[2],
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.living_room_table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )
# custom
    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6065773716836134, 0.0, 0.96],
            quat=[
                0.6182166934013367,
                0.3432307541370392,
                0.3432314395904541,
                0.6182177066802979,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.5, 0.0, 0.9], quat=[0.56, 0.43, 0.43, 0.56]
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
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.735, 1.875],
            quat=[0.513, 0.353, 0.443, 0.645],
        )

# origin
    # def _setup_camera(self, mujoco_arena):
    #     mujoco_arena.set_camera(
    #         camera_name="agentview",
    #         pos=[1.1586131746834771, 0.0, 1.7103500240372423],
    #         quat=[
    #             0.6380177736282349,
    #             0.3048497438430786,
    #             0.30484986305236816,
    #             0.6380177736282349,
    #         ],
    #     )

    #     # For visualization purpose
    #     mujoco_arena.set_camera(
    #         camera_name="frontview", pos=[1.5, 0.0, 0.9], quat=[0.56, 0.43, 0.43, 0.56]
    #     )
    #     mujoco_arena.set_camera(
    #         camera_name="galleryview",
    #         pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
    #         quat=[
    #             0.42261379957199097,
    #             0.23374411463737488,
    #             0.41646939516067505,
    #             0.7702690958976746,
    #         ],
    #     )
    #     mujoco_arena.set_camera(
    #         camera_name="paperview",
    #         pos=[2.1, 0.735, 1.875],
    #         quat=[0.513, 0.353, 0.443, 0.645],
    #     )
