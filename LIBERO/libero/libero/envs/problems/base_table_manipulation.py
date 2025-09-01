from robosuite.utils.mjcf_utils import new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


class BaseTableManipulation(BDDLBaseDomain):
    """
    Base table manipulation class containing common functionality for all manipulation environments
    """
    
    # Common configuration constants
    DEFAULT_POUR_DISTANCE_THRESHOLD = 0.3
    DEFAULT_POUR_QUAT_THRESHOLD = 0.2
    DEFAULT_PICK_LOG_LENGTH = 40
    DEFAULT_PICK_QUAT_THRESHOLD = 0.1
    
    # Common check_inside regions list
    COMMON_CHECK_INSIDE_REGIONS = [
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
    ]

    def __init__(self, bddl_file_name, *args, **kwargs):
        # Initialize common attributes
        self._init_common_attributes()
        
        # Configure workspace and get updated kwargs
        workspace_kwargs = self._configure_workspace(**kwargs)
        # Call parent class __init__ with updated kwargs
        super().__init__(bddl_file_name, *args, **workspace_kwargs)
        
        # Post initialization setup
        self._post_init_setup()

    def _init_common_attributes(self):
        """Initialize common attributes for all manipulation classes"""
        self.visualization_sites_list = []
        self._task_progress = {}
        self._pour_logs = {}
        self.pour_distance_threshold = self.DEFAULT_POUR_DISTANCE_THRESHOLD
        self.pour_quat_threshold = self.DEFAULT_POUR_QUAT_THRESHOLD
        self.pick_log_length = self.DEFAULT_PICK_LOG_LENGTH
        self.pick_quat_threshold = self.DEFAULT_PICK_QUAT_THRESHOLD
        self._pick_logs = {}
        self.skip_pick_quat_once = False

    def _configure_workspace(self, **kwargs):
        """To be implemented by subclasses, configure workspace-specific parameters"""
        raise NotImplementedError("Subclasses must implement _configure_workspace method")

    def _post_init_setup(self):
        """Common setup after initialization"""
        self.regions = list(self.parsed_problem['regions'].keys())
        self.object_names = list(self.objects_dict.keys())
        self._has_left_table = {obj: False for obj in self.object_names}
        self._last_regions = {obj: None for obj in self.object_names}
        self._location_log = {obj: [] for obj in self.object_names}
        
        # Get specific table region name and add to check_inside
        table_region = self._get_table_region_name()
        self.check_inside = self.COMMON_CHECK_INSIDE_REGIONS + [table_region]
        self.regions = [x for x in self.check_inside if x in self.regions]

    def _get_table_region_name(self):
        """To be implemented by subclasses, return corresponding table region name"""
        raise NotImplementedError("Subclasses must implement _get_table_region_name method")

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Common fixture loading logic"""
        excluded_category = self._get_excluded_fixture_category()
        
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == excluded_category:
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _get_excluded_fixture_category(self):
        """To be implemented by subclasses, return fixture category to exclude"""
        raise NotImplementedError("Subclasses must implement _get_excluded_fixture_category method")

    def _load_objects_in_arena(self, mujoco_arena):
        """Common object loading logic"""
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        """Common site loading logic"""
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        
        for object_region_name in list(region_dict.keys()):
            if self._is_table_region(object_region_name):
                # Process table region
                self._process_table_region(object_region_name, region_dict, object_sites_dict, mujoco_arena)
            else:
                # Process other regions
                self._process_other_region(object_region_name, region_dict, object_sites_dict)
        
        self.object_sites_dict = object_sites_dict
        self._setup_visualization_objects()

    def _is_table_region(self, region_name):
        """Check if it's a table region"""
        raise NotImplementedError("Subclasses must implement _is_table_region method")

    def _process_table_region(self, region_name, region_dict, object_sites_dict, mujoco_arena):
        """Common logic for processing table regions"""
        ranges = region_dict[region_name]["ranges"][0]
        assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
        zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
        zone_centroid_xy = self._get_zone_centroid(ranges)
        
        z_offset = self._get_table_z_offset(region_name)
        if z_offset is not None:
            target_zone = TargetZone(
                z_offset=z_offset,
                name=region_name,
                rgba=region_dict[region_name]["rgba"],
                zone_size=zone_size,
                zone_centroid_xy=zone_centroid_xy,
            )
        else:
            target_zone = TargetZone(
                name=region_name,
                rgba=region_dict[region_name]["rgba"],
                zone_size=zone_size,
                zone_centroid_xy=zone_centroid_xy,
            )
        
        object_sites_dict[region_name] = target_zone
        self._append_table_site(mujoco_arena, target_zone)

    def _get_zone_centroid(self, ranges):
        """To be implemented by subclasses, calculate zone centroid"""
        raise NotImplementedError("Subclasses must implement _get_zone_centroid method")

    def _get_table_z_offset(self, region_name):
        """To be implemented by subclasses, get table z offset"""
        raise NotImplementedError("Subclasses must implement _get_table_z_offset method")

    def _append_table_site(self, mujoco_arena, target_zone):
        """To be implemented by subclasses, append site to arena"""
        raise NotImplementedError("Subclasses must implement _append_table_site method")

    def _process_other_region(self, region_name, region_dict, object_sites_dict):
        """Common logic for processing non-table regions"""
        for query_dict in [self.objects_dict, self.fixtures_dict]:
            for (name, body) in query_dict.items():
                try:
                    if "worldbody" not in list(body.__dict__.keys()):
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
                        if site_name == region_name:
                            object_sites_dict[region_name] = SiteObject(
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

    def _setup_visualization_objects(self):
        """Setup visualization objects"""
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _check_success(self, monitor_dict):
        """Common success checking logic"""
        # Initialize internal cache
        if not hasattr(self, "_state_progress"):
            self._state_progress = {obj: 0 for obj in monitor_dict}
        if not hasattr(self, "_last_regions"):
            self._last_regions = {obj: None for obj in self.object_names}
        if not hasattr(self, "_location_log"):
            self._location_log = {obj: [] for obj in self.object_names}

        # Calculate current regions for each object
        object_locations = {obj: [] for obj in self.object_names}
        for region in self.regions:
            for obj in self.object_names:
                if self._eval_predicate(['in', obj, region]):
                    object_locations[obj].append(region)
        for obj, regs in object_locations.items():
            if not regs:
                object_locations[obj] = ['None']

        # Record location log
        for obj, curr_regs in object_locations.items():
            last_regs = self._last_regions[obj]
            if last_regs is None or set(curr_regs) != set(last_regs):
                self._location_log[obj].append({
                    'time': self.sim.data.time,
                    'regions': curr_regs.copy()
                })
                self._last_regions[obj] = curr_regs.copy()

        # Sequentially evaluate target states in monitor_dict
        completion = {}
        for obj, state_list in monitor_dict.items():
            curr_idx = self._state_progress[obj]
            total = len(state_list)

            if curr_idx >= total:
                completion[obj] = 100.0
                continue

            state = state_list[curr_idx]
            table_region_name = self._get_table_region_name()

            # Determine success or failure
            if len(state) == 3:
                pred, _, region = state
                if region == table_region_name:
                    in_table = self._eval_predicate(['in', obj, region])
                    if not self._has_left_table[obj]:
                        if not in_table:
                            self._has_left_table[obj] = True
                        success = False
                    else:
                        success = in_table
                else:
                    success = self._eval_predicate(state)
            elif len(state) == 2:
                success = self._eval_predicate(state)
            else:
                raise ValueError(f"Unsupported state format (length should be 2 or 3): {state}")

            if success:
                pred = state[0]
                if len(state) == 3:
                    _, _, region = state
                    desc = f"{pred} {obj} → {region}"
                else:
                    desc = f"{pred} {obj}"

                self._state_progress[obj] += 1
                print(f"[Monitor] {obj}: Completed subtask `{desc}` "
                      f"({self._state_progress[obj]}/{total}) ✅")

            completion[obj] = self._state_progress[obj] / total * 100.0

        # Count completed subtasks and check if all are done
        total_completed = sum(self._state_progress[obj] for obj in monitor_dict)
        total_subtasks = sum(len(state_list) for state_list in monitor_dict.values())
        all_done = (total_completed == total_subtasks)
        
        if getattr(self, "skip_pick_quat_once", False):
            self.skip_pick_quat_once = False

        return completion, total_completed, all_done

    def _eval_predicate(self, state):
        """Common predicate evaluation logic"""
        # Multi-region objects and their region lists
        MULTI_REGION_OBJECT_REGIONS = {
            'microwave_1': ['top_side', 'heating_region'],
            'short_cabinet_1': ['middle_region', 'top_region', 'bottom_region'],
            'short_fridge_1': ['upper_region', 'middle_region', 'lower_region'],
            'wooden_cabinet_1': ['top_side', 'top_region', 'middle_region', 'bottom_region'],
            'white_cabinet_1': ['top_side', 'top_region', 'middle_region', 'bottom_region'],
            'white_storage_box_1': ['top_side', 'bottom_side', 'right_side', 'left_side'],
            'wooden_shelf_1': ['top_side', 'top_region', 'middle_region', 'bottom_region'],
            'wooden_two_layer_shelf_1': ['top_side', 'top_region', 'bottom_region'],
            'bowl_drainer_1': ['left_region', 'right_region']
        }

        # Special logic for Pick
        if len(state) == 2 and state[0] == 'pick':
            return self._eval_pick_predicate(state[1])

        # Special logic for Pour
        if len(state) == 3 and state[0] == 'pour':
            return self._eval_pour_predicate(state[1], state[2])

        # Common processing logic for multi-region objects
        if len(state) == 3:
            predicate_fn_name, object_1_name, object_2_name = state
            obj2 = self.object_states_dict[object_2_name]

            if object_2_name in MULTI_REGION_OBJECT_REGIONS:
                for region_name in MULTI_REGION_OBJECT_REGIONS[object_2_name]:
                    if hasattr(obj2, region_name):
                        region_obj = getattr(obj2, region_name)
                        if eval_predicate_fn(predicate_fn_name,
                                             self.object_states_dict[object_1_name],
                                             region_obj):
                            return True
                return False
            else:
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

    def _eval_pick_predicate(self, object_name):
        """Evaluate Pick predicate"""
        # Update quaternion log
        geom = self.object_states_dict[object_name].get_geom_state()
        curr_quat = geom['quat'].copy()
        logs = self._pick_logs.setdefault(object_name, [])
        logs.append(curr_quat)
        if len(logs) > self.pick_log_length:
            logs.pop(0)

        # Original region evaluation logic
        curr_regions = [
            r for r in self.regions 
            if self._eval_predicate(['in', object_name, r])
        ]
        if not curr_regions:
            curr_regions = ['None']

        if object_name not in getattr(self, '_pick_last_regions', {}):
            if not hasattr(self, '_pick_last_regions'):
                self._pick_last_regions = {}
            self._pick_last_regions[object_name] = curr_regions
            return False

        prev_regions = self._pick_last_regions[object_name]
        region_success = (prev_regions != ['None'] and curr_regions == ['None'])
        self._pick_last_regions[object_name] = curr_regions

        if not region_success:
            return False

        if getattr(self, "skip_pick_quat_once", False):
            return region_success
        elif len(logs) >= 2:
            import numpy as np
            q_start, q_end = logs[0], logs[-1]
            quat_diff = np.linalg.norm(q_end - q_start)
            if quat_diff > self.pick_quat_threshold:
                print(f"[Monitor] {object_name} pick suppressed: quaternion change {quat_diff:.3f} "
                      f"exceeds threshold {self.pick_quat_threshold}")
                return False

        print(f"[Monitor] {object_name} region changed from {prev_regions} to {curr_regions}, Pick successful")
        return True

    def _eval_pour_predicate(self, obj1_name, obj2_name):
        """Evaluate Pour predicate"""
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
        return False

    def _reset_success_buffers(self):
        """Clear cache related to _check_success"""
        for attr in ("_state_progress", "_last_regions", "_location_log"):
            if hasattr(self, attr):
                delattr(self, attr)

        self._has_left_table = {obj: False for obj in self.object_names}
        self._pick_last_regions = {}
        self._pour_logs = {}
        self._pick_logs = {}
        self.skip_pick_quat_once = False

    def reset(self, *args, **kwargs):
        """Reset environment"""
        obs = super().reset(*args, **kwargs)
        self._reset_success_buffers()
        return obs

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()
        self.set_visualization()

    def set_visualization(self):
        """Common visualization setup"""
        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    ) 