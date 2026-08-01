"""MuJoCo helpers for visualizing the position targets used by GMR."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ReferencePoint:
    reference_name: str
    world_position: np.ndarray
    world_rotation: np.ndarray


@dataclass(frozen=True)
class PositionMapping:
    stage: int
    reference_name: str
    robot_name: str
    frame_type: str
    weight: float


@dataclass(frozen=True)
class Correspondence:
    stage: int
    reference_name: str
    robot_name: str
    weight: float
    reference_world_position: np.ndarray
    robot_world_position: np.ndarray
    error_vector: np.ndarray
    error_norm: float


@dataclass(frozen=True)
class ErrorStatistics:
    frame_index: int
    mean: float
    rms: float
    maximum: float
    per_point: Mapping[str, float]


_SMPLX_DEBUG_PARENTS = {
    "spine3": "pelvis",
    "left_knee": "pelvis",
    "right_knee": "pelvis",
    "left_foot": "left_knee",
    "right_foot": "right_knee",
    "left_elbow": "spine3",
    "right_elbow": "spine3",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_shoulder": "spine3",
    "right_shoulder": "spine3",
}


def load_effective_ik_config(
    path: str | Path, actual_human_height: float | None = None
) -> dict:
    """Load a bare or calibration-wrapped config as GMR sees it."""
    with Path(path).open() as stream:
        loaded = json.load(stream)
    config = copy.deepcopy(loaded.get("config", loaded))
    if actual_human_height is not None:
        ratio = actual_human_height / config["human_height_assumption"]
        config["human_scale_table"] = {
            name: float(scale) * ratio
            for name, scale in config["human_scale_table"].items()
        }
    return config


def _frame_type(model: mujoco.MjModel, name: str) -> str:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if body_id >= 0 and site_id >= 0:
        raise ValueError(f"ambiguous MuJoCo frame {name!r}")
    if body_id >= 0:
        return "body"
    if site_id >= 0:
        return "site"
    raise ValueError(f"unknown MuJoCo body or site frame: {name!r}")


def build_position_correspondences(
    model: mujoco.MjModel, config: Mapping
) -> tuple[PositionMapping, ...]:
    mappings = []
    for stage in (1, 2):
        if not config.get(f"use_ik_match_table{stage}", True):
            continue
        for robot_name, entry in config[f"ik_match_table{stage}"].items():
            reference_name, position_weight = entry[:2]
            if float(position_weight) == 0.0:
                continue
            mappings.append(
                PositionMapping(
                    stage=stage,
                    reference_name=reference_name,
                    robot_name=robot_name,
                    frame_type=_frame_type(model, robot_name),
                    weight=float(position_weight),
                )
            )
    return tuple(mappings)


def transform_reference_frame(
    frame: Mapping[str, Sequence[np.ndarray]], config: Mapping
) -> dict[str, ReferencePoint]:
    """Reproduce the target preprocessing order in ``update_targets``."""
    root_name = config["human_root_name"]
    root_position = np.asarray(frame[root_name][0], dtype=float)
    scales = config["human_scale_table"]
    scaled_root = root_position * float(scales[root_name])
    scaled = {}
    for name, scale in scales.items():
        position, quaternion = frame[name]
        position = np.asarray(position, dtype=float)
        if name == root_name:
            new_position = scaled_root.copy()
        else:
            new_position = (
                position - root_position
            ) * float(scale) + scaled_root
        scaled[name] = [new_position, np.asarray(quaternion, dtype=float)]

    # The current GMR implementation applies table-1 offsets to the shared
    # target dictionary. Table-2 tasks consume those same transformed targets.
    ground = float(config.get("ground_height", 0.0)) * np.array([0, 0, 1.0])
    table = config["ik_match_table1"]
    by_reference = {entry[0]: entry for entry in table.values()}
    global_offsets = config.get("global_position_offsets", {})
    result = {}
    for name, (position, quaternion) in scaled.items():
        entry = by_reference.get(name)
        if entry is None:
            continue
        local_offset = np.asarray(entry[3], dtype=float) - ground
        rotation_offset = Rotation.from_quat(entry[4], scalar_first=True)
        updated_rotation = (
            Rotation.from_quat(quaternion, scalar_first=True) * rotation_offset
        )
        world_position = position + updated_rotation.apply(local_offset)
        world_position += np.asarray(global_offsets.get(name, 0.0), dtype=float)
        result[name] = ReferencePoint(
            reference_name=name,
            world_position=world_position,
            world_rotation=updated_rotation.as_matrix(),
        )
    return result


def transform_full_reference_frame(
    frame: Mapping[str, Sequence[np.ndarray]],
    config: Mapping,
    *,
    hierarchy_edges: Sequence[tuple[str, str]] = (),
    exact_targets: Mapping[str, ReferencePoint] | None = None,
) -> dict[str, ReferencePoint]:
    """Place the complete source skeleton for contextual visualization.

    Unmapped joints inherit the nearest mapped descendant scale, or the
    nearest mapped ancestor scale when their subtree has no mapped target.
    Exact GMR targets overwrite mapped joints after the hierarchy is scaled.
    """
    root_name = config["human_root_name"]
    scales = config["human_scale_table"]
    root_position = np.asarray(frame[root_name][0], dtype=float)
    scaled_root = root_position * float(scales[root_name])
    parents = {child: parent for parent, child in hierarchy_edges}
    children = {}
    for parent, child in hierarchy_edges:
        children.setdefault(parent, []).append(child)

    def inherited_scale(name: str) -> float:
        if name in scales:
            return float(scales[name])
        frontier = list(children.get(name, ()))
        while frontier:
            next_frontier = []
            for descendant in frontier:
                if descendant in scales:
                    return float(scales[descendant])
                next_frontier.extend(children.get(descendant, ()))
            frontier = next_frontier
        ancestor = parents.get(name)
        while ancestor is not None:
            if ancestor in scales:
                return float(scales[ancestor])
            ancestor = parents.get(ancestor)
        return float(scales[root_name])

    result = {
        name: ReferencePoint(
            reference_name=name,
            world_position=(np.asarray(value[0]) - root_position)
            * inherited_scale(name)
            + scaled_root,
            world_rotation=Rotation.from_quat(
                np.asarray(value[1]), scalar_first=True
            ).as_matrix(),
        )
        for name, value in frame.items()
    }
    for name, point in (exact_targets or {}).items():
        if name in result:
            result[name] = point
    return result


def _robot_frame_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: PositionMapping,
) -> np.ndarray:
    if mapping.frame_type == "site":
        return data.site(mapping.robot_name).xpos.copy()
    return data.body(mapping.robot_name).xpos.copy()


def compute_correspondences(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_points: Mapping[str, ReferencePoint],
    mappings: Sequence[PositionMapping],
) -> tuple[Correspondence, ...]:
    rows = []
    for mapping in mappings:
        if mapping.reference_name not in reference_points:
            continue
        reference = reference_points[mapping.reference_name].world_position
        robot = _robot_frame_position(model, data, mapping)
        error = robot - reference
        rows.append(
            Correspondence(
                stage=mapping.stage,
                reference_name=mapping.reference_name,
                robot_name=mapping.robot_name,
                weight=mapping.weight,
                reference_world_position=reference.copy(),
                robot_world_position=robot,
                error_vector=error,
                error_norm=float(np.linalg.norm(error)),
            )
        )
    return tuple(rows)


def reference_edges(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return the sparse SMPL-X hierarchy restricted to visible targets."""
    available = set(names)
    return tuple(
        (parent, child)
        for child, parent in _SMPLX_DEBUG_PARENTS.items()
        if child in available and parent in available
    )


def _mapping_body_id(model: mujoco.MjModel, mapping: PositionMapping) -> int:
    if mapping.frame_type == "body":
        return model.body(mapping.robot_name).id
    return int(model.site(mapping.robot_name).bodyid[0])


def robot_edges(
    model: mujoco.MjModel, mappings: Sequence[PositionMapping]
) -> tuple[tuple[str, str], ...]:
    """Connect mapped frames through their nearest mapped body ancestor."""
    unique = {mapping.robot_name: mapping for mapping in mappings}
    body_to_names = {}
    for name, mapping in unique.items():
        body_to_names.setdefault(_mapping_body_id(model, mapping), []).append(name)
    edges = []
    for child_name, mapping in unique.items():
        body_id = _mapping_body_id(model, mapping)
        same_body = sorted(
            name for name in body_to_names.get(body_id, []) if name != child_name
        )
        if mapping.frame_type == "site" and same_body:
            edges.append((same_body[0], child_name))
            continue
        parent_id = int(model.body_parentid[body_id])
        distance = 1
        while parent_id > 0 and parent_id not in body_to_names:
            parent_id = int(model.body_parentid[parent_id])
            distance += 1
        # A long jump is not a skeleton edge: intermediate joints must be
        # represented explicitly rather than drawing a misleading chord.
        if parent_id in body_to_names and distance <= 2:
            parent_name = sorted(body_to_names[parent_id])[0]
            if parent_name != child_name:
                edges.append((parent_name, child_name))
    return tuple(dict.fromkeys(edges))


def compose_scene_xml(mjcf_path: str | Path) -> str:
    """Compose a lit floor scene while keeping the supplied robot dynamic."""
    path = Path(mjcf_path).resolve()
    return f"""<mujoco model="gmr_debug_scene">
  <include file="{path.as_posix()}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.78 0.82 0.88 1"/>
  </visual>
  <asset>
    <texture name="gmr_debug_skybox" type="skybox" builtin="gradient"
             rgb1="0.82 0.86 0.92" rgb2="0.55 0.63 0.74"
             width="512" height="3072"/>
  </asset>
  <worldbody>
    <light name="gmr_debug_key" pos="0 -3 4" dir="0 1 -1" directional="true"/>
    <light name="gmr_debug_fill" pos="2 2 3" dir="-1 -1 -1" directional="true" diffuse="0.4 0.4 0.4"/>
    <geom name="gmr_debug_floor" type="plane" size="20 20 0.1" rgba="0.8 0.82 0.85 1"/>
  </worldbody>
</mujoco>"""


def compute_height_offset(
    model: mujoco.MjModel,
    motion: Mapping,
    *,
    foot_frames: Sequence[str] | None = None,
    percentile: float = 2.0,
) -> float:
    """Return the Z translation that grounds the motion's foot percentile."""
    if foot_frames is None:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index)
            for index in range(model.nsite)
        ]
        foot_frames = tuple(
            name for name in names if name and any(k in name.lower() for k in ("foot", "sole"))
        )
    if not foot_frames:
        raise ValueError("no foot/sole sites available for automatic height correction")
    frame_specs = [(name, _frame_type(model, name)) for name in foot_frames]
    data = mujoco.MjData(model)
    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"])
    heights = []
    for index in range(len(root_pos)):
        data.qpos[:3] = root_pos[index]
        data.qpos[3:7] = root_rot[index][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[index]
        mujoco.mj_forward(model, data)
        for name, frame_type in frame_specs:
            position = data.site(name).xpos if frame_type == "site" else data.body(name).xpos
            heights.append(float(position[2]))
    return -float(np.percentile(heights, percentile))


class GMRDebugVisualizer:
    """Own robot transparency and MuJoCo user-scene debug geometry."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Mapping,
        *,
        robot_alpha: float = 0.3,
        point_radius: float = 0.018,
        line_width: float = 0.006,
        error_threshold: float = 0.01,
        display_name: str | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.mappings = build_position_correspondences(model, config)
        self.point_radius = float(point_radius)
        self.line_width = float(line_width)
        self.error_threshold = float(error_threshold)
        self.display_name = display_name
        self.reference_edges = reference_edges(
            tuple(mapping.reference_name for mapping in self.mappings)
        )
        self.robot_edges = robot_edges(model, self.mappings)
        self._original_rgba = model.geom_rgba.copy()
        robot_geoms = model.geom_bodyid != 0
        model.geom_rgba[robot_geoms, 3] = robot_alpha

    def restore_robot_alpha(self) -> None:
        self.model.geom_rgba[:] = self._original_rgba

    @staticmethod
    def statistics(
        rows: Sequence[Correspondence], frame_index: int
    ) -> ErrorStatistics:
        values = np.asarray([row.error_norm for row in rows], dtype=float)
        if len(values) == 0:
            return ErrorStatistics(frame_index, 0.0, 0.0, 0.0, {})
        return ErrorStatistics(
            frame_index=frame_index,
            mean=float(np.mean(values)),
            rms=float(np.sqrt(np.mean(values**2))),
            maximum=float(np.max(values)),
            per_point={
                f"stage{row.stage}:{row.reference_name}->{row.robot_name}": row.error_norm
                for row in rows
            },
        )

    def _add_sphere(self, scene, position, rgba) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([self.point_radius] * 3),
            np.asarray(position, dtype=float),
            np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def _add_connector(self, scene, start, end, rgba, width=None) -> None:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-9:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            self.line_width if width is None else float(width),
            start,
            end,
        )
        scene.ngeom += 1

    def update(
        self,
        viewer,
        data: mujoco.MjData,
        reference_skeleton: Mapping[str, ReferencePoint],
        frame_index: int,
        *,
        show_error_lines: bool = True,
        full_reference_skeleton: Mapping[str, ReferencePoint] | None = None,
        full_reference_edges: Sequence[tuple[str, str]] = (),
    ) -> ErrorStatistics:
        scene = viewer.user_scn
        scene.ngeom = 0
        rows = compute_correspondences(
            self.model, data, reference_skeleton, self.mappings
        )
        unique_reference = {}
        unique_robot = {}
        for row in rows:
            unique_reference[row.reference_name] = row.reference_world_position
            unique_robot[row.robot_name] = row.robot_world_position
        full_reference_skeleton = full_reference_skeleton or {}
        for position in (
            point.world_position for point in full_reference_skeleton.values()
        ):
            self._add_sphere(scene, position, [0.1, 0.35, 1.0, 0.42])
        for parent, child in full_reference_edges:
            if parent in full_reference_skeleton and child in full_reference_skeleton:
                self._add_connector(
                    scene,
                    full_reference_skeleton[parent].world_position,
                    full_reference_skeleton[child].world_position,
                    [0.1, 0.35, 1.0, 0.5],
                )
        for position in unique_reference.values():
            self._add_sphere(scene, position, [0.1, 0.35, 1.0, 0.8])
        for position in unique_robot.values():
            self._add_sphere(scene, position, [1.0, 0.55, 0.05, 0.9])
        if not full_reference_skeleton:
            for parent, child in self.reference_edges:
                if parent in unique_reference and child in unique_reference:
                    self._add_connector(
                        scene,
                        unique_reference[parent],
                        unique_reference[child],
                        [0.1, 0.35, 1.0, 0.55],
                    )
        for parent, child in self.robot_edges:
            if parent in unique_robot and child in unique_robot:
                self._add_connector(
                    scene,
                    unique_robot[parent],
                    unique_robot[child],
                    [1.0, 0.55, 0.05, 0.65],
                )
        if show_error_lines:
            for row in rows:
                if row.error_norm < self.error_threshold:
                    continue
                alpha = min(1.0, 0.35 + row.error_norm * 4.0)
                self._add_connector(
                    scene,
                    row.reference_world_position,
                    row.robot_world_position,
                    [1.0, 0.05, 0.05, alpha],
                    width=self.line_width,
                )
        statistics = self.statistics(rows, frame_index)
        if hasattr(viewer, "set_texts"):
            ordered = sorted(
                statistics.per_point.items(), key=lambda item: item[1], reverse=True
            )
            labels = "frame\nmean\nrms\nmax"
            values = (
                f"{frame_index}\n{statistics.mean:.4f} m\n"
                f"{statistics.rms:.4f} m\n{statistics.maximum:.4f} m"
            )
            if self.display_name:
                labels = f"motion\n{labels}"
                values = f"{self.display_name}\n{values}"
            detail_labels = "\n".join(name for name, _ in ordered[:8])
            detail_values = "\n".join(f"{value:.4f}" for _, value in ordered[:8])
            viewer.set_texts(
                [
                    (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values),
                    (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, detail_labels, detail_values),
                ]
            )
        return statistics
