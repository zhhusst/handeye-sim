#!/usr/bin/env python3
"""Interactive six-pose star-acquisition visualization from real seed data.

The plotted transforms come directly from one calibration run:

* flange pose: ``seeds.json`` -> ``R_BF``, ``t_BF``;
* hand-eye: ``calibration_result.json`` -> ``handeye`` = ``^F T_S``;
* sensor pose: ``^B T_S = ^B T_F ^F T_S``.

No pose is synthesized.  The sensor body is loaded from the Gocator visual
mesh and visual-origin transform declared by the active robot URDF.  Only the
translucent board-patch extent is schematic; its pose is data-derived.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_RUN = Path(
    "/workspace/data/calibration_runs/"
    "20260820_144753_焊接书_位置1_真机_1"
)
DEFAULT_OUTPUT = Path(
    "/workspace/star_seed_pose_figure/star_seed_poses_current_view.png"
)
DEFAULT_URDF = Path("/workspace/urdf/calib_robot.urdf")

DISPLAY_LABELS = {
    "reference": "Reference",
    "ry_positive": r"$R_{y_F}(+)$",
    "ry_negative": r"$R_{y_F}(-)$",
    "rx_positive": r"$R_{x_F}(+)$",
    "rx_negative": r"$R_{x_F}(-)$",
    "rx_ry_positive": r"$R_{x_F,y_F}(+,+)$",
}


@dataclass(frozen=True)
class PoseRecord:
    index: int
    label: str
    display_label: str
    R_BF: np.ndarray
    t_BF: np.ndarray
    R_BS: np.ndarray
    t_BS: np.ndarray
    joints_rad: np.ndarray
    endpoint_u_S: np.ndarray
    endpoint_v_S: np.ndarray


@dataclass(frozen=True)
class SceneData:
    run_dir: Path
    poses: tuple[PoseRecord, ...]
    R_FS: np.ndarray
    t_FS: np.ndarray
    board_corner_B: np.ndarray | None
    board_rotation_B: np.ndarray | None
    board_extent_uv: tuple[float, float] | None


@dataclass(frozen=True)
class TriangleMesh:
    """Simplified URDF visual mesh expressed in its link frame."""

    vertices_S: np.ndarray
    faces: np.ndarray
    normals_S: np.ndarray
    source_mesh: Path
    source_urdf: Path
    visual_origin_xyz_m: np.ndarray
    visual_origin_rpy_rad: np.ndarray
    R_FS_rviz: np.ndarray
    t_FS_rviz: np.ndarray
    fixed_joint_name: str
    link_name: str
    material_rgba: np.ndarray
    source_face_count: int


def _array(value, shape, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _rotation_vector_deg(rotation: np.ndarray) -> np.ndarray:
    """Return axis-angle vector in degrees without requiring SciPy."""
    angle = np.deg2rad(_rotation_angle_deg(rotation))
    if angle < 1.0e-10:
        return np.zeros(3)
    if abs(np.pi - angle) < 1.0e-6:
        values = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        axis = values / max(np.linalg.norm(values), 1.0e-12)
    else:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * np.sin(angle))
    return np.rad2deg(angle) * axis


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    R_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_y @ R_x


def _parse_vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    result = np.fromstring(text, sep=" ", dtype=float)
    if result.shape != (len(default),):
        raise ValueError(f"expected {len(default)} values, got {text!r}")
    return result


def load_scene(run_dir: Path) -> SceneData:
    seeds_path = run_dir / "seeds.json"
    result_path = run_dir / "calibration_result.json"
    if not seeds_path.is_file():
        raise FileNotFoundError(seeds_path)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    seeds_payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    seeds = seeds_payload.get("seeds", [])
    if len(seeds) != 6:
        raise ValueError(f"expected six physical seeds, found {len(seeds)}")

    handeye = result_payload["handeye"]
    R_FS = _array(handeye["rotation"], (3, 3), "handeye.rotation")
    t_FS = _array(handeye["translation"], (3,), "handeye.translation")

    poses: list[PoseRecord] = []
    for index, seed in enumerate(seeds):
        label = str(seed["label"])
        R_BF = _array(seed["R_BF"], (3, 3), f"seeds[{index}].R_BF")
        t_BF = _array(seed["t_BF"], (3,), f"seeds[{index}].t_BF")
        R_BS = R_BF @ R_FS
        t_BS = t_BF + R_BF @ t_FS
        poses.append(
            PoseRecord(
                index=index,
                label=label,
                display_label=DISPLAY_LABELS.get(label, label),
                R_BF=R_BF,
                t_BF=t_BF,
                R_BS=R_BS,
                t_BS=t_BS,
                joints_rad=_array(seed["joints"], (6,), f"seeds[{index}].joints"),
                endpoint_u_S=_array(
                    seed["endpoint_u_S"], (3,), f"seeds[{index}].endpoint_u_S"
                ),
                endpoint_v_S=_array(
                    seed["endpoint_v_S"], (3,), f"seeds[{index}].endpoint_v_S"
                ),
            )
        )

    board = result_payload.get("board", {})
    board_corner = None
    board_rotation = None
    board_extent = None
    if "corner" in board and "rotation" in board:
        board_corner = _array(board["corner"], (3,), "board.corner")
        board_rotation = _array(board["rotation"], (3, 3), "board.rotation")
        local_endpoints = []
        for pose in poses:
            for endpoint in (pose.endpoint_u_S, pose.endpoint_v_S):
                point_B = pose.R_BF @ (R_FS @ endpoint + t_FS) + pose.t_BF
                local_endpoints.append(board_rotation.T @ (point_B - board_corner))
        local_endpoints = np.asarray(local_endpoints)
        # Show only the observed corner region, not an invented full board outline.
        extent_u = max(0.06, 1.30 * float(np.max(local_endpoints[:, 0])))
        extent_v = max(0.09, 1.30 * float(np.max(local_endpoints[:, 1])))
        board_extent = (extent_u, extent_v)

    return SceneData(
        run_dir=run_dir,
        poses=tuple(poses),
        R_FS=R_FS,
        t_FS=t_FS,
        board_corner_B=board_corner,
        board_rotation_B=board_rotation,
        board_extent_uv=board_extent,
    )


def _mesh_path_from_uri(uri: str, urdf_path: Path) -> Path:
    if uri.startswith("file://"):
        return Path(uri[len("file://") :]).resolve()
    if uri.startswith("package://"):
        raise ValueError(
            "package:// mesh URI cannot be resolved without a ROS package index: " + uri
        )
    path = Path(uri)
    return (path if path.is_absolute() else urdf_path.parent / path).resolve()


def _load_collada_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the single geometry used by the project's Gocator COLLADA file."""
    root = ET.parse(mesh_path).getroot()
    namespace = root.tag.partition("}")[0].lstrip("{")
    ns = {"c": namespace} if namespace else {}
    prefix = "c:" if namespace else ""

    mesh = root.find(f".//{prefix}geometry/{prefix}mesh", ns)
    if mesh is None:
        raise ValueError(f"no COLLADA geometry found in {mesh_path}")

    vertices_element = mesh.find(f"{prefix}vertices", ns)
    triangles_element = mesh.find(f"{prefix}triangles", ns)
    if vertices_element is None or triangles_element is None:
        raise ValueError(f"the Gocator mesh must contain vertices and triangles: {mesh_path}")

    position_input = vertices_element.find(
        f"{prefix}input[@semantic='POSITION']", ns
    )
    if position_input is None:
        raise ValueError("COLLADA vertices do not reference POSITION data")
    source_id = position_input.attrib["source"].lstrip("#")
    position_source = mesh.find(f"{prefix}source[@id='{source_id}']", ns)
    if position_source is None:
        raise ValueError(f"missing COLLADA position source {source_id}")
    float_array = position_source.find(f"{prefix}float_array", ns)
    if float_array is None or not float_array.text:
        raise ValueError("empty COLLADA position array")
    vertices = np.fromstring(float_array.text, sep=" ", dtype=float).reshape(-1, 3)

    inputs = triangles_element.findall(f"{prefix}input", ns)
    stride = 1 + max(int(item.attrib.get("offset", "0")) for item in inputs)
    vertex_input = next(item for item in inputs if item.attrib.get("semantic") == "VERTEX")
    vertex_offset = int(vertex_input.attrib.get("offset", "0"))
    index_element = triangles_element.find(f"{prefix}p", ns)
    if index_element is None or not index_element.text:
        raise ValueError("empty COLLADA triangle index array")
    raw_indices = np.fromstring(index_element.text, sep=" ", dtype=np.int64)
    vertex_indices = raw_indices.reshape(-1, stride)[:, vertex_offset]
    faces = vertex_indices.reshape(-1, 3)
    return vertices, faces


def _load_binary_stl_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the binary STL layout used by the project's weld-gun model."""
    with mesh_path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    record = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    triangles = np.memmap(
        mesh_path,
        dtype=record,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    vertices = np.asarray(triangles["vertices"], dtype=float).reshape(-1, 3)
    faces = np.arange(3 * triangle_count, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _load_mesh_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = mesh_path.suffix.lower()
    if suffix == ".dae":
        return _load_collada_geometry(mesh_path)
    if suffix == ".stl":
        return _load_binary_stl_geometry(mesh_path)
    raise ValueError(f"unsupported visual mesh format: {mesh_path}")


def _simplify_mesh(
    vertices: np.ndarray, faces: np.ndarray, voxel_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic vertex clustering used only to keep Matplotlib responsive."""
    if voxel_size_m <= 0.0:
        return vertices, faces
    quantized = np.rint(vertices / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    clustered = np.zeros((len(counts), 3), dtype=float)
    for coordinate in range(3):
        clustered[:, coordinate] = np.bincount(
            inverse, weights=vertices[:, coordinate]
        ) / counts

    reduced_faces = inverse[faces]
    nondegenerate = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
        & (reduced_faces[:, 2] != reduced_faces[:, 0])
    )
    reduced_faces = reduced_faces[nondegenerate]
    # Remove duplicate triangles while retaining the winding of their first copy.
    canonical = np.sort(reduced_faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    return clustered, reduced_faces[np.sort(first)]


def load_urdf_visual_mesh(
    urdf_path: Path,
    *,
    parent_link: str,
    child_link: str,
    voxel_size_mm: float,
) -> TriangleMesh:
    """Load one fixed-link visual, including joint and visual origins, from URDF."""
    root = ET.parse(urdf_path).getroot()
    fixed_joint = None
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is not None
            and child is not None
            and parent.attrib.get("link") == parent_link
            and child.attrib.get("link") == child_link
        ):
            fixed_joint = joint
            break
    if fixed_joint is None:
        raise ValueError(
            f"{parent_link} -> {child_link} joint not found in {urdf_path}"
        )
    if fixed_joint.attrib.get("type") != "fixed":
        raise ValueError(f"{parent_link} -> {child_link} must be a fixed joint")
    joint_origin = fixed_joint.find("origin")
    joint_xyz = _parse_vector(
        joint_origin.attrib.get("xyz") if joint_origin is not None else None,
        (0, 0, 0),
    )
    joint_rpy = _parse_vector(
        joint_origin.attrib.get("rpy") if joint_origin is not None else None,
        (0, 0, 0),
    )

    link = root.find(f"./link[@name='{child_link}']")
    if link is None:
        raise ValueError(f"{child_link} link not found in {urdf_path}")
    visual = link.find("visual")
    if visual is None:
        raise ValueError(f"{child_link} has no visual element")
    origin = visual.find("origin")
    xyz = _parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0))
    rpy = _parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0))
    mesh_element = visual.find("geometry/mesh")
    if mesh_element is None:
        raise ValueError(f"{child_link} visual is not a mesh")
    mesh_path = _mesh_path_from_uri(mesh_element.attrib["filename"], urdf_path)
    scale = _parse_vector(mesh_element.attrib.get("scale"), (1, 1, 1))
    color_element = visual.find("material/color")
    material_rgba = _parse_vector(
        color_element.attrib.get("rgba") if color_element is not None else None,
        (0.4, 0.4, 0.4, 1.0),
    )

    vertices_V, faces = _load_mesh_geometry(mesh_path)
    source_face_count = int(len(faces))
    vertices_V = vertices_V * scale
    vertices_V, faces = _simplify_mesh(
        vertices_V, faces, 0.001 * float(voxel_size_mm)
    )
    R_SV = _rpy_matrix(rpy)
    vertices_S = xyz + (R_SV @ vertices_V.T).T
    triangle_edges_1 = vertices_S[faces[:, 1]] - vertices_S[faces[:, 0]]
    triangle_edges_2 = vertices_S[faces[:, 2]] - vertices_S[faces[:, 0]]
    normals_S = np.cross(triangle_edges_1, triangle_edges_2)
    normal_norms = np.linalg.norm(normals_S, axis=1)
    valid = normal_norms > 1.0e-12
    faces = faces[valid]
    normals_S = normals_S[valid] / normal_norms[valid, None]
    return TriangleMesh(
        vertices_S=vertices_S,
        faces=faces,
        normals_S=normals_S,
        source_mesh=mesh_path,
        source_urdf=urdf_path,
        visual_origin_xyz_m=xyz,
        visual_origin_rpy_rad=rpy,
        R_FS_rviz=_rpy_matrix(joint_rpy),
        t_FS_rviz=joint_xyz,
        fixed_joint_name=str(fixed_joint.attrib.get("name", "")),
        link_name=child_link,
        material_rgba=material_rgba,
        source_face_count=source_face_count,
    )


def load_gocator_mesh(urdf_path: Path, voxel_size_mm: float) -> TriangleMesh:
    return load_urdf_visual_mesh(
        urdf_path,
        parent_link="fanuc_flange",
        child_link="gocator_sensor",
        voxel_size_mm=voxel_size_mm,
    )


def load_weld_gun_mesh(urdf_path: Path, voxel_size_mm: float) -> TriangleMesh:
    return load_urdf_visual_mesh(
        urdf_path,
        parent_link="fanuc_flange",
        child_link="weld_gun",
        voxel_size_mm=voxel_size_mm,
    )


def draw_triad(ax, origin, rotation, length: float, linewidth: float = 2.5):
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for axis, color in enumerate(colors):
        direction = rotation[:, axis] * length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=linewidth,
            arrow_length_ratio=0.18,
            normalize=False,
        )


def _urdf_link_pose(pose: PoseRecord, mesh: TriangleMesh):
    R_BS = pose.R_BF @ mesh.R_FS_rviz
    t_BS = pose.t_BF + pose.R_BF @ mesh.t_FS_rviz
    return R_BS, t_BS


def _scene_points(
    scene: SceneData,
    gocator_mesh: TriangleMesh,
    weld_gun_mesh: TriangleMesh | None,
    show_board: bool,
    show_base_origin: bool,
):
    points = []
    for pose in scene.poses:
        points.append(pose.t_BF)
        for mesh in (gocator_mesh, weld_gun_mesh):
            if mesh is None:
                continue
            R_BL, t_BL = _urdf_link_pose(pose, mesh)
            vertices_B = t_BL + (R_BL @ mesh.vertices_S.T).T
            points.append(t_BL)
            points.extend(vertices_B)
    if show_board and scene.board_corner_B is not None and scene.board_rotation_B is not None:
        u_extent, v_extent = scene.board_extent_uv
        C = scene.board_corner_B
        u = scene.board_rotation_B[:, 0]
        v = scene.board_rotation_B[:, 1]
        points.extend((C, C + u_extent * u, C + v_extent * v, C + u_extent * u + v_extent * v))
    if show_base_origin:
        points.append(np.zeros(3))
    return np.asarray(points)


def set_equal_limits(ax, points: np.ndarray, margin_fraction: float = 0.10):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1.0e-6)
    radius = 0.5 * float(np.max(span)) * (1.0 + 2.0 * margin_fraction)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_urdf_visual(
    ax,
    pose: PoseRecord,
    mesh: TriangleMesh,
    alpha_scale=0.96,
    color_override=None,
):
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    R_BL, t_BL = _urdf_link_pose(pose, mesh)
    vertices = t_BL + (R_BL @ mesh.vertices_S.T).T
    triangles = vertices[mesh.faces]
    normals_B = (R_BL @ mesh.normals_S.T).T
    light_direction = np.array([0.35, -0.45, 0.82])
    light_direction /= np.linalg.norm(light_direction)
    diffuse = np.maximum(0.0, normals_B @ light_direction)
    intensity = 0.42 + 0.58 * diffuse
    base_rgb = (
        np.asarray(to_rgb(color_override), dtype=float)
        if color_override is not None
        else mesh.material_rgba[:3]
    )
    alpha = float(mesh.material_rgba[3]) * float(alpha_scale)
    facecolors = np.column_stack(
        (intensity[:, None] * base_rgb[None, :], np.full(len(intensity), alpha))
    )
    body = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.add_collection3d(body)
    return R_BL, t_BL


def render_scene(
    ax,
    scene: SceneData,
    gocator_mesh: TriangleMesh,
    weld_gun_mesh: TriangleMesh | None,
    *,
    axis_length: float,
    show_labels: bool,
    show_board: bool,
    show_base_origin: bool,
    show_transform_guides: bool,
    show_grid: bool,
    show_base_axes: bool,
    clean: bool,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    branch_colors = (
        "#111111",
        "#e69f00",
        "#56b4e9",
        "#009e73",
        "#cc79a7",
        "#d55e00",
    )

    reference = scene.poses[0]
    for pose, branch_color in zip(scene.poses[1:], branch_colors[1:]):
        # Every final target was reached from the same verified reference pose.
        ax.plot(
            [reference.t_BF[0], pose.t_BF[0]],
            [reference.t_BF[1], pose.t_BF[1]],
            [reference.t_BF[2], pose.t_BF[2]],
            linestyle="--",
            linewidth=1.15,
            color=branch_color,
            alpha=0.72,
            zorder=1,
        )

    for pose, branch_color in zip(scene.poses, branch_colors):
        # RViz displays the body through the URDF fixed joint, independently
        # of the calibrated measurement-frame transform stored in the run.
        R_BS_rviz, t_BS_rviz = _urdf_link_pose(pose, gocator_mesh)
        if show_transform_guides:
            # Optional diagnostic guide from flange origin to the URDF
            # gocator_sensor origin. It is not a physical component.
            ax.plot(
                [pose.t_BF[0], t_BS_rviz[0]],
                [pose.t_BF[1], t_BS_rviz[1]],
                [pose.t_BF[2], t_BS_rviz[2]],
                color="#707070",
                linewidth=0.9,
                alpha=0.50,
                zorder=1,
            )
        ax.scatter(
            *pose.t_BF,
            s=24 if pose.index else 38,
            color=branch_color,
            edgecolor="white",
            linewidth=0.5,
            depthshade=False,
            zorder=4,
        )
        draw_triad(ax, pose.t_BF, pose.R_BF, axis_length)

        # Use exactly the same branch color for the flange-pose marker and its
        # physical Gocator body. Transparency keeps overlapping poses legible.
        draw_urdf_visual(
            ax,
            pose,
            gocator_mesh,
            alpha_scale=0.48,
            color_override=branch_color,
        )
        if weld_gun_mesh is not None:
            # Keep the gun translucent so the flange triads and the nearby
            # Gocator body remain readable when six real poses overlap.
            draw_urdf_visual(ax, pose, weld_gun_mesh, alpha_scale=0.24)
        if show_transform_guides:
            # Optional +z_S cue. It is neither a laser ray nor a physical part.
            optical_end = t_BS_rviz + 0.045 * R_BS_rviz[:, 2]
            ax.plot(
                [t_BS_rviz[0], optical_end[0]],
                [t_BS_rviz[1], optical_end[1]],
                [t_BS_rviz[2], optical_end[2]],
                color="#202020",
                linewidth=1.2,
            )

        if show_labels:
            # Place branch labels away from the reference pose instead of using
            # one identical offset.  This keeps the five closely spaced final
            # poses legible in the default view while remaining data-driven.
            radial = pose.t_BF - reference.t_BF
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1.0e-9:
                offset = 0.020 * radial / radial_norm + np.array([0.0, 0.0, 0.010])
            else:
                offset = np.array([0.014, 0.010, 0.018])
            ax.text(
                *(pose.t_BF + offset),
                f"{pose.index}",
                fontsize=14.0,
                fontweight="bold",
                color=branch_color,
                zorder=5,
            )

    if show_board and scene.board_corner_B is not None:
        C = scene.board_corner_B
        R = scene.board_rotation_B
        u_extent, v_extent = scene.board_extent_uv
        u = R[:, 0]
        v = R[:, 1]
        corners = np.array(
            [C, C + u_extent * u, C + u_extent * u + v_extent * v, C + v_extent * v]
        )
        patch = Poly3DCollection(
            [corners],
            facecolors="#f2d38b",
            edgecolors="#7a5a18",
            linewidths=1.0,
            alpha=0.28,
        )
        ax.add_collection3d(patch)
        ax.scatter(*C, marker="D", s=26, color="#7a5a18", depthshade=False)
        if show_labels:
            ax.text(*(C + 0.008 * R[:, 2]), "C", fontsize=9, color="#5d430f")

    if show_base_origin:
        draw_triad(ax, np.zeros(3), np.eye(3), 1.5 * axis_length, linewidth=2.0)
        ax.text(0.0, 0.0, 0.0, r"  $\{B\}$", fontsize=9)

    if show_base_axes or show_grid:
        ax.set_xlabel(r"$x_B$ (m)", labelpad=8)
        ax.set_ylabel(r"$y_B$ (m)", labelpad=8)
        ax.set_zlabel(r"$z_B$ (m)", labelpad=8)
    else:
        # Paper-view default: remove the 3-D axes box, labels and all numeric
        # ticks. This does not affect the data-derived flange triads.
        ax.set_axis_off()
    if not clean:
        ax.set_title(
            "Six-pose star acquisition in robot base frame\n"
            "URDF tool visuals at each pose; RGB arrows: flange axes",
            fontsize=11,
            pad=16,
        )
    ax.grid(show_grid)

    if clean:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)

    legend_handles = [
        Line2D([0], [0], color="#d62728", lw=2, label=r"flange $x_F$"),
        Line2D([0], [0], color="#2ca02c", lw=2, label=r"flange $y_F$"),
        Line2D([0], [0], color="#1f77b4", lw=2, label=r"flange $z_F$"),
        Patch(
            facecolor="#999999",
            alpha=0.48,
            label="Gocator = pose color",
        ),
    ]
    if show_transform_guides:
        legend_handles.insert(
            3,
            Line2D(
                [0], [0], color="#707070", lw=1, label=r"$F\rightarrow S$ guide"
            ),
        )
    if weld_gun_mesh is not None:
        legend_handles.append(
            Patch(facecolor=weld_gun_mesh.material_rgba[:3], label="weld gun")
        )
    axis_legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        framealpha=0.90,
        title="Coordinate axes",
        title_fontsize=8,
    )
    ax.add_artist(axis_legend)
    if show_labels:
        pose_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                markersize=4.5,
                linestyle="--" if pose.index else "none",
                lw=1.0,
                label=f"{pose.index}: {pose.display_label}",
            )
            for pose, color in zip(scene.poses, branch_colors)
        ]
        ax.legend(
            handles=pose_handles,
            loc="upper right",
            fontsize=7.5,
            framealpha=0.90,
            title="Final seed poses",
            title_fontsize=8,
        )

    points = _scene_points(
        scene,
        gocator_mesh,
        weld_gun_mesh,
        show_board,
        show_base_origin,
    )
    set_equal_limits(ax, points)


def set_view(ax, elev: float, azim: float, roll: float):
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:  # Matplotlib versions before roll support.
        ax.view_init(elev=elev, azim=azim)


def save_clean_figure(
    scene: SceneData,
    gocator_mesh: TriangleMesh,
    weld_gun_mesh: TriangleMesh | None,
    output: Path,
    *,
    elev: float,
    azim: float,
    roll: float,
    dpi: int,
    axis_length: float,
    show_labels: bool,
    show_board: bool,
    show_base_origin: bool,
    show_transform_guides: bool,
    show_grid: bool,
    show_base_axes: bool,
    transparent: bool,
):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10.0, 8.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    render_scene(
        axis,
        scene,
        gocator_mesh,
        weld_gun_mesh,
        axis_length=axis_length,
        show_labels=show_labels,
        show_board=show_board,
        show_base_origin=show_base_origin,
        show_transform_guides=show_transform_guides,
        show_grid=show_grid,
        show_base_axes=show_base_axes,
        clean=True,
    )
    set_view(axis, elev, azim, roll)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, transparent=transparent, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Saved: {output}\n"
        f"View: elev={elev:.1f}, azim={azim:.1f}, roll={roll:.1f}"
    )


def write_metadata(
    scene: SceneData,
    gocator_mesh: TriangleMesh,
    weld_gun_mesh: TriangleMesh | None,
    output: Path,
    view: dict[str, float],
):
    reference = scene.poses[0]
    records = []
    for pose in scene.poses:
        relative_rotation = reference.R_BF.T @ pose.R_BF
        R_BS_rviz, t_BS_rviz = _urdf_link_pose(pose, gocator_mesh)
        records.append(
            {
                "index": pose.index,
                "label": pose.label,
                "R_BF": pose.R_BF.tolist(),
                "t_BF_m": pose.t_BF.tolist(),
                "R_BS": pose.R_BS.tolist(),
                "t_BS_m": pose.t_BS.tolist(),
                "R_BS_rviz_visual": R_BS_rviz.tolist(),
                "t_BS_rviz_visual_m": t_BS_rviz.tolist(),
                "joints_deg": np.rad2deg(pose.joints_rad).tolist(),
                "relative_to_reference_rotation_vector_deg_in_reference_flange": (
                    _rotation_vector_deg(relative_rotation).tolist()
                ),
                "relative_to_reference_rotation_angle_deg": _rotation_angle_deg(
                    relative_rotation
                ),
                "relative_to_reference_translation_B_mm": (
                    1000.0 * (pose.t_BF - reference.t_BF)
                ).tolist(),
            }
        )
    payload = {
        "source_run": str(scene.run_dir),
        "transform_convention": {
            "flange": "^B T_F from seeds.json",
            "calibration_measurement_frame": (
                "^F T_S from calibration_result.json; used only to interpret "
                "observations and the estimated board"
            ),
            "rviz_sensor_link": (
                "^F T_S from the URDF fixed joint; used to place the Gocator body"
            ),
            "legacy_R_BS_fields": (
                "R_BS/t_BS are calibration-frame poses; R_BS_rviz_visual/"
                "t_BS_rviz_visual_m are the displayed URDF-link poses"
            ),
        },
        "view": view,
        "gocator_visual": {
            "urdf": str(gocator_mesh.source_urdf),
            "mesh": str(gocator_mesh.source_mesh),
            "visual_origin_xyz_m": gocator_mesh.visual_origin_xyz_m.tolist(),
            "visual_origin_rpy_rad": gocator_mesh.visual_origin_rpy_rad.tolist(),
            "fixed_joint": gocator_mesh.fixed_joint_name,
            "fixed_joint_R_FS": gocator_mesh.R_FS_rviz.tolist(),
            "fixed_joint_t_FS_m": gocator_mesh.t_FS_rviz.tolist(),
            "calibration_to_rviz_translation_difference_mm": (
                1000.0 * (scene.t_FS - gocator_mesh.t_FS_rviz)
            ).tolist(),
            "calibration_to_rviz_rotation_difference_deg": _rotation_angle_deg(
                gocator_mesh.R_FS_rviz.T @ scene.R_FS
            ),
            "source_triangle_count": gocator_mesh.source_face_count,
            "display_triangle_count": int(len(gocator_mesh.faces)),
        },
        "weld_gun_visual": (
            {
                "enabled": True,
                "urdf": str(weld_gun_mesh.source_urdf),
                "mesh": str(weld_gun_mesh.source_mesh),
                "visual_origin_xyz_m": weld_gun_mesh.visual_origin_xyz_m.tolist(),
                "visual_origin_rpy_rad": weld_gun_mesh.visual_origin_rpy_rad.tolist(),
                "fixed_joint": weld_gun_mesh.fixed_joint_name,
                "fixed_joint_R_FW": weld_gun_mesh.R_FS_rviz.tolist(),
                "fixed_joint_t_FW_m": weld_gun_mesh.t_FS_rviz.tolist(),
                "source_triangle_count": weld_gun_mesh.source_face_count,
                "display_triangle_count": int(len(weld_gun_mesh.faces)),
            }
            if weld_gun_mesh is not None
            else {"enabled": False}
        ),
        "poses": records,
        "notes": [
            "Star lines connect the reference flange pose to each final target pose.",
            "The Gocator body is the URDF visual mesh simplified only for interactive rendering.",
            "Each Gocator body uses the same translucent branch color as its seed-pose identity.",
            "The body pose follows the same fanuc_flange-to-gocator_sensor fixed joint used by RViz.",
            "The calibration measurement-frame transform is not used to place the visual body.",
            "The displayed board-patch extent is schematic; its pose is data-derived.",
            "All flange/sensor poses and coordinate-axis directions are data-derived.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def interactive(
    scene: SceneData,
    gocator_mesh: TriangleMesh,
    weld_gun_mesh: TriangleMesh | None,
    args,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    figure = plt.figure(figsize=(12.0, 8.6))
    figure.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.18)
    axis = figure.add_subplot(111, projection="3d")
    render_scene(
        axis,
        scene,
        gocator_mesh,
        weld_gun_mesh,
        axis_length=args.axis_length,
        show_labels=not args.hide_labels,
        show_board=not args.hide_board,
        show_base_origin=args.show_base_origin,
        show_transform_guides=args.show_transform_guides,
        show_grid=args.show_grid,
        show_base_axes=args.show_base_axes,
        clean=False,
    )
    set_view(axis, args.elev, args.azim, args.roll)

    elevation_axis = figure.add_axes([0.12, 0.105, 0.58, 0.025])
    azimuth_axis = figure.add_axes([0.12, 0.070, 0.58, 0.025])
    roll_axis = figure.add_axes([0.12, 0.035, 0.58, 0.025])
    elevation = Slider(elevation_axis, "Elevation", -90.0, 90.0, valinit=args.elev)
    azimuth = Slider(azimuth_axis, "Azimuth", -180.0, 180.0, valinit=args.azim)
    roll = Slider(roll_axis, "Roll", -180.0, 180.0, valinit=args.roll)

    save_axis = figure.add_axes([0.75, 0.075, 0.10, 0.045])
    reset_axis = figure.add_axes([0.87, 0.075, 0.10, 0.045])
    save_button = Button(save_axis, "Save PNG")
    reset_button = Button(reset_axis, "Reset view")

    def update_view(_value=None):
        set_view(axis, elevation.val, azimuth.val, roll.val)
        figure.canvas.draw_idle()

    def current_view() -> tuple[float, float, float]:
        return (
            float(getattr(axis, "elev", elevation.val)),
            float(getattr(axis, "azim", azimuth.val)),
            float(getattr(axis, "roll", roll.val)),
        )

    def save_callback(_event=None):
        elev, azim, roll_value = current_view()
        save_clean_figure(
            scene,
            gocator_mesh,
            weld_gun_mesh,
            args.output,
            elev=elev,
            azim=azim,
            roll=roll_value,
            dpi=args.dpi,
            axis_length=args.axis_length,
            show_labels=not args.hide_labels,
            show_board=not args.hide_board,
            show_base_origin=args.show_base_origin,
            show_transform_guides=args.show_transform_guides,
            show_grid=args.show_grid,
            show_base_axes=args.show_base_axes,
            transparent=args.transparent,
        )
        write_metadata(
            scene,
            gocator_mesh,
            weld_gun_mesh,
            args.metadata,
            {"elevation_deg": elev, "azimuth_deg": azim, "roll_deg": roll_value},
        )

    def reset_callback(_event=None):
        elevation.reset()
        azimuth.reset()
        roll.reset()
        update_view()

    def key_callback(event):
        if event.key == "s":
            save_callback()
        elif event.key == "p":
            elev, azim, roll_value = current_view()
            print(
                f"Current view: --elev {elev:.1f} --azim {azim:.1f} "
                f"--roll {roll_value:.1f}"
            )
        elif event.key == "r":
            reset_callback()

    elevation.on_changed(update_view)
    azimuth.on_changed(update_view)
    roll.on_changed(update_view)
    save_button.on_clicked(save_callback)
    reset_button.on_clicked(reset_callback)
    figure.canvas.mpl_connect("key_press_event", key_callback)

    print(
        "Interactive controls:\n"
        "  mouse drag: rotate; mouse wheel/right drag: zoom (backend dependent)\n"
        "  sliders: exact elevation / azimuth / roll\n"
        "  s: save clean PNG; p: print current angles; r: reset view"
    )
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize six real star-acquisition flange and sensor poses."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="URDF that defines the gocator_sensor visual mesh and origin",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/workspace/star_seed_pose_figure/star_seed_pose_data.json"),
    )
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--axis-length", type=float, default=0.055)
    parser.add_argument(
        "--mesh-voxel-mm",
        type=float,
        default=2.0,
        help="display-mesh vertex-clustering size; smaller is more detailed and slower",
    )
    parser.add_argument(
        "--weld-gun-voxel-mm",
        type=float,
        default=5.0,
        help="weld-gun display simplification; independent of the Gocator mesh",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-board", action="store_true")
    parser.add_argument(
        "--hide-weld-gun",
        action="store_true",
        help="do not draw the URDF weld_gun visual at the six poses",
    )
    parser.add_argument("--show-base-origin", action="store_true")
    parser.add_argument(
        "--show-transform-guides",
        action="store_true",
        help="show flange-to-sensor and short +z_S diagnostic lines",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="show the 3-D background grid; hidden by default",
    )
    parser.add_argument(
        "--show-base-axes",
        action="store_true",
        help="show x_B/y_B/z_B labels, axis box and numeric ticks",
    )
    parser.add_argument("--transparent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.save_only or not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")

    scene = load_scene(args.run_dir.resolve())
    gocator_mesh = load_gocator_mesh(
        args.urdf.resolve(), voxel_size_mm=args.mesh_voxel_mm
    )
    weld_gun_mesh = None
    if not args.hide_weld_gun:
        weld_gun_mesh = load_weld_gun_mesh(
            args.urdf.resolve(), voxel_size_mm=args.weld_gun_voxel_mm
        )
    print(
        f"Gocator visual: {gocator_mesh.source_mesh} | "
        f"RViz joint: {gocator_mesh.fixed_joint_name}, "
        f"xyz={gocator_mesh.t_FS_rviz.tolist()} m | "
        f"URDF origin xyz={gocator_mesh.visual_origin_xyz_m.tolist()} m | "
        f"triangles {gocator_mesh.source_face_count} -> {len(gocator_mesh.faces)}"
    )
    if weld_gun_mesh is not None:
        print(
            f"Weld-gun visual: {weld_gun_mesh.source_mesh} | "
            f"RViz joint: {weld_gun_mesh.fixed_joint_name}, "
            f"xyz={weld_gun_mesh.t_FS_rviz.tolist()} m | "
            f"triangles {weld_gun_mesh.source_face_count} -> "
            f"{len(weld_gun_mesh.faces)}"
        )
    if args.save_only or not os.environ.get("DISPLAY"):
        if not args.save_only and not os.environ.get("DISPLAY"):
            print(
                "DISPLAY is unavailable; generating a static preview. "
                "Run in a desktop terminal without --save-only for mouse interaction."
            )
        save_clean_figure(
            scene,
            gocator_mesh,
            weld_gun_mesh,
            args.output.resolve(),
            elev=args.elev,
            azim=args.azim,
            roll=args.roll,
            dpi=args.dpi,
            axis_length=args.axis_length,
            show_labels=not args.hide_labels,
            show_board=not args.hide_board,
            show_base_origin=args.show_base_origin,
            show_transform_guides=args.show_transform_guides,
            show_grid=args.show_grid,
            show_base_axes=args.show_base_axes,
            transparent=args.transparent,
        )
        write_metadata(
            scene,
            gocator_mesh,
            weld_gun_mesh,
            args.metadata.resolve(),
            {
                "elevation_deg": args.elev,
                "azimuth_deg": args.azim,
                "roll_deg": args.roll,
            },
        )
        return

    interactive(scene, gocator_mesh, weld_gun_mesh, args)


if __name__ == "__main__":
    main()
