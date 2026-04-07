from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from .models import ExportRequest
from .planner import (
    PI,
    JointLimits,
    LinkLengths,
    PlannerSettings,
    RRTPlanner,
    angle_delta,
    build_cspace_grid,
    coerce_joint_limits,
    coerce_link_lengths,
    normalize_angles,
    resolve_wrap_mode,
)

TAU = 2.0 * PI
TORUS_MAJOR_RADIUS = 1.35
TORUS_MINOR_RADIUS = 0.55
TORUS_ROT_X = 0.92
TORUS_ROT_Y = -0.88
TORUS_CAMERA_Z = 5.2
TORUS_SCALE = 82
TORUS_PLOT_SIZE = 440
TORUS_SURFACE_U = 72
TORUS_SURFACE_V = 48
EXPORT_FIGSIZE = (12.4, 4.2)


@dataclass
class ExportBundle:
    wrap_angles: bool
    start: np.ndarray
    goal: np.ndarray
    obstacle_rect: dict[str, float]
    joint_limits: JointLimits
    link_lengths: LinkLengths
    cspace: np.ndarray
    tree: list[np.ndarray]
    parents: list[int | None]
    path: list[np.ndarray]
    dense_path: list[np.ndarray]
    iterations: int
    time_ms: float


def obstacle_to_rect(x: float, y: float, size: float) -> dict[str, float]:
    half = size / 2.0
    return {
        "xmin": x - half,
        "xmax": x + half,
        "ymin": y - half,
        "ymax": y + half,
    }


def get_arm_coords(theta1: float, theta2: float, l1: float, l2: float) -> tuple[np.ndarray, np.ndarray]:
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.asarray([0.0, x1, x2], dtype=float), np.asarray([0.0, y1, y2], dtype=float)


def to_torus_angles(q: np.ndarray, joint_limits: JointLimits) -> tuple[float, float]:
    widths = joint_limits.widths_array()
    u = ((float(q[0]) - joint_limits.theta1_min) / widths[0]) * TAU - PI
    v = ((float(q[1]) - joint_limits.theta2_min) / widths[1]) * TAU - PI
    return float(u), float(v)


def torus_point(u: float, v: float) -> tuple[float, float, float]:
    ring = TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * np.cos(v)
    x = ring * np.cos(u)
    y = ring * np.sin(u)
    z = TORUS_MINOR_RADIUS * np.sin(v)
    return float(x), float(y), float(z)


def rotate_point3d(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    cos_y = np.cos(TORUS_ROT_Y)
    sin_y = np.sin(TORUS_ROT_Y)
    x_y = x * cos_y + z * sin_y
    z_y = -x * sin_y + z * cos_y

    cos_x = np.cos(TORUS_ROT_X)
    sin_x = np.sin(TORUS_ROT_X)
    y_x = y * cos_x - z_y * sin_x
    z_x = y * sin_x + z_y * cos_x
    return np.asarray([x_y, y_x, z_x], dtype=float)


def project_point3d(point: np.ndarray, size_px: int = TORUS_PLOT_SIZE) -> np.ndarray:
    perspective = TORUS_CAMERA_Z / (TORUS_CAMERA_Z - point[2])
    px = size_px / 2.0 + point[0] * TORUS_SCALE * perspective
    py = size_px / 2.0 - point[1] * TORUS_SCALE * perspective
    return np.asarray([px, py, point[2], perspective], dtype=float)


def render_torus_background(cspace: np.ndarray) -> np.ndarray:
    image = Image.new("RGBA", (TORUS_PLOT_SIZE, TORUS_PLOT_SIZE), (248, 252, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    c_h, c_w = cspace.shape
    light_dir = np.asarray([0.28, 0.38, 0.88], dtype=float)
    quads: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []

    def blocked_at(u: float, v: float) -> bool:
        fx = ((u + PI) / TAU) * (c_w - 1)
        fy = ((v + PI) / TAU) * (c_h - 1)
        ix = int(np.clip(np.floor(fx), 0, c_w - 1))
        iy = int(np.clip(np.floor(fy), 0, c_h - 1))
        return bool(cspace[iy, ix] == 1)

    for u_idx in range(TORUS_SURFACE_U):
        u0 = -PI + (u_idx / TORUS_SURFACE_U) * TAU
        u1 = -PI + ((u_idx + 1) / TORUS_SURFACE_U) * TAU
        for v_idx in range(TORUS_SURFACE_V):
            v0 = -PI + (v_idx / TORUS_SURFACE_V) * TAU
            v1 = -PI + ((v_idx + 1) / TORUS_SURFACE_V) * TAU

            corners_rot = [
                rotate_point3d(np.asarray(torus_point(u0, v0), dtype=float)),
                rotate_point3d(np.asarray(torus_point(u1, v0), dtype=float)),
                rotate_point3d(np.asarray(torus_point(u1, v1), dtype=float)),
                rotate_point3d(np.asarray(torus_point(u0, v1), dtype=float)),
            ]
            corners_projected = [project_point3d(p, TORUS_PLOT_SIZE) for p in corners_rot]
            points = [(float(p[0]), float(p[1])) for p in corners_projected]
            avg_depth = float(sum(p[2] for p in corners_rot) / 4.0)

            uc = -PI + ((u_idx + 0.5) / TORUS_SURFACE_U) * TAU
            vc = -PI + ((v_idx + 0.5) / TORUS_SURFACE_V) * TAU

            n = rotate_point3d(
                np.asarray(
                    [np.cos(uc) * np.cos(vc), np.sin(uc) * np.cos(vc), np.sin(vc)],
                    dtype=float,
                )
            )
            dot = max(0.0, float(np.dot(n, light_dir)))
            intensity = 0.26 + 0.74 * dot

            if blocked_at(uc, vc):
                rgba = (178, 24, 24, 178)
            else:
                r = int(round(38 + 82 * intensity))
                g = int(round(92 + 102 * intensity))
                b = int(round(132 + 96 * intensity))
                rgba = (r, g, b, 178)

            quads.append((avg_depth, points, rgba))

    quads.sort(key=lambda item: item[0])
    for _depth, points, rgba in quads:
        draw.polygon(points, fill=rgba)

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def densify_path(
    path: list[np.ndarray],
    wrap_angles: bool,
    joint_limits: JointLimits,
    samples_per_edge: int,
) -> list[np.ndarray]:
    if len(path) < 2:
        return [np.asarray(path[0], dtype=float)] if path else []

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    for idx in range(1, len(path)):
        q1 = np.asarray(path[idx - 1], dtype=float)
        q2 = np.asarray(path[idx], dtype=float)
        diff = angle_delta(q2, q1, wrap_angles, joint_limits=joint_limits)

        for step in range(1, samples_per_edge + 1):
            alpha = step / samples_per_edge
            q = q1 + alpha * diff
            dense.append(normalize_angles(q, wrap_angles, joint_limits=joint_limits))

    return dense


def limit_frame_count(path: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    if len(path) <= max_frames:
        return path

    idxs = np.linspace(0, len(path) - 1, max_frames, dtype=int)
    unique_idxs = np.unique(idxs)
    return [path[int(i)] for i in unique_idxs]


def build_cspace_segments(
    path: list[np.ndarray],
    wrap_angles: bool,
    joint_limits: JointLimits,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    half_span1 = (joint_limits.theta1_max - joint_limits.theta1_min) / 2.0
    half_span2 = (joint_limits.theta2_max - joint_limits.theta2_min) / 2.0

    for idx, q in enumerate(path):
        if idx > 0 and wrap_angles:
            prev = path[idx - 1]
            if abs(float(q[0]) - float(prev[0])) > half_span1 or abs(float(q[1]) - float(prev[1])) > half_span2:
                xs.append(np.nan)
                ys.append(np.nan)
        xs.append(float(q[0]))
        ys.append(float(q[1]))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle

        return plt, animation, ListedColormap, Rectangle
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for MP4 export. Install it in the backend environment.") from exc


def build_export_bundle(request: ExportRequest) -> ExportBundle:
    joint_limits = coerce_joint_limits(request.joint_limits.model_dump())
    link_lengths = coerce_link_lengths(request.link_lengths.model_dump())
    wrap_angles = resolve_wrap_mode(request.wrap_angles, joint_limits=joint_limits)

    start = normalize_angles(np.asarray(request.start, dtype=float), wrap_angles, joint_limits=joint_limits)
    goal = normalize_angles(np.asarray(request.goal, dtype=float), wrap_angles, joint_limits=joint_limits)

    rect = obstacle_to_rect(request.obstacle.x, request.obstacle.y, request.obstacle.size)
    settings = PlannerSettings(
        expand_dist=request.expand_dist,
        goal_rate=request.goal_rate,
        max_iter=request.max_iter,
        edge_check_steps=request.edge_check_steps,
        collision_buffer=request.collision_buffer,
        seed=request.seed,
    )

    planner = RRTPlanner(
        start=start,
        goal=goal,
        rect=rect,
        settings=settings,
        wrap_angles=wrap_angles,
        joint_limits=joint_limits,
        link_lengths=link_lengths,
    )
    result = planner.plan()
    if not result.success or result.path is None:
        raise RuntimeError(
            "Could not export MP4: planner did not find a path with current settings."
        )

    cspace = np.asarray(
        build_cspace_grid(
            rect,
            resolution=request.cspace_resolution,
            buffer=request.collision_buffer,
            joint_limits=joint_limits,
            link_lengths=link_lengths,
        ),
        dtype=np.uint8,
    )

    dense_path = densify_path(
        result.path,
        wrap_angles=wrap_angles,
        joint_limits=joint_limits,
        samples_per_edge=request.samples_per_edge,
    )
    dense_path = limit_frame_count(dense_path, request.max_frames)

    return ExportBundle(
        wrap_angles=wrap_angles,
        start=np.asarray(start, dtype=float),
        goal=np.asarray(goal, dtype=float),
        obstacle_rect=rect,
        joint_limits=joint_limits,
        link_lengths=link_lengths,
        cspace=cspace,
        tree=result.tree,
        parents=result.parents,
        path=result.path,
        dense_path=dense_path,
        iterations=result.iterations,
        time_ms=result.time_ms,
    )


def render_mp4(bundle: ExportBundle, output_path: Path, fps: int, dpi: int, bitrate: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on the backend host. Install ffmpeg to export MP4.")

    plt, animation, ListedColormap, Rectangle = _load_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    workspace_limit = max(1.0, bundle.link_lengths.l1 + bundle.link_lengths.l2 + 0.2)
    cspace_cmap = ListedColormap(["#eef6ff", "#3d4a5f"])
    t1_min, t1_max = bundle.joint_limits.theta1_min, bundle.joint_limits.theta1_max
    t2_min, t2_max = bundle.joint_limits.theta2_min, bundle.joint_limits.theta2_max
    t1_mid = (t1_min + t1_max) / 2.0
    t2_mid = (t2_min + t2_max) / 2.0

    fig = plt.figure(figsize=EXPORT_FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.90, bottom=0.12, wspace=0.20)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2])
    ax_workspace = fig.add_subplot(grid[0, 0])
    ax_cspace = fig.add_subplot(grid[0, 1])
    ax_torus = fig.add_subplot(grid[0, 2])

    ax_workspace.set_title("Workspace", fontsize=16, fontweight="bold", pad=10)
    ax_workspace.set_xlim(-workspace_limit, workspace_limit)
    ax_workspace.set_ylim(-workspace_limit, workspace_limit)
    ax_workspace.set_aspect("equal", adjustable="box")
    ax_workspace.set_xlabel("x", fontsize=13, fontweight="bold", labelpad=12)
    ax_workspace.set_ylabel("y", fontsize=13, fontweight="bold", labelpad=12)
    ax_workspace.tick_params(axis="both", labelsize=11)
    ax_workspace.axhline(0.0, color="#9fb2c6", linewidth=1.2, zorder=0)
    ax_workspace.axvline(0.0, color="#9fb2c6", linewidth=1.2, zorder=0)

    rect = bundle.obstacle_rect
    obstacle_patch = Rectangle(
        (rect["xmin"], rect["ymin"]),
        rect["xmax"] - rect["xmin"],
        rect["ymax"] - rect["ymin"],
        facecolor="#c85d5d",
        edgecolor="#7b1f1f",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
    )
    ax_workspace.add_patch(obstacle_patch)

    start_x, start_y = get_arm_coords(
        float(bundle.start[0]), float(bundle.start[1]), bundle.link_lengths.l1, bundle.link_lengths.l2
    )
    goal_x, goal_y = get_arm_coords(
        float(bundle.goal[0]), float(bundle.goal[1]), bundle.link_lengths.l1, bundle.link_lengths.l2
    )
    ax_workspace.plot(start_x, start_y, color="#1b9e77", linewidth=2.5, alpha=0.85, label="Start")
    ax_workspace.plot(goal_x, goal_y, color="#1d5ac7", linewidth=2.5, alpha=0.85, label="Goal")

    curr_x, curr_y = get_arm_coords(
        float(bundle.dense_path[0][0]),
        float(bundle.dense_path[0][1]),
        bundle.link_lengths.l1,
        bundle.link_lengths.l2,
    )
    (current_arm_line,) = ax_workspace.plot(curr_x, curr_y, color="#111111", linewidth=3.2, label="Current")
    current_joints = ax_workspace.scatter(curr_x, curr_y, color="#111111", s=[26, 28, 34], zorder=4)
    ax_workspace.legend(loc="upper right", fontsize=10)

    ax_cspace.set_title("C-Space / No-Go Zone", fontsize=16, fontweight="bold", pad=10)
    ax_cspace.imshow(
        bundle.cspace,
        extent=(t1_min, t1_max, t2_min, t2_max),
        origin="lower",
        cmap=cspace_cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax_cspace.set_xlim(t1_min, t1_max)
    ax_cspace.set_ylim(t2_min, t2_max)
    ax_cspace.set_xlabel("theta1", fontsize=13, fontweight="bold", labelpad=12)
    ax_cspace.set_ylabel("theta2", fontsize=13, fontweight="bold", labelpad=12)
    ax_cspace.set_xticks([t1_min, t1_mid, t1_max], [f"{t1_min:.2f}", f"{t1_mid:.2f}", f"{t1_max:.2f}"])
    ax_cspace.set_yticks([t2_min, t2_mid, t2_max], [f"{t2_min:.2f}", f"{t2_mid:.2f}", f"{t2_max:.2f}"])
    ax_cspace.tick_params(axis="both", labelsize=11)
    ax_cspace.axhline(t2_mid, color="#9fb2c6", linewidth=1.2)
    ax_cspace.axvline(t1_mid, color="#9fb2c6", linewidth=1.2)

    for idx in range(1, len(bundle.tree)):
        parent = bundle.parents[idx]
        if parent is None:
            continue
        p1 = bundle.tree[parent]
        p2 = bundle.tree[idx]
        if bundle.wrap_angles:
            half_span = bundle.joint_limits.widths_array() / 2.0
            if abs(float(p2[0] - p1[0])) > half_span[0] or abs(float(p2[1] - p1[1])) > half_span[1]:
                continue
        ax_cspace.plot(
            [float(p1[0]), float(p2[0])],
            [float(p1[1]), float(p2[1])],
            color="#6a7b91",
            linewidth=0.6,
            alpha=0.23,
        )

    path_xs, path_ys = build_cspace_segments(bundle.path, bundle.wrap_angles, bundle.joint_limits)
    ax_cspace.plot(path_xs, path_ys, color="#f18f01", linewidth=2.2, alpha=0.95)
    ax_cspace.scatter([bundle.start[0]], [bundle.start[1]], color="#1b9e77", s=52, zorder=4)
    ax_cspace.scatter([bundle.goal[0]], [bundle.goal[1]], color="#1d5ac7", s=52, zorder=4)
    (cspace_current_marker,) = ax_cspace.plot(
        [bundle.dense_path[0][0]],
        [bundle.dense_path[0][1]],
        marker="o",
        markersize=8,
        color="#111111",
        zorder=5,
    )

    ax_torus.set_title("Torus C-Space", fontsize=16, fontweight="bold", pad=10)
    torus_bg = render_torus_background(bundle.cspace)
    ax_torus.imshow(torus_bg, extent=(0, TORUS_PLOT_SIZE, TORUS_PLOT_SIZE, 0))
    ax_torus.set_xlim(0, TORUS_PLOT_SIZE)
    ax_torus.set_ylim(TORUS_PLOT_SIZE, 0)
    ax_torus.set_aspect("equal")
    ax_torus.set_xticks([])
    ax_torus.set_yticks([])
    ax_torus.set_frame_on(True)

    def torus_project(q: np.ndarray) -> tuple[float, float]:
        u, v = to_torus_angles(np.asarray(q, dtype=float), bundle.joint_limits)
        p = project_point3d(rotate_point3d(np.asarray(torus_point(u, v), dtype=float)), TORUS_PLOT_SIZE)
        return float(p[0]), float(p[1])

    torus_path_xy = np.asarray([torus_project(np.asarray(q, dtype=float)) for q in bundle.path], dtype=float)
    if len(torus_path_xy) > 1:
        ax_torus.plot(torus_path_xy[:, 0], torus_path_xy[:, 1], color=(1.0, 1.0, 1.0, 0.8), linewidth=5.0, zorder=3)
        ax_torus.plot(torus_path_xy[:, 0], torus_path_xy[:, 1], color="#f18f01", linewidth=3.0, zorder=4)

    sx, sy = torus_project(bundle.start)
    gx, gy = torus_project(bundle.goal)
    cx, cy = torus_project(bundle.dense_path[0])
    ax_torus.scatter([sx], [sy], color="#1b9e77", s=44, zorder=5)
    ax_torus.scatter([gx], [gy], color="#1d5ac7", s=44, zorder=5)
    (torus_current_marker,) = ax_torus.plot([cx], [cy], marker="o", markersize=7, color="#111111", zorder=6)

    status_text = fig.text(
        0.5,
        0.01,
        (
            f"Iterations: {bundle.iterations}    "
            f"Planner Time: {bundle.time_ms:.1f} ms    "
            f"Frames: {len(bundle.dense_path)}"
        ),
        ha="center",
        fontsize=10,
    )

    def update(frame_idx: int):
        q = bundle.dense_path[frame_idx]
        x_vals, y_vals = get_arm_coords(float(q[0]), float(q[1]), bundle.link_lengths.l1, bundle.link_lengths.l2)
        current_arm_line.set_data(x_vals, y_vals)
        current_joints.set_offsets(np.column_stack([x_vals, y_vals]))
        cspace_current_marker.set_data([float(q[0])], [float(q[1])])

        tx, ty = torus_project(q)
        torus_current_marker.set_data([tx], [ty])

        status_text.set_text(
            f"Iterations: {bundle.iterations}    Planner Time: {bundle.time_ms:.1f} ms    "
            f"Frame: {frame_idx + 1}/{len(bundle.dense_path)}"
        )
        return current_arm_line, current_joints, cspace_current_marker, torus_current_marker, status_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(bundle.dense_path),
        interval=1000.0 / float(fps),
        blit=False,
        repeat=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=bitrate,
        extra_args=[
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ],
    )
    try:
        anim.save(str(output_path), writer=writer, dpi=dpi)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MP4 export failed during ffmpeg encoding: {exc}") from exc
    plt.close(fig)


def generate_mp4_bytes(request: ExportRequest) -> bytes:
    bundle = build_export_bundle(request)
    with tempfile.TemporaryDirectory(prefix="rrt_mp4_") as tmp_dir:
        out_path = Path(tmp_dir) / "rrt_animation.mp4"
        render_mp4(bundle, out_path, fps=request.fps, dpi=request.dpi, bitrate=request.bitrate)
        return out_path.read_bytes()
