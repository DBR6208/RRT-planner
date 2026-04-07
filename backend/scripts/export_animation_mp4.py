from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.planner import (  # noqa: E402
    PI,
    PlannerSettings,
    RRTPlanner,
    angle_delta,
    build_cspace_grid,
    normalize_angles,
)


WORKSPACE_LIMIT = 2.2
L1 = 1.0
L2 = 1.0

TORUS_MAJOR_RADIUS = 1.35
TORUS_MINOR_RADIUS = 0.55


@dataclass
class ExportBundle:
    wrap_angles: bool
    cspace: np.ndarray
    tree: list[np.ndarray]
    parents: list[int | None]
    path: list[np.ndarray]
    dense_path: list[np.ndarray]
    start: np.ndarray
    goal: np.ndarray
    obstacle_rect: dict[str, float]
    plan_iterations: int
    plan_time_ms: float


def obstacle_to_rect(x: float, y: float, size: float) -> dict[str, float]:
    half = size / 2.0
    return {
        "xmin": x - half,
        "xmax": x + half,
        "ymin": y - half,
        "ymax": y + half,
    }


def parse_pair(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated values, e.g. 0.0,1.57")
    try:
        return np.asarray([float(parts[0]), float(parts[1])], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid number in '{text}'") from exc


def get_arm_coords(theta1: float, theta2: float) -> tuple[np.ndarray, np.ndarray]:
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return np.asarray([0.0, x1, x2], dtype=float), np.asarray([0.0, y1, y2], dtype=float)


def torus_point(theta1: float, theta2: float) -> tuple[float, float, float]:
    ring = TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * np.cos(theta2)
    x = ring * np.cos(theta1)
    y = ring * np.sin(theta1)
    z = TORUS_MINOR_RADIUS * np.sin(theta2)
    return float(x), float(y), float(z)


def densify_path(path: list[np.ndarray], wrap_angles: bool, samples_per_edge: int) -> list[np.ndarray]:
    if len(path) < 2:
        return [np.asarray(path[0], dtype=float)] if path else []

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    for idx in range(1, len(path)):
        q1 = np.asarray(path[idx - 1], dtype=float)
        q2 = np.asarray(path[idx], dtype=float)

        diff = angle_delta(q2, q1, wrap_angles)
        for step in range(1, samples_per_edge + 1):
            alpha = step / samples_per_edge
            q = q1 + alpha * diff
            dense.append(normalize_angles(q, wrap_angles))

    return dense


def build_cspace_segments(path: list[np.ndarray], wrap_angles: bool) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    for idx, q in enumerate(path):
        if idx > 0 and wrap_angles:
            prev = path[idx - 1]
            if abs(float(q[0]) - float(prev[0])) > PI or abs(float(q[1]) - float(prev[1])) > PI:
                xs.append(np.nan)
                ys.append(np.nan)

        xs.append(float(q[0]))
        ys.append(float(q[1]))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plan_and_bundle(args: argparse.Namespace) -> ExportBundle:
    start = normalize_angles(np.asarray(args.start, dtype=float), args.wrap_angles)
    goal = normalize_angles(np.asarray(args.goal, dtype=float), args.wrap_angles)
    rect = obstacle_to_rect(args.obstacle_x, args.obstacle_y, args.obstacle_size)

    settings = PlannerSettings(
        expand_dist=args.expand_dist,
        goal_rate=args.goal_rate,
        max_iter=args.max_iter,
        edge_check_steps=args.edge_check_steps,
        collision_buffer=args.collision_buffer,
        seed=args.seed,
    )

    planner = RRTPlanner(
        start=start,
        goal=goal,
        rect=rect,
        settings=settings,
        wrap_angles=args.wrap_angles,
    )

    result = planner.plan()
    if not result.success or result.path is None:
        raise RuntimeError(
            "RRT did not find a path with the current settings. "
            "Try increasing --max-iter, --goal-rate, or --expand-dist."
        )

    cspace = np.asarray(
        build_cspace_grid(rect, resolution=args.cspace_resolution, buffer=args.collision_buffer),
        dtype=np.uint8,
    )

    dense_path = densify_path(result.path, wrap_angles=args.wrap_angles, samples_per_edge=args.samples_per_edge)

    return ExportBundle(
        wrap_angles=args.wrap_angles,
        cspace=cspace,
        tree=result.tree,
        parents=result.parents,
        path=result.path,
        dense_path=dense_path,
        start=np.asarray(start, dtype=float),
        goal=np.asarray(goal, dtype=float),
        obstacle_rect=rect,
        plan_iterations=result.iterations,
        plan_time_ms=result.time_ms,
    )


def create_animation(bundle: ExportBundle, output_path: Path, fps: int, dpi: int, bitrate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cspace_cmap = ListedColormap(["#eef6ff", "#3d4a5f"])

    fig = plt.figure(figsize=(17.0, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2])

    ax_workspace = fig.add_subplot(grid[0, 0])
    ax_cspace = fig.add_subplot(grid[0, 1])
    ax_torus = fig.add_subplot(grid[0, 2], projection="3d")

    # Workspace panel
    ax_workspace.set_title("Workspace", fontsize=16, fontweight="bold", pad=10)
    ax_workspace.set_xlim(-WORKSPACE_LIMIT, WORKSPACE_LIMIT)
    ax_workspace.set_ylim(-WORKSPACE_LIMIT, WORKSPACE_LIMIT)
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

    start_x, start_y = get_arm_coords(float(bundle.start[0]), float(bundle.start[1]))
    goal_x, goal_y = get_arm_coords(float(bundle.goal[0]), float(bundle.goal[1]))

    ax_workspace.plot(start_x, start_y, color="#1b9e77", linewidth=2.5, alpha=0.85, label="Start")
    ax_workspace.plot(goal_x, goal_y, color="#1d5ac7", linewidth=2.5, alpha=0.85, label="Goal")

    current_x, current_y = get_arm_coords(float(bundle.dense_path[0][0]), float(bundle.dense_path[0][1]))
    (current_arm_line,) = ax_workspace.plot(current_x, current_y, color="#111111", linewidth=3.2, label="Current")
    current_joints = ax_workspace.scatter(current_x, current_y, color="#111111", s=[26, 28, 34], zorder=4)
    ax_workspace.legend(loc="upper right", fontsize=10)

    # C-space panel
    ax_cspace.set_title("C-Space / No-Go Zone", fontsize=16, fontweight="bold", pad=10)
    ax_cspace.imshow(
        bundle.cspace,
        extent=(-PI, PI, -PI, PI),
        origin="lower",
        cmap=cspace_cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax_cspace.set_xlim(-PI, PI)
    ax_cspace.set_ylim(-PI, PI)
    ax_cspace.set_xlabel("theta1", fontsize=13, fontweight="bold", labelpad=12)
    ax_cspace.set_ylabel("theta2", fontsize=13, fontweight="bold", labelpad=12)
    ax_cspace.set_xticks([-PI, 0.0, PI], ["-pi", "0", "pi"])
    ax_cspace.set_yticks([-PI, 0.0, PI], ["-pi", "0", "pi"])
    ax_cspace.tick_params(axis="both", labelsize=11)
    ax_cspace.axhline(0.0, color="#9fb2c6", linewidth=1.2)
    ax_cspace.axvline(0.0, color="#9fb2c6", linewidth=1.2)

    for idx in range(1, len(bundle.tree)):
        parent = bundle.parents[idx]
        if parent is None:
            continue
        p1 = bundle.tree[parent]
        p2 = bundle.tree[idx]
        if bundle.wrap_angles and (abs(float(p2[0] - p1[0])) > PI or abs(float(p2[1] - p1[1])) > PI):
            continue
        ax_cspace.plot(
            [float(p1[0]), float(p2[0])],
            [float(p1[1]), float(p2[1])],
            color="#6a7b91",
            linewidth=0.6,
            alpha=0.23,
        )

    path_xs, path_ys = build_cspace_segments(bundle.path, bundle.wrap_angles)
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

    # Torus panel
    ax_torus.set_title("Torus C-Space", fontsize=16, fontweight="bold", pad=10)

    n_u = 84
    n_v = 56
    u_vals = np.linspace(-PI, PI, n_u)
    v_vals = np.linspace(-PI, PI, n_v)
    U, V = np.meshgrid(u_vals, v_vals, indexing="ij")

    X = (TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * np.cos(V)) * np.cos(U)
    Y = (TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * np.cos(V)) * np.sin(U)
    Z = TORUS_MINOR_RADIUS * np.sin(V)

    c_h, c_w = bundle.cspace.shape
    gx = np.clip(((U + PI) / (2.0 * PI) * (c_w - 1)).astype(int), 0, c_w - 1)
    gy = np.clip(((V + PI) / (2.0 * PI) * (c_h - 1)).astype(int), 0, c_h - 1)
    blocked = bundle.cspace[gy, gx] == 1

    facecolors = np.empty((*blocked.shape, 4), dtype=float)
    facecolors[~blocked] = np.array([0.20, 0.55, 0.75, 0.70], dtype=float)
    facecolors[blocked] = np.array([0.75, 0.14, 0.14, 0.70], dtype=float)

    ax_torus.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        zorder=1,
    )

    torus_path_xyz = np.asarray([torus_point(float(q[0]), float(q[1])) for q in bundle.path], dtype=float)
    ax_torus.plot(
        torus_path_xyz[:, 0],
        torus_path_xyz[:, 1],
        torus_path_xyz[:, 2],
        color="#f18f01",
        linewidth=3.0,
        zorder=3,
    )

    start_xyz = np.asarray(torus_point(float(bundle.start[0]), float(bundle.start[1])), dtype=float)
    goal_xyz = np.asarray(torus_point(float(bundle.goal[0]), float(bundle.goal[1])), dtype=float)
    curr_xyz = np.asarray(torus_point(float(bundle.dense_path[0][0]), float(bundle.dense_path[0][1])), dtype=float)

    ax_torus.scatter([start_xyz[0]], [start_xyz[1]], [start_xyz[2]], color="#1b9e77", s=44, depthshade=False, zorder=4)
    ax_torus.scatter([goal_xyz[0]], [goal_xyz[1]], [goal_xyz[2]], color="#1d5ac7", s=44, depthshade=False, zorder=4)
    torus_current_marker = ax_torus.scatter(
        [curr_xyz[0]],
        [curr_xyz[1]],
        [curr_xyz[2]],
        color="#111111",
        s=40,
        depthshade=False,
        zorder=5,
    )

    ax_torus.set_xlabel("X", fontsize=13, fontweight="bold", labelpad=10)
    ax_torus.set_ylabel("Y", fontsize=13, fontweight="bold", labelpad=10)
    ax_torus.set_zlabel("Z", fontsize=13, fontweight="bold", labelpad=10)
    ax_torus.tick_params(axis="both", labelsize=10)
    ax_torus.grid(False)
    ax_torus.view_init(elev=26, azim=-36)
    ax_torus.set_box_aspect((1.0, 1.0, 0.62))

    status_text = fig.text(
        0.5,
        0.01,
        (
            f"Iterations: {bundle.plan_iterations}    "
            f"Planner Time: {bundle.plan_time_ms:.1f} ms    "
            f"Frames: {len(bundle.dense_path)}"
        ),
        ha="center",
        fontsize=10,
    )

    def update(frame_idx: int):
        q = bundle.dense_path[frame_idx]

        x_vals, y_vals = get_arm_coords(float(q[0]), float(q[1]))
        current_arm_line.set_data(x_vals, y_vals)
        current_joints.set_offsets(np.column_stack([x_vals, y_vals]))

        cspace_current_marker.set_data([float(q[0])], [float(q[1])])

        x_t, y_t, z_t = torus_point(float(q[0]), float(q[1]))
        torus_current_marker._offsets3d = ([x_t], [y_t], [z_t])

        status_text.set_text(
            (
                f"Iterations: {bundle.plan_iterations}    "
                f"Planner Time: {bundle.plan_time_ms:.1f} ms    "
                f"Frame: {frame_idx + 1}/{len(bundle.dense_path)}"
            )
        )

        return (
            current_arm_line,
            current_joints,
            cspace_current_marker,
            torus_current_marker,
            status_text,
        )

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
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export planner animation to MP4.")

    parser.add_argument("--output", default="output/rrt_animation.mp4", help="Output MP4 path")
    parser.add_argument("--start", type=parse_pair, default=np.asarray([0.0, 0.0], dtype=float))
    parser.add_argument("--goal", type=parse_pair, default=np.asarray([PI / 2.0, 0.0], dtype=float))

    parser.add_argument("--obstacle-x", type=float, default=0.9)
    parser.add_argument("--obstacle-y", type=float, default=0.2)
    parser.add_argument("--obstacle-size", type=float, default=0.2)

    parser.add_argument("--expand-dist", type=float, default=0.03)
    parser.add_argument("--goal-rate", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=12000)
    parser.add_argument("--edge-check-steps", type=int, default=20)
    parser.add_argument("--collision-buffer", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wrap-angles", dest="wrap_angles", action="store_true", default=True)
    parser.add_argument("--no-wrap-angles", dest="wrap_angles", action="store_false")

    parser.add_argument("--cspace-resolution", type=int, default=720)
    parser.add_argument("--samples-per-edge", type=int, default=6)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--bitrate", type=int, default=2600)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output).resolve()

    print("[1/3] Running planner...")
    bundle = plan_and_bundle(args)

    print("[2/3] Rendering animation frames...")
    create_animation(bundle, output_path=output_path, fps=args.fps, dpi=args.dpi, bitrate=args.bitrate)

    print(f"[3/3] MP4 written to: {output_path}")


if __name__ == "__main__":
    main()
