from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Optional, Sequence

import numpy as np

PI = float(np.pi)
ANGLE_MIN = -PI
ANGLE_MAX = PI
FULL_CIRCLE = 2.0 * PI
WRAP_SPAN_TOL = 1e-6


@dataclass
class PlannerSettings:
    expand_dist: float = 0.03
    goal_rate: float = 0.1
    max_iter: int = 12_000
    edge_check_steps: int = 20
    collision_buffer: float = 0.05
    seed: Optional[int] = 42


@dataclass
class JointLimits:
    theta1_min: float = ANGLE_MIN
    theta1_max: float = ANGLE_MAX
    theta2_min: float = ANGLE_MIN
    theta2_max: float = ANGLE_MAX

    def mins_array(self) -> np.ndarray:
        return np.asarray([self.theta1_min, self.theta2_min], dtype=float)

    def maxs_array(self) -> np.ndarray:
        return np.asarray([self.theta1_max, self.theta2_max], dtype=float)

    def widths_array(self) -> np.ndarray:
        return self.maxs_array() - self.mins_array()


@dataclass
class LinkLengths:
    l1: float = 1.0
    l2: float = 1.0


@dataclass
class PlanData:
    success: bool
    tree: list[np.ndarray]
    parents: list[Optional[int]]
    path: Optional[list[np.ndarray]]
    iterations: int
    time_ms: float


def coerce_joint_limits(raw: JointLimits | dict[str, float] | None = None) -> JointLimits:
    if raw is None:
        limits = JointLimits()
    elif isinstance(raw, JointLimits):
        limits = raw
    else:
        limits = JointLimits(**raw)

    widths = limits.widths_array()
    if np.any(widths <= 0.0):
        raise ValueError("Each joint limit max must be greater than min.")

    return limits


def coerce_link_lengths(raw: LinkLengths | dict[str, float] | None = None) -> LinkLengths:
    if raw is None:
        lengths = LinkLengths()
    elif isinstance(raw, LinkLengths):
        lengths = raw
    else:
        lengths = LinkLengths(**raw)

    if lengths.l1 <= 0.0 or lengths.l2 <= 0.0:
        raise ValueError("Link lengths must be positive.")

    return lengths


def supports_wrapping(joint_limits: JointLimits | dict[str, float] | None = None) -> bool:
    limits = coerce_joint_limits(joint_limits)
    widths = limits.widths_array()
    return bool(np.all(np.abs(widths - FULL_CIRCLE) <= WRAP_SPAN_TOL))


def resolve_wrap_mode(
    wrap_angles: bool,
    joint_limits: JointLimits | dict[str, float] | None = None,
) -> bool:
    if not wrap_angles:
        return False
    return supports_wrapping(joint_limits)


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + PI) % (2.0 * PI) - PI


def normalize_angles(
    q: np.ndarray,
    wrap_angles: bool,
    joint_limits: JointLimits | dict[str, float] | None = None,
) -> np.ndarray:
    limits = coerce_joint_limits(joint_limits)
    mins = limits.mins_array()
    maxs = limits.maxs_array()

    if wrap_angles:
        widths = maxs - mins
        return np.asarray(((np.asarray(q, dtype=float) - mins) % widths) + mins, dtype=float)

    return np.clip(np.asarray(q, dtype=float), mins, maxs).astype(float)


def angle_delta(
    a: np.ndarray,
    b: np.ndarray,
    wrap_angles: bool,
    joint_limits: JointLimits | dict[str, float] | None = None,
) -> np.ndarray:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if wrap_angles:
        limits = coerce_joint_limits(joint_limits)
        widths = limits.widths_array()
        return np.asarray(((diff + widths / 2.0) % widths) - widths / 2.0, dtype=float)

    return diff


def angle_distance(
    a: np.ndarray,
    b: np.ndarray,
    wrap_angles: bool,
    joint_limits: JointLimits | dict[str, float] | None = None,
) -> float:
    return float(np.linalg.norm(angle_delta(a, b, wrap_angles, joint_limits=joint_limits)))


def angle_distance_batch(
    nodes_arr: np.ndarray,
    target: np.ndarray,
    wrap_angles: bool,
    joint_limits: JointLimits | dict[str, float] | None = None,
) -> np.ndarray:
    d = np.asarray(nodes_arr, dtype=float) - np.asarray(target, dtype=float)
    if wrap_angles:
        limits = coerce_joint_limits(joint_limits)
        widths = limits.widths_array()
        d = np.asarray(((d + widths / 2.0) % widths) - widths / 2.0, dtype=float)

    return np.linalg.norm(d, axis=1)


def line_rect_intersect(p1: tuple[float, float], p2: tuple[float, float], rect: dict[str, float]) -> bool:
    xmin, xmax = rect["xmin"], rect["xmax"]
    ymin, ymax = rect["ymin"], rect["ymax"]

    for p in (p1, p2):
        if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
            return True

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    p_ = [-dx, dx, -dy, dy]
    q_ = [p1[0] - xmin, xmax - p1[0], p1[1] - ymin, ymax - p1[1]]

    u1, u2 = 0.0, 1.0
    for i in range(4):
        if p_[i] == 0:
            if q_[i] < 0:
                return False
        else:
            t = q_[i] / p_[i]
            if p_[i] < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    return u1 <= u2


def check_collision_single(
    t1: float,
    t2: float,
    rect: dict[str, float],
    buffer: float = 0.05,
    l1: float = 1.0,
    l2: float = 1.0,
) -> bool:
    buffered_rect = {
        "xmin": rect["xmin"] - buffer,
        "xmax": rect["xmax"] + buffer,
        "ymin": rect["ymin"] - buffer,
        "ymax": rect["ymax"] + buffer,
    }

    x1, y1 = l1 * np.cos(t1), l1 * np.sin(t1)
    x2, y2 = x1 + l2 * np.cos(t1 + t2), y1 + l2 * np.sin(t1 + t2)

    return line_rect_intersect((0.0, 0.0), (float(x1), float(y1)), buffered_rect) or line_rect_intersect(
        (float(x1), float(y1)),
        (float(x2), float(y2)),
        buffered_rect,
    )


def build_cspace_grid(
    rect: dict[str, float],
    resolution: int = 120,
    buffer: float = 0.05,
    joint_limits: JointLimits | dict[str, float] | None = None,
    link_lengths: LinkLengths | dict[str, float] | None = None,
) -> list[list[int]]:
    limits = coerce_joint_limits(joint_limits)
    lengths = coerce_link_lengths(link_lengths)

    angles1 = np.linspace(limits.theta1_min, limits.theta1_max, resolution)
    angles2 = np.linspace(limits.theta2_min, limits.theta2_max, resolution)
    grid = np.zeros((resolution, resolution), dtype=np.uint8)

    for i, t1 in enumerate(angles1):
        for j, t2 in enumerate(angles2):
            grid[j, i] = (
                1
                if check_collision_single(
                    float(t1),
                    float(t2),
                    rect,
                    buffer=buffer,
                    l1=lengths.l1,
                    l2=lengths.l2,
                )
                else 0
            )

    return grid.tolist()


class RRTPlanner:
    def __init__(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        rect: dict[str, float],
        settings: PlannerSettings,
        wrap_angles: bool,
        joint_limits: JointLimits | dict[str, float] | None = None,
        link_lengths: LinkLengths | dict[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.rect = rect
        self.joint_limits = coerce_joint_limits(joint_limits)
        self.link_lengths = coerce_link_lengths(link_lengths)
        self.wrap_angles = resolve_wrap_mode(wrap_angles, joint_limits=self.joint_limits)

        self.start = normalize_angles(
            np.asarray(start, dtype=float),
            self.wrap_angles,
            joint_limits=self.joint_limits,
        )
        self.goal = normalize_angles(
            np.asarray(goal, dtype=float),
            self.wrap_angles,
            joint_limits=self.joint_limits,
        )

        self.tree: list[np.ndarray] = [self.start]
        self.parents: list[Optional[int]] = [None]

        self.py_rng = random.Random(settings.seed)
        self.np_rng = np.random.default_rng(settings.seed)

    def _sample(self) -> np.ndarray:
        if self.py_rng.random() < self.settings.goal_rate:
            return self.goal.copy()

        mins = self.joint_limits.mins_array()
        maxs = self.joint_limits.maxs_array()
        return self.np_rng.uniform(mins, maxs, size=2).astype(float)

    def _edge_in_collision(self, n1: np.ndarray, n2: np.ndarray) -> bool:
        diff = angle_delta(n2, n1, self.wrap_angles, joint_limits=self.joint_limits)
        for t in np.linspace(0.0, 1.0, self.settings.edge_check_steps):
            q = n1 + t * diff
            q = normalize_angles(q, self.wrap_angles, joint_limits=self.joint_limits)
            if check_collision_single(
                float(q[0]),
                float(q[1]),
                self.rect,
                buffer=self.settings.collision_buffer,
                l1=self.link_lengths.l1,
                l2=self.link_lengths.l2,
            ):
                return True
        return False

    def plan(self) -> PlanData:
        nodes = np.asarray(self.tree, dtype=float)
        started = time.perf_counter()

        for iteration in range(1, self.settings.max_iter + 1):
            rnd = self._sample()

            nearest_idx = int(
                np.argmin(
                    angle_distance_batch(
                        nodes,
                        rnd,
                        self.wrap_angles,
                        joint_limits=self.joint_limits,
                    )
                )
            )
            nearest = self.tree[nearest_idx]

            diff = angle_delta(rnd, nearest, self.wrap_angles, joint_limits=self.joint_limits)
            dist = float(np.linalg.norm(diff))
            if dist < 1e-9:
                continue

            if dist > self.settings.expand_dist:
                new_node = nearest + (diff / dist) * self.settings.expand_dist
            else:
                new_node = rnd.copy()

            new_node = normalize_angles(new_node, self.wrap_angles, joint_limits=self.joint_limits)

            if check_collision_single(
                float(new_node[0]),
                float(new_node[1]),
                self.rect,
                buffer=self.settings.collision_buffer,
                l1=self.link_lengths.l1,
                l2=self.link_lengths.l2,
            ):
                continue

            if self._edge_in_collision(nearest, new_node):
                continue

            self.tree.append(new_node)
            self.parents.append(nearest_idx)
            nodes = np.vstack([nodes, new_node])

            if (
                angle_distance(
                    new_node,
                    self.goal,
                    self.wrap_angles,
                    joint_limits=self.joint_limits,
                )
                < self.settings.expand_dist
            ):
                path: list[np.ndarray] = []
                current = len(self.tree) - 1
                while current is not None:
                    path.append(self.tree[current])
                    current = self.parents[current]

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return PlanData(
                    success=True,
                    tree=self.tree,
                    parents=self.parents,
                    path=list(reversed(path)),
                    iterations=iteration,
                    time_ms=elapsed_ms,
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PlanData(
            success=False,
            tree=self.tree,
            parents=self.parents,
            path=None,
            iterations=self.settings.max_iter,
            time_ms=elapsed_ms,
        )
