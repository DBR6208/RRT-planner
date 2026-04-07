from __future__ import annotations

import numpy as np

from app.planner import JointLimits, PlannerSettings, RRTPlanner, angle_delta, normalize_angles, resolve_wrap_mode


def test_angle_delta_wrap_toggle_changes_distance() -> None:
    a = np.array([-np.pi + 0.1, -np.pi + 0.05], dtype=float)
    b = np.array([np.pi - 0.1, np.pi - 0.05], dtype=float)

    wrapped = np.linalg.norm(angle_delta(a, b, wrap_angles=True))
    non_wrapped = np.linalg.norm(angle_delta(a, b, wrap_angles=False))

    assert wrapped < non_wrapped


def test_rrt_plans_in_both_wrap_modes_without_collision_obstacle() -> None:
    no_collision_rect = {
        "xmin": 3.0,
        "xmax": 3.4,
        "ymin": 3.0,
        "ymax": 3.4,
    }

    settings = PlannerSettings(
        expand_dist=0.2,
        goal_rate=0.15,
        max_iter=4000,
        edge_check_steps=20,
        collision_buffer=0.0,
        seed=7,
    )

    start = [np.pi - 0.2, 0.0]
    goal = [-np.pi + 0.2, 0.0]

    wrapped_result = RRTPlanner(start, goal, no_collision_rect, settings, wrap_angles=True).plan()
    non_wrapped_result = RRTPlanner(start, goal, no_collision_rect, settings, wrap_angles=False).plan()

    assert wrapped_result.success is True
    assert wrapped_result.path is not None
    assert len(wrapped_result.path) > 1

    assert non_wrapped_result.success is True
    assert non_wrapped_result.path is not None
    assert len(non_wrapped_result.path) > 1


def test_normalize_respects_custom_joint_limits() -> None:
    limits = JointLimits(theta1_min=-1.0, theta1_max=1.0, theta2_min=-0.5, theta2_max=0.5)
    q = np.array([2.2, -2.0], dtype=float)

    non_wrapped = normalize_angles(q, wrap_angles=False, joint_limits=limits)
    wrapped = normalize_angles(q, wrap_angles=True, joint_limits=limits)

    assert np.allclose(non_wrapped, np.array([1.0, -0.5]))
    assert limits.theta1_min <= wrapped[0] <= limits.theta1_max
    assert limits.theta2_min <= wrapped[1] <= limits.theta2_max


def test_wrap_mode_is_disabled_for_bounded_joint_limits() -> None:
    limits = JointLimits(theta1_min=0.0, theta1_max=np.pi, theta2_min=-np.pi, theta2_max=np.pi)

    assert resolve_wrap_mode(True, limits) is False

    settings = PlannerSettings(seed=1, max_iter=2000, expand_dist=0.12, goal_rate=0.2, collision_buffer=0.0)
    no_collision_rect = {
        "xmin": 3.0,
        "xmax": 3.2,
        "ymin": 3.0,
        "ymax": 3.2,
    }

    planner = RRTPlanner(
        start=[0.2, 0.0],
        goal=[2.8, 0.0],
        rect=no_collision_rect,
        settings=settings,
        wrap_angles=True,
        joint_limits=limits,
    )

    assert planner.wrap_angles is False
    result = planner.plan()
    assert result.success is True
    assert result.path is not None
    theta1_values = np.asarray([node[0] for node in result.path], dtype=float)
    assert np.all(theta1_values >= limits.theta1_min - 1e-9)
    assert np.all(theta1_values <= limits.theta1_max + 1e-9)
