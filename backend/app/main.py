from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .models import CspaceRequest, CspaceResponse, ExportRequest, PlanResponse, PlannerRequest
from .planner import (
    JointLimits,
    LinkLengths,
    PlannerSettings,
    RRTPlanner,
    build_cspace_grid,
    check_collision_single,
    coerce_joint_limits,
    coerce_link_lengths,
    normalize_angles,
)
from .video_export import generate_mp4_bytes


app = FastAPI(title="RRT Planner API", version="1.0.0")


def _origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_from_env(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def obstacle_to_rect(x: float, y: float, size: float) -> dict[str, float]:
    half = size / 2.0
    return {
        "xmin": x - half,
        "xmax": x + half,
        "ymin": y - half,
        "ymax": y + half,
    }


def _to_plain_nodes(nodes: list[np.ndarray]) -> list[list[float]]:
    return [[float(node[0]), float(node[1])] for node in nodes]


def _joint_limits_dict(limits: JointLimits) -> dict[str, float]:
    return {
        "theta1_min": float(limits.theta1_min),
        "theta1_max": float(limits.theta1_max),
        "theta2_min": float(limits.theta2_min),
        "theta2_max": float(limits.theta2_max),
    }


def _link_lengths_dict(lengths: LinkLengths) -> dict[str, float]:
    return {
        "l1": float(lengths.l1),
        "l2": float(lengths.l2),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "rrt-planner-api",
        "health": "/api/health",
        "plan": "/api/plan",
        "cspace": "/api/cspace",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/cspace", response_model=CspaceResponse)
def cspace(request: CspaceRequest) -> CspaceResponse:
    rect = obstacle_to_rect(request.obstacle.x, request.obstacle.y, request.obstacle.size)

    try:
        joint_limits = coerce_joint_limits(request.joint_limits.model_dump())
        link_lengths = coerce_link_lengths(request.link_lengths.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    start = normalize_angles(np.asarray(request.start, dtype=float), False, joint_limits=joint_limits)
    goal = normalize_angles(np.asarray(request.goal, dtype=float), False, joint_limits=joint_limits)

    return CspaceResponse(
        start=[float(start[0]), float(start[1])],
        goal=[float(goal[0]), float(goal[1])],
        obstacle=rect,
        joint_limits=_joint_limits_dict(joint_limits),
        link_lengths=_link_lengths_dict(link_lengths),
        collision_buffer=request.collision_buffer,
        cspace=build_cspace_grid(
            rect,
            resolution=request.cspace_resolution,
            buffer=request.collision_buffer,
            joint_limits=joint_limits,
            link_lengths=link_lengths,
        ),
        cspace_resolution=request.cspace_resolution,
        start_in_collision=check_collision_single(
            float(start[0]),
            float(start[1]),
            rect,
            buffer=request.collision_buffer,
            l1=link_lengths.l1,
            l2=link_lengths.l2,
        ),
        goal_in_collision=check_collision_single(
            float(goal[0]),
            float(goal[1]),
            rect,
            buffer=request.collision_buffer,
            l1=link_lengths.l1,
            l2=link_lengths.l2,
        ),
    )


@app.post("/api/plan", response_model=PlanResponse)
def plan(request: PlannerRequest) -> PlanResponse:
    rect = obstacle_to_rect(request.obstacle.x, request.obstacle.y, request.obstacle.size)

    try:
        joint_limits = coerce_joint_limits(request.joint_limits.model_dump())
        link_lengths = coerce_link_lengths(request.link_lengths.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    settings = PlannerSettings(
        expand_dist=request.expand_dist,
        goal_rate=request.goal_rate,
        max_iter=request.max_iter,
        edge_check_steps=request.edge_check_steps,
        collision_buffer=request.collision_buffer,
        seed=request.seed,
    )

    planner = RRTPlanner(
        start=request.start,
        goal=request.goal,
        rect=rect,
        settings=settings,
        wrap_angles=request.wrap_angles,
        joint_limits=joint_limits,
        link_lengths=link_lengths,
    )
    result = planner.plan()

    cspace = None
    cspace_resolution = None
    if request.include_cspace:
        cspace = build_cspace_grid(
            rect,
            resolution=request.cspace_resolution,
            buffer=request.collision_buffer,
            joint_limits=joint_limits,
            link_lengths=link_lengths,
        )
        cspace_resolution = request.cspace_resolution

    message = "Path found" if result.success else "No path found with current parameters"

    return PlanResponse(
        success=result.success,
        message=message,
        wrap_angles=planner.wrap_angles,
        start=[float(planner.start[0]), float(planner.start[1])],
        goal=[float(planner.goal[0]), float(planner.goal[1])],
        obstacle=rect,
        joint_limits=_joint_limits_dict(joint_limits),
        link_lengths=_link_lengths_dict(link_lengths),
        tree=_to_plain_nodes(result.tree),
        parents=result.parents,
        path=_to_plain_nodes(result.path) if result.path is not None else None,
        iterations=result.iterations,
        time_ms=float(result.time_ms),
        cspace=cspace,
        cspace_resolution=cspace_resolution,
    )


@app.post("/api/export-mp4")
def export_mp4(request: ExportRequest) -> Response:
    try:
        mp4_bytes = generate_mp4_bytes(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected MP4 export error: {exc}") from exc

    return Response(
        content=mp4_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="rrt_animation.mp4"'},
    )
