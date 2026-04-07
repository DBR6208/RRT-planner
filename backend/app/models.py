from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field


class ObstacleInput(BaseModel):
    x: float = 0.9
    y: float = 0.2
    size: float = Field(default=0.2, gt=0.0, le=2.0)


class JointLimitsInput(BaseModel):
    theta1_min: float = -math.pi
    theta1_max: float = math.pi
    theta2_min: float = -math.pi
    theta2_max: float = math.pi


class LinkLengthsInput(BaseModel):
    l1: float = Field(default=1.0, gt=0.0, le=5.0)
    l2: float = Field(default=1.0, gt=0.0, le=5.0)


class PlannerRequest(BaseModel):
    start: list[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)
    goal: list[float] = Field(default_factory=lambda: [math.pi / 2.0, 0.0], min_length=2, max_length=2)
    obstacle: ObstacleInput = Field(default_factory=ObstacleInput)
    joint_limits: JointLimitsInput = Field(default_factory=JointLimitsInput)
    link_lengths: LinkLengthsInput = Field(default_factory=LinkLengthsInput)

    expand_dist: float = Field(default=0.03, gt=0.0, le=1.0)
    goal_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    max_iter: int = Field(default=12000, ge=100, le=250000)
    edge_check_steps: int = Field(default=20, ge=2, le=500)
    collision_buffer: float = Field(default=0.05, ge=0.0, le=0.5)

    wrap_angles: bool = True

    include_cspace: bool = True
    cspace_resolution: int = Field(default=120, ge=40, le=1500)

    seed: Optional[int] = 42


class CspaceRequest(BaseModel):
    start: list[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)
    goal: list[float] = Field(default_factory=lambda: [math.pi / 2.0, 0.0], min_length=2, max_length=2)
    obstacle: ObstacleInput = Field(default_factory=ObstacleInput)
    joint_limits: JointLimitsInput = Field(default_factory=JointLimitsInput)
    link_lengths: LinkLengthsInput = Field(default_factory=LinkLengthsInput)

    collision_buffer: float = Field(default=0.05, ge=0.0, le=0.5)
    cspace_resolution: int = Field(default=120, ge=40, le=1500)


class PlanResponse(BaseModel):
    success: bool
    message: str

    wrap_angles: bool
    start: list[float]
    goal: list[float]

    obstacle: dict[str, float]
    joint_limits: dict[str, float]
    link_lengths: dict[str, float]

    tree: list[list[float]]
    parents: list[Optional[int]]
    path: Optional[list[list[float]]]

    iterations: int
    time_ms: float

    cspace: Optional[list[list[int]]] = None
    cspace_resolution: Optional[int] = None


class CspaceResponse(BaseModel):
    start: list[float]
    goal: list[float]
    obstacle: dict[str, float]
    joint_limits: dict[str, float]
    link_lengths: dict[str, float]
    collision_buffer: float

    cspace: list[list[int]]
    cspace_resolution: int

    start_in_collision: bool
    goal_in_collision: bool


class ExportRequest(PlannerRequest):
    fps: int = Field(default=24, ge=10, le=60)
    dpi: int = Field(default=110, ge=72, le=220)
    bitrate: int = Field(default=2200, ge=500, le=12000)
    samples_per_edge: int = Field(default=2, ge=1, le=20)
    max_frames: int = Field(default=160, ge=10, le=2000)
