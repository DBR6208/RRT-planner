import { useEffect, useMemo, useRef, useState } from "react";
import { requestCspace, requestExportMp4, requestPlan } from "./api";
import { angleToPixel, getArmCoords, shouldSkipWrappedSegment, worldToPixel } from "./math";

const DEFAULT_START_Q = [0.0, 0.0];
const DEFAULT_GOAL_Q = [Math.PI / 2.0, 0.0];
const DEFAULT_JOINT_LIMITS = {
  theta1_min: -Math.PI,
  theta1_max: Math.PI,
  theta2_min: -Math.PI,
  theta2_max: Math.PI,
};
const DEFAULT_LINK_LENGTHS = { l1: 1.0, l2: 1.0 };

const PLOT_SIZE = 440;
const COLLISION_BUFFER = 0.05;
const DEFAULT_CSPACE_RESOLUTION = 720;
const MIN_CSPACE_RESOLUTION = 120;
const MAX_CSPACE_RESOLUTION = 1200;

const TORUS_MAJOR_RADIUS = 1.35;
const TORUS_MINOR_RADIUS = 0.55;
const TORUS_ROT_X = 0.92;
const TORUS_ROT_Y = -0.88;
const TORUS_CAMERA_Z = 5.2;
const TORUS_SCALE = 82;
const TAU = 2 * Math.PI;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sanitizeJointLimits(raw) {
  const minGap = 0.05;
  const theta1_min = Number.isFinite(raw.theta1_min) ? raw.theta1_min : DEFAULT_JOINT_LIMITS.theta1_min;
  let theta1_max = Number.isFinite(raw.theta1_max) ? raw.theta1_max : DEFAULT_JOINT_LIMITS.theta1_max;
  const theta2_min = Number.isFinite(raw.theta2_min) ? raw.theta2_min : DEFAULT_JOINT_LIMITS.theta2_min;
  let theta2_max = Number.isFinite(raw.theta2_max) ? raw.theta2_max : DEFAULT_JOINT_LIMITS.theta2_max;

  if (theta1_max <= theta1_min + minGap) theta1_max = theta1_min + minGap;
  if (theta2_max <= theta2_min + minGap) theta2_max = theta2_min + minGap;

  return { theta1_min, theta1_max, theta2_min, theta2_max };
}

function sanitizeLinkLengths(raw) {
  const l1 = clamp(Number.isFinite(raw.l1) ? raw.l1 : DEFAULT_LINK_LENGTHS.l1, 0.1, 5.0);
  const l2 = clamp(Number.isFinite(raw.l2) ? raw.l2 : DEFAULT_LINK_LENGTHS.l2, 0.1, 5.0);
  return { l1, l2 };
}

function clampConfigToJointLimits(q, jointLimits) {
  return [
    clamp(q[0], jointLimits.theta1_min, jointLimits.theta1_max),
    clamp(q[1], jointLimits.theta2_min, jointLimits.theta2_max),
  ];
}

function formatTick(value) {
  return Number(value).toFixed(2);
}

function toSvgPoints(xs, ys, mapper) {
  return xs
    .map((x, idx) => {
      const [px, py] = mapper(x, ys[idx]);
      return `${px},${py}`;
    })
    .join(" ");
}

function toTorusAngles(theta1, theta2, jointLimits) {
  const theta1Span = Math.max(1e-9, jointLimits.theta1_max - jointLimits.theta1_min);
  const theta2Span = Math.max(1e-9, jointLimits.theta2_max - jointLimits.theta2_min);
  const u = ((theta1 - jointLimits.theta1_min) / theta1Span) * TAU - Math.PI;
  const v = ((theta2 - jointLimits.theta2_min) / theta2Span) * TAU - Math.PI;
  return [u, v];
}

function supportsWrapForLimits(jointLimits) {
  const eps = 1e-6;
  const s1 = Math.abs((jointLimits.theta1_max - jointLimits.theta1_min) - TAU) <= eps;
  const s2 = Math.abs((jointLimits.theta2_max - jointLimits.theta2_min) - TAU) <= eps;
  return s1 && s2;
}

function torusPoint(theta1, theta2) {
  const ring = TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * Math.cos(theta2);
  return [ring * Math.cos(theta1), ring * Math.sin(theta1), TORUS_MINOR_RADIUS * Math.sin(theta2)];
}

function rotatePoint3D(point) {
  const [x, y, z] = point;
  const cosY = Math.cos(TORUS_ROT_Y);
  const sinY = Math.sin(TORUS_ROT_Y);
  const xY = x * cosY + z * sinY;
  const zY = -x * sinY + z * cosY;
  const cosX = Math.cos(TORUS_ROT_X);
  const sinX = Math.sin(TORUS_ROT_X);
  return [xY, y * cosX - zY * sinX, y * sinX + zY * cosX];
}

function projectPoint3D(point, sizePx) {
  const perspective = TORUS_CAMERA_Z / (TORUS_CAMERA_Z - point[2]);
  return [
    sizePx / 2 + point[0] * TORUS_SCALE * perspective,
    sizePx / 2 - point[1] * TORUS_SCALE * perspective,
    point[2],
    perspective,
  ];
}

function AxisAnnotatedPlot({ title, xLabel, yLabel, xTicks, yTicks, children, ariaLabel }) {
  return (
    <div className="plot-card">
      <h3>{title}</h3>
      <div className="axis-plot-shell" role="group" aria-label={ariaLabel}>
        <div className="axis-plot-core">{children}</div>
        <div className="axis-label-strong axis-label-x">{xLabel}</div>
        <div className="axis-label-strong axis-label-y">{yLabel}</div>
        <div className="axis-ticks-x">{xTicks.map((tick) => <span key={`x-${tick}`}>{tick}</span>)}</div>
        <div className="axis-ticks-y">{yTicks.map((tick) => <span key={`y-${tick}`}>{tick}</span>)}</div>
      </div>
    </div>
  );
}

function WorkspacePlot({ obstacle, currentQ, startQ, goalQ, linkLengths }) {
  const workspaceLimit = Math.max(1.0, linkLengths.l1 + linkLengths.l2 + 0.2);
  const startArm = getArmCoords(startQ[0], startQ[1], linkLengths.l1, linkLengths.l2);
  const goalArm = getArmCoords(goalQ[0], goalQ[1], linkLengths.l1, linkLengths.l2);
  const currArm = getArmCoords(currentQ[0], currentQ[1], linkLengths.l1, linkLengths.l2);
  const mapPoint = (x, y) => worldToPixel(x, y, -workspaceLimit, workspaceLimit, PLOT_SIZE);

  const [leftTopX, leftTopY] = mapPoint(obstacle.x - obstacle.size / 2.0, obstacle.y + obstacle.size / 2.0);
  const obstacleSide = (obstacle.size / (2.0 * workspaceLimit)) * PLOT_SIZE;
  const [xMinPx] = mapPoint(-workspaceLimit, 0);
  const [xZeroPx] = mapPoint(0, 0);
  const [xMaxPx] = mapPoint(workspaceLimit, 0);
  const [, yMinPx] = mapPoint(0, -workspaceLimit);
  const [, yZeroPx] = mapPoint(0, 0);
  const [, yMaxPx] = mapPoint(0, workspaceLimit);

  return (
    <AxisAnnotatedPlot
      title="Workspace"
      xLabel="x"
      yLabel="y"
      xTicks={[`-${workspaceLimit.toFixed(1)}`, "0", workspaceLimit.toFixed(1)]}
      yTicks={[workspaceLimit.toFixed(1), "0", `-${workspaceLimit.toFixed(1)}`]}
      ariaLabel="Workspace with x and y annotations"
    >
      <svg width={PLOT_SIZE} height={PLOT_SIZE} viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`} className="plot-svg" role="img" aria-label="Robot workspace">
        <rect x="0" y="0" width={PLOT_SIZE} height={PLOT_SIZE} className="workspace-bg" />
        <line x1="0" y1={PLOT_SIZE / 2} x2={PLOT_SIZE} y2={PLOT_SIZE / 2} className="axis-line" />
        <line x1={PLOT_SIZE / 2} y1="0" x2={PLOT_SIZE / 2} y2={PLOT_SIZE} className="axis-line" />
        <line x1={xMinPx} y1={yZeroPx - 4} x2={xMinPx} y2={yZeroPx + 4} className="axis-tick" />
        <line x1={xZeroPx} y1={yZeroPx - 4} x2={xZeroPx} y2={yZeroPx + 4} className="axis-tick" />
        <line x1={xMaxPx} y1={yZeroPx - 4} x2={xMaxPx} y2={yZeroPx + 4} className="axis-tick" />
        <line x1={xZeroPx - 4} y1={yMinPx} x2={xZeroPx + 4} y2={yMinPx} className="axis-tick" />
        <line x1={xZeroPx - 4} y1={yZeroPx} x2={xZeroPx + 4} y2={yZeroPx} className="axis-tick" />
        <line x1={xZeroPx - 4} y1={yMaxPx} x2={xZeroPx + 4} y2={yMaxPx} className="axis-tick" />
        <rect x={leftTopX} y={leftTopY} width={obstacleSide} height={obstacleSide} className="obstacle" />
        <polyline points={toSvgPoints(startArm.x, startArm.y, mapPoint)} className="arm start-arm" />
        <polyline points={toSvgPoints(goalArm.x, goalArm.y, mapPoint)} className="arm goal-arm" />
        <polyline points={toSvgPoints(currArm.x, currArm.y, mapPoint)} className="arm current-arm" />
        {currArm.x.map((x, idx) => {
          const [px, py] = mapPoint(x, currArm.y[idx]);
          return <circle key={`joint-${idx}`} cx={px} cy={py} r={idx === 2 ? 5 : 4} className="joint" />;
        })}
      </svg>
    </AxisAnnotatedPlot>
  );
}

function CspacePlot({ cspace, tree, parents, path, wrapAngles, currentQ, startQ, goalQ, jointLimits }) {
  const canvasRef = useRef(null);
  const theta1Center = (jointLimits.theta1_min + jointLimits.theta1_max) / 2.0;
  const theta2Center = (jointLimits.theta2_min + jointLimits.theta2_max) / 2.0;
  const hasCspace = Boolean(cspace && cspace.length > 0 && cspace[0]?.length > 0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = ctx.createImageData(PLOT_SIZE, PLOT_SIZE);
    const width = cspace?.[0]?.length ?? 0;
    const height = cspace?.length ?? 0;

    for (let y = 0; y < PLOT_SIZE; y += 1) {
      const gy = height > 0 ? Math.min(height - 1, Math.floor(((PLOT_SIZE - 1 - y) / PLOT_SIZE) * height)) : 0;
      for (let x = 0; x < PLOT_SIZE; x += 1) {
        const gx = width > 0 ? Math.min(width - 1, Math.floor((x / PLOT_SIZE) * width)) : 0;
        const blocked = height > 0 && width > 0 ? cspace[gy][gx] === 1 : false;
        const idx = (y * PLOT_SIZE + x) * 4;
        const shade = blocked ? 58 : 238;

        image.data[idx] = shade;
        image.data[idx + 1] = blocked ? 66 : 245;
        image.data[idx + 2] = blocked ? 79 : 255;
        image.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(image, 0, 0);
    ctx.strokeStyle = "rgba(16, 31, 52, 0.4)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, PLOT_SIZE - 1, PLOT_SIZE - 1);
  }, [cspace]);

  const treeSegments = useMemo(() => {
    if (!tree || !parents || tree.length <= 1) return [];

    const segments = [];
    for (let i = 1; i < tree.length; i += 1) {
      const parent = parents[i];
      if (parent === null || parent === undefined) continue;
      const p1 = tree[parent];
      const p2 = tree[i];
      if (!p1 || !p2 || shouldSkipWrappedSegment(p1, p2, wrapAngles, jointLimits)) continue;

      const [x1, y1] = angleToPixel(p1[0], p1[1], PLOT_SIZE, jointLimits);
      const [x2, y2] = angleToPixel(p2[0], p2[1], PLOT_SIZE, jointLimits);
      segments.push({ x1, y1, x2, y2 });
    }
    return segments;
  }, [tree, parents, wrapAngles, jointLimits]);

  const pathSegments = useMemo(() => {
    if (!path || path.length < 2) return [];

    const segments = [];
    for (let i = 1; i < path.length; i += 1) {
      const p1 = path[i - 1];
      const p2 = path[i];
      if (shouldSkipWrappedSegment(p1, p2, wrapAngles, jointLimits)) continue;
      const [x1, y1] = angleToPixel(p1[0], p1[1], PLOT_SIZE, jointLimits);
      const [x2, y2] = angleToPixel(p2[0], p2[1], PLOT_SIZE, jointLimits);
      segments.push({ x1, y1, x2, y2 });
    }
    return segments;
  }, [path, wrapAngles, jointLimits]);

  const [sx, sy] = angleToPixel(startQ[0], startQ[1], PLOT_SIZE, jointLimits);
  const [gx, gy] = angleToPixel(goalQ[0], goalQ[1], PLOT_SIZE, jointLimits);
  const [cx, cy] = angleToPixel(currentQ[0], currentQ[1], PLOT_SIZE, jointLimits);

  const [xMinPx] = angleToPixel(jointLimits.theta1_min, theta2Center, PLOT_SIZE, jointLimits);
  const [xMidPx] = angleToPixel(theta1Center, theta2Center, PLOT_SIZE, jointLimits);
  const [xMaxPx] = angleToPixel(jointLimits.theta1_max, theta2Center, PLOT_SIZE, jointLimits);
  const [, yMinPx] = angleToPixel(theta1Center, jointLimits.theta2_min, PLOT_SIZE, jointLimits);
  const [, yMidPx] = angleToPixel(theta1Center, theta2Center, PLOT_SIZE, jointLimits);
  const [, yMaxPx] = angleToPixel(theta1Center, jointLimits.theta2_max, PLOT_SIZE, jointLimits);

  return (
    <AxisAnnotatedPlot
      title="C-Space / No-Go Zone"
      xLabel="theta1"
      yLabel="theta2"
      xTicks={[formatTick(jointLimits.theta1_min), formatTick(theta1Center), formatTick(jointLimits.theta1_max)]}
      yTicks={[formatTick(jointLimits.theta2_max), formatTick(theta2Center), formatTick(jointLimits.theta2_min)]}
      ariaLabel="C-space with theta1 and theta2 annotations"
    >
      <div className="stacked-plot plot-core-fill">
        <canvas ref={canvasRef} width={PLOT_SIZE} height={PLOT_SIZE} />
        <svg width={PLOT_SIZE} height={PLOT_SIZE} viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`}>
          {treeSegments.map((segment, index) => (
            <line
              key={`tree-${index}`}
              x1={segment.x1}
              y1={segment.y1}
              x2={segment.x2}
              y2={segment.y2}
              className="tree-edge"
            />
          ))}
          {pathSegments.map((segment, index) => (
            <line
              key={`path-${index}`}
              x1={segment.x1}
              y1={segment.y1}
              x2={segment.x2}
              y2={segment.y2}
              className="path-edge"
            />
          ))}

          <line x1="0" y1={yMidPx} x2={PLOT_SIZE} y2={yMidPx} className="axis-line" />
          <line x1={xMidPx} y1="0" x2={xMidPx} y2={PLOT_SIZE} className="axis-line" />

          <line x1={xMinPx} y1={yMidPx - 4} x2={xMinPx} y2={yMidPx + 4} className="axis-tick" />
          <line x1={xMidPx} y1={yMidPx - 4} x2={xMidPx} y2={yMidPx + 4} className="axis-tick" />
          <line x1={xMaxPx} y1={yMidPx - 4} x2={xMaxPx} y2={yMidPx + 4} className="axis-tick" />

          <line x1={xMidPx - 4} y1={yMinPx} x2={xMidPx + 4} y2={yMinPx} className="axis-tick" />
          <line x1={xMidPx - 4} y1={yMidPx} x2={xMidPx + 4} y2={yMidPx} className="axis-tick" />
          <line x1={xMidPx - 4} y1={yMaxPx} x2={xMidPx + 4} y2={yMaxPx} className="axis-tick" />

          <circle cx={sx} cy={sy} r="7" className="dot start-dot" />
          <circle cx={gx} cy={gy} r="7" className="dot goal-dot" />
          <circle cx={cx} cy={cy} r="6" className="dot current-dot" />
          {!hasCspace && (
            <text x={PLOT_SIZE / 2} y={PLOT_SIZE / 2} textAnchor="middle" className="plot-hint">
              Compute no-go zone
            </text>
          )}
        </svg>
      </div>
    </AxisAnnotatedPlot>
  );
}

function TorusPlot({ cspace, path, startQ, goalQ, currentQ, jointLimits }) {
  const canvasRef = useRef(null);
  const hasCspace = Boolean(cspace && cspace.length > 0 && cspace[0]?.length > 0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, PLOT_SIZE, PLOT_SIZE);
    ctx.fillStyle = "#f8fcff";
    ctx.fillRect(0, 0, PLOT_SIZE, PLOT_SIZE);

    const width = cspace?.[0]?.length ?? 0;
    const height = cspace?.length ?? 0;
    const hasGrid = width > 0 && height > 0;
    const surfaceU = 72;
    const surfaceV = 48;
    const lightDir = [0.28, 0.38, 0.88];
    const clampIndex = (v, max) => Math.max(0, Math.min(max, v));

    function isBlockedAt(u, v) {
      if (width <= 0 || height <= 0) return false;
      const fx = ((u + Math.PI) / TAU) * (width - 1);
      const fy = ((v + Math.PI) / TAU) * (height - 1);
      const ix = clampIndex(Math.floor(fx), width - 1);
      const iy = clampIndex(Math.floor(fy), height - 1);
      return cspace[iy][ix] === 1;
    }

    const quads = [];
    for (let u = 0; u < surfaceU; u += 1) {
      const u0 = -Math.PI + (u / surfaceU) * TAU;
      const u1 = -Math.PI + ((u + 1) / surfaceU) * TAU;
      for (let v = 0; v < surfaceV; v += 1) {
        const v0 = -Math.PI + (v / surfaceV) * TAU;
        const v1 = -Math.PI + ((v + 1) / surfaceV) * TAU;
        const cornersRot = [
          rotatePoint3D(torusPoint(u0, v0)),
          rotatePoint3D(torusPoint(u1, v0)),
          rotatePoint3D(torusPoint(u1, v1)),
          rotatePoint3D(torusPoint(u0, v1)),
        ];
        const cornersProjected = cornersRot.map((p) => projectPoint3D(p, PLOT_SIZE));
        const avgDepth = (cornersRot[0][2] + cornersRot[1][2] + cornersRot[2][2] + cornersRot[3][2]) / 4;

        const uc = -Math.PI + ((u + 0.5) / surfaceU) * TAU;
        const vc = -Math.PI + ((v + 0.5) / surfaceV) * TAU;
        const n = rotatePoint3D([Math.cos(uc) * Math.cos(vc), Math.sin(uc) * Math.cos(vc), Math.sin(vc)]);
        const dot = Math.max(0, n[0] * lightDir[0] + n[1] * lightDir[1] + n[2] * lightDir[2]);
        const intensity = 0.26 + 0.74 * dot;
        const blocked = isBlockedAt(uc, vc);

        let fill;
        if (!hasGrid) {
          const shade = Math.round(120 + 60 * intensity);
          fill = `rgba(${shade}, ${shade + 16}, ${shade + 28}, 0.55)`;
        } else if (blocked) {
          fill = "rgba(178, 24, 24, 0.7)";
        } else {
          const r = Math.round(38 + 82 * intensity);
          const g = Math.round(92 + 102 * intensity);
          const b = Math.round(132 + 96 * intensity);
          fill = `rgba(${r}, ${g}, ${b}, 0.7)`;
        }

        quads.push({ points: cornersProjected, depth: avgDepth, fill });
      }
    }

    quads.sort((a, b) => a.depth - b.depth);
    for (const quad of quads) {
      ctx.fillStyle = quad.fill;
      ctx.beginPath();
      ctx.moveTo(quad.points[0][0], quad.points[0][1]);
      ctx.lineTo(quad.points[1][0], quad.points[1][1]);
      ctx.lineTo(quad.points[2][0], quad.points[2][1]);
      ctx.lineTo(quad.points[3][0], quad.points[3][1]);
      ctx.closePath();
      ctx.fill();
    }

    if (path && path.length > 1) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 5;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      path.forEach((q, idx) => {
        const [u, v] = toTorusAngles(q[0], q[1], jointLimits);
        const projected = projectPoint3D(rotatePoint3D(torusPoint(u, v)), PLOT_SIZE);
        if (idx === 0) ctx.moveTo(projected[0], projected[1]);
        else ctx.lineTo(projected[0], projected[1]);
      });
      ctx.stroke();

      ctx.strokeStyle = "#f18f01";
      ctx.lineWidth = 3;
      ctx.beginPath();
      path.forEach((q, idx) => {
        const [u, v] = toTorusAngles(q[0], q[1], jointLimits);
        const projected = projectPoint3D(rotatePoint3D(torusPoint(u, v)), PLOT_SIZE);
        if (idx === 0) ctx.moveTo(projected[0], projected[1]);
        else ctx.lineTo(projected[0], projected[1]);
      });
      ctx.stroke();
    }

    const [startU, startV] = toTorusAngles(startQ[0], startQ[1], jointLimits);
    const [goalU, goalV] = toTorusAngles(goalQ[0], goalQ[1], jointLimits);
    const [currU, currV] = toTorusAngles(currentQ[0], currentQ[1], jointLimits);
    const startProjected = projectPoint3D(rotatePoint3D(torusPoint(startU, startV)), PLOT_SIZE);
    const goalProjected = projectPoint3D(rotatePoint3D(torusPoint(goalU, goalV)), PLOT_SIZE);
    const currentProjected = projectPoint3D(rotatePoint3D(torusPoint(currU, currV)), PLOT_SIZE);

    function drawMarker(projected, fillColor, radius) {
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.arc(projected[0], projected[1], radius, 0, TAU);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    drawMarker(startProjected, "#1b9e77", 6);
    drawMarker(goalProjected, "#1d5ac7", 6);
    drawMarker(currentProjected, "#111111", 5.5);
  }, [cspace, path, startQ, goalQ, currentQ, jointLimits]);

  return (
    <div className="plot-card">
      <h3>Torus C-Space (3D)</h3>
      <div className="stacked-plot torus-plot-box">
        <canvas ref={canvasRef} width={PLOT_SIZE} height={PLOT_SIZE} />
      </div>
      <p className="panel-note torus-note">
        {hasCspace
          ? "Red surface: no-go zone, orange line: path, black dot: current state."
          : "Compute no-go zone to display blocked regions on the torus."}
      </p>
    </div>
  );
}

function StartGoalPanel({
  startQ,
  goalQ,
  jointLimits,
  linkLengths,
  onStartChange,
  onGoalChange,
  onJointLimitChange,
  onLinkLengthChange,
}) {
  return (
    <aside className="plot-card start-goal-panel">
      <details className="settings-group start-goal-group" open>
        <summary>Start & Goal State</summary>
        <div className="settings-body">
          <p className="panel-note">Set the joint angles before computing the no-go zone and running RRT.</p>
          <div className="control-grid start-goal-grid">
            <label>
              Start theta1 (rad)
              <input type="number" step="0.05" value={startQ[0]} onChange={(e) => onStartChange(0, e.target.value)} />
            </label>
            <label>
              Start theta2 (rad)
              <input type="number" step="0.05" value={startQ[1]} onChange={(e) => onStartChange(1, e.target.value)} />
            </label>
            <label>
              Goal theta1 (rad)
              <input type="number" step="0.05" value={goalQ[0]} onChange={(e) => onGoalChange(0, e.target.value)} />
            </label>
            <label>
              Goal theta2 (rad)
              <input type="number" step="0.05" value={goalQ[1]} onChange={(e) => onGoalChange(1, e.target.value)} />
            </label>
          </div>
        </div>
      </details>

      <details className="settings-group" open>
        <summary>Joint Limits</summary>
        <div className="settings-body">
          <p className="panel-note">Planner sampling and C-space boundaries use these limits.</p>
          <div className="control-grid">
            <label>
              theta1 min (rad)
              <input type="number" step="0.05" value={jointLimits.theta1_min} onChange={(e) => onJointLimitChange("theta1_min", e.target.value)} />
            </label>
            <label>
              theta1 max (rad)
              <input type="number" step="0.05" value={jointLimits.theta1_max} onChange={(e) => onJointLimitChange("theta1_max", e.target.value)} />
            </label>
            <label>
              theta2 min (rad)
              <input type="number" step="0.05" value={jointLimits.theta2_min} onChange={(e) => onJointLimitChange("theta2_min", e.target.value)} />
            </label>
            <label>
              theta2 max (rad)
              <input type="number" step="0.05" value={jointLimits.theta2_max} onChange={(e) => onJointLimitChange("theta2_max", e.target.value)} />
            </label>
          </div>
        </div>
      </details>

      <details className="settings-group" open>
        <summary>Link Lengths</summary>
        <div className="settings-body">
          <p className="panel-note">Collision checks and workspace figure use these link lengths.</p>
          <div className="control-grid">
            <label>
              Link 1 length
              <input type="number" min="0.1" max="5" step="0.05" value={linkLengths.l1} onChange={(e) => onLinkLengthChange("l1", e.target.value)} />
            </label>
            <label>
              Link 2 length
              <input type="number" min="0.1" max="5" step="0.05" value={linkLengths.l2} onChange={(e) => onLinkLengthChange("l2", e.target.value)} />
            </label>
          </div>
        </div>
      </details>
    </aside>
  );
}

function App() {
  const [startQ, setStartQ] = useState(DEFAULT_START_Q);
  const [goalQ, setGoalQ] = useState(DEFAULT_GOAL_Q);
  const [jointLimits, setJointLimits] = useState(DEFAULT_JOINT_LIMITS);
  const [linkLengths, setLinkLengths] = useState(DEFAULT_LINK_LENGTHS);
  const [obstacle, setObstacle] = useState({ x: 0.9, y: 0.2, size: 0.2 });

  const [expandDist, setExpandDist] = useState(0.03);
  const [goalRate, setGoalRate] = useState(0.1);
  const [maxIter, setMaxIter] = useState(12000);
  const [edgeCheckSteps, setEdgeCheckSteps] = useState(20);
  const [cspaceResolution, setCspaceResolution] = useState(DEFAULT_CSPACE_RESOLUTION);

  const [wrapAngles, setWrapAngles] = useState(true);

  const [cspaceData, setCspaceData] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isComputingNoGo, setIsComputingNoGo] = useState(false);
  const [isExportingMp4, setIsExportingMp4] = useState(false);

  const [isAnimating, setIsAnimating] = useState(false);
  const [animIndex, setAnimIndex] = useState(0);

  const path = result?.path ?? null;
  const canWrap = supportsWrapForLimits(jointLimits);
  const effectiveWrapAngles = wrapAngles && canWrap;
  const activeCspace = cspaceData?.cspace ?? result?.cspace ?? null;
  const activeJointLimits = cspaceData?.joint_limits ?? result?.joint_limits ?? jointLimits;
  const currentQ = useMemo(() => {
    if (!path || path.length === 0) return startQ;
    if (isAnimating) return path[Math.min(animIndex, path.length - 1)];
    return path[path.length - 1];
  }, [path, isAnimating, animIndex, startQ]);

  useEffect(() => {
    if (!isAnimating || !path || path.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setAnimIndex((prev) => {
        const next = prev + 1;
        if (next >= path.length) {
          setIsAnimating(false);
          return path.length - 1;
        }
        return next;
      });
    }, 60);
    return () => window.clearInterval(timer);
  }, [isAnimating, path]);

  useEffect(() => {
    setStartQ((prev) => clampConfigToJointLimits(prev, jointLimits));
    setGoalQ((prev) => clampConfigToJointLimits(prev, jointLimits));
  }, [jointLimits]);

  useEffect(() => {
    if (!canWrap && wrapAngles) {
      setWrapAngles(false);
    }
  }, [canWrap, wrapAngles]);

  function parseNumber(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function parseIntInRange(value, fallback, min, max) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    const rounded = Math.round(n);
    return Math.max(min, Math.min(max, rounded));
  }

  function invalidateComputedData({ clearPlanner = true, clearNoGo = true } = {}) {
    setError("");
    setIsAnimating(false);
    setAnimIndex(0);
    if (clearPlanner) setResult(null);
    if (clearNoGo) setCspaceData(null);
  }

  function updateStartQ(index, value) {
    invalidateComputedData();
    setStartQ((prev) => {
      const next = [...prev];
      next[index] = parseNumber(value, prev[index]);
      return clampConfigToJointLimits(next, jointLimits);
    });
  }

  function updateGoalQ(index, value) {
    invalidateComputedData();
    setGoalQ((prev) => {
      const next = [...prev];
      next[index] = parseNumber(value, prev[index]);
      return clampConfigToJointLimits(next, jointLimits);
    });
  }

  function updateJointLimitField(field, value) {
    invalidateComputedData();
    setJointLimits((prev) => sanitizeJointLimits({ ...prev, [field]: parseNumber(value, prev[field]) }));
  }

  function updateLinkLengthField(field, value) {
    invalidateComputedData();
    setLinkLengths((prev) => sanitizeLinkLengths({ ...prev, [field]: parseNumber(value, prev[field]) }));
  }

  function updateObstacleField(field, value) {
    invalidateComputedData();
    setObstacle((prev) => ({ ...prev, [field]: parseNumber(value, prev[field]) }));
  }

  async function handleComputeNoGoZone() {
    setError("");
    setIsComputingNoGo(true);
    setIsAnimating(false);
    setAnimIndex(0);
    setResult(null);
    try {
      const data = await requestCspace({
        start: startQ,
        goal: goalQ,
        obstacle,
        joint_limits: jointLimits,
        link_lengths: linkLengths,
        collision_buffer: COLLISION_BUFFER,
        cspace_resolution: cspaceResolution,
      });
      setCspaceData(data);
      setStartQ(data.start);
      setGoalQ(data.goal);
    } catch (runError) {
      setCspaceData(null);
      setError(runError instanceof Error ? runError.message : "Unknown C-space error");
    } finally {
      setIsComputingNoGo(false);
    }
  }

  async function handleRunPlanner() {
    if (!cspaceData?.cspace) {
      setError("Compute the no-go zone before running the planner.");
      return;
    }
    if (cspaceData.start_in_collision || cspaceData.goal_in_collision) {
      setError("Start or goal is in collision. Update the configuration and recompute the no-go zone.");
      return;
    }

    setError("");
    setIsLoading(true);
    setIsAnimating(false);
    setAnimIndex(0);
    try {
      const data = await requestPlan({
        start: startQ,
        goal: goalQ,
        obstacle,
        joint_limits: jointLimits,
        link_lengths: linkLengths,
        expand_dist: expandDist,
        goal_rate: goalRate,
        max_iter: maxIter,
        edge_check_steps: edgeCheckSteps,
        collision_buffer: COLLISION_BUFFER,
        wrap_angles: effectiveWrapAngles,
        include_cspace: false,
        cspace_resolution: cspaceResolution,
        seed: 42,
      });
      setResult(data);
      setStartQ(data.start);
      setGoalQ(data.goal);
      if (data.wrap_angles !== wrapAngles) {
        setWrapAngles(data.wrap_angles);
      }
    } catch (runError) {
      setResult(null);
      setError(runError instanceof Error ? runError.message : "Unknown planner error");
    } finally {
      setIsLoading(false);
    }
  }

  function handleAnimate() {
    if (!path || path.length < 2) return;
    setAnimIndex(0);
    setIsAnimating(true);
  }

  async function handleExportMp4() {
    setError("");
    setIsExportingMp4(true);
    try {
      const { blob, filename } = await requestExportMp4({
        start: startQ,
        goal: goalQ,
        obstacle,
        joint_limits: jointLimits,
        link_lengths: linkLengths,
        expand_dist: expandDist,
        goal_rate: goalRate,
        max_iter: maxIter,
        edge_check_steps: edgeCheckSteps,
        collision_buffer: COLLISION_BUFFER,
        wrap_angles: effectiveWrapAngles,
        include_cspace: true,
        cspace_resolution: Math.min(cspaceResolution, 180),
        seed: 42,
        fps: 10,
        dpi: 76,
        bitrate: 1200,
        samples_per_edge: 1,
        max_frames: 30,
      });

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unknown MP4 export error");
    } finally {
      setIsExportingMp4(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="controls-panel">
        <h1>RRT Planner</h1>
        <p className="subtitle">Settings are grouped below and can be collapsed. Start/goal, joint limits, and link lengths are in the right panel.</p>

        <details className="settings-group" open>
          <summary>Obstacle & C-Space Settings</summary>
          <div className="settings-body">
            <div className="toggle-row settings-toggle">
              <span>Wrap Angles</span>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={wrapAngles}
                  disabled={!canWrap}
                  onChange={(e) => {
                    setWrapAngles(e.target.checked);
                    invalidateComputedData({ clearPlanner: true, clearNoGo: false });
                  }}
                />
                <span className="slider" />
              </label>
            </div>
            {!canWrap && <p className="panel-note">Wrap is disabled because current joint limits are bounded (not a full 2*pi range).</p>}

            <div className="control-grid">
              <label>
                Obstacle X
                <input type="number" step="0.1" value={obstacle.x} onChange={(e) => updateObstacleField("x", e.target.value)} />
              </label>
              <label>
                Obstacle Y
                <input type="number" step="0.1" value={obstacle.y} onChange={(e) => updateObstacleField("y", e.target.value)} />
              </label>
              <label>
                Obstacle Size
                <input type="number" min="0.1" max="1.0" step="0.1" value={obstacle.size} onChange={(e) => updateObstacleField("size", e.target.value)} />
              </label>
              <label>
                No-Go Grid Resolution
                <input
                  type="number"
                  min={MIN_CSPACE_RESOLUTION}
                  max={MAX_CSPACE_RESOLUTION}
                  step="20"
                  value={cspaceResolution}
                  onChange={(e) => {
                    setCspaceResolution(parseIntInRange(e.target.value, cspaceResolution, MIN_CSPACE_RESOLUTION, MAX_CSPACE_RESOLUTION));
                    invalidateComputedData();
                  }}
                />
              </label>
            </div>
          </div>
        </details>

        <details className="settings-group" open>
          <summary>Planner Settings</summary>
          <div className="settings-body">
            <div className="control-grid">
              <label>
                Expand Dist
                <input type="number" min="0.005" max="0.4" step="0.005" value={expandDist} onChange={(e) => { setExpandDist(parseNumber(e.target.value, expandDist)); invalidateComputedData({ clearPlanner: true, clearNoGo: false }); }} />
              </label>
              <label>
                Goal Rate
                <input type="number" min="0.0" max="1.0" step="0.01" value={goalRate} onChange={(e) => { setGoalRate(parseNumber(e.target.value, goalRate)); invalidateComputedData({ clearPlanner: true, clearNoGo: false }); }} />
              </label>
              <label>
                Max Iter
                <input type="number" min="100" max="250000" step="100" value={maxIter} onChange={(e) => { setMaxIter(parseNumber(e.target.value, maxIter)); invalidateComputedData({ clearPlanner: true, clearNoGo: false }); }} />
              </label>
              <label>
                Edge Checks
                <input type="number" min="2" max="200" step="1" value={edgeCheckSteps} onChange={(e) => { setEdgeCheckSteps(parseNumber(e.target.value, edgeCheckSteps)); invalidateComputedData({ clearPlanner: true, clearNoGo: false }); }} />
              </label>
            </div>
          </div>
        </details>

        <div className="button-row">
          <button onClick={handleComputeNoGoZone} disabled={isComputingNoGo || isLoading} className="secondary-btn">{isComputingNoGo ? "Computing..." : "Compute No-Go Zone"}</button>
          <button onClick={handleRunPlanner} disabled={isLoading || isComputingNoGo || !cspaceData?.cspace} className="primary-btn">{isLoading ? "Planning..." : "Run Planner"}</button>
          <button onClick={handleAnimate} disabled={!path || path.length < 2 || isAnimating} className="ghost-btn">Animate Path</button>
          <button onClick={handleExportMp4} disabled={isLoading || isComputingNoGo || isExportingMp4} className="ghost-btn">{isExportingMp4 ? "Exporting MP4..." : "Generate MP4"}</button>
        </div>
        {isExportingMp4 && <p className="panel-note">Rendering video... this can take around 20-60 seconds.</p>}

        {error && <p className="error">{error}</p>}

        <details className="settings-group" open>
          <summary>Run Status</summary>
          <div className="settings-body">
            <div className="status-card">
              <div>No-Go Zone: {cspaceData ? "Computed" : "Not computed"}</div>
              <div>No-Go Grid Resolution: {cspaceData?.cspace_resolution ?? cspaceResolution}</div>
              <div>Start Collision: {cspaceData ? (cspaceData.start_in_collision ? "Yes" : "No") : "-"}</div>
              <div>Goal Collision: {cspaceData ? (cspaceData.goal_in_collision ? "Yes" : "No") : "-"}</div>
              <div>Status: {result ? (result.success ? "Path found" : "No path") : "Not run yet"}</div>
              <div>Mode: {effectiveWrapAngles ? "Wrapped" : "Not wrapped"}</div>
              <div>Link Lengths: {`${linkLengths.l1.toFixed(2)}, ${linkLengths.l2.toFixed(2)}`}</div>
              <div>Joint 1 Limits: {`${jointLimits.theta1_min.toFixed(2)} to ${jointLimits.theta1_max.toFixed(2)}`}</div>
              <div>Joint 2 Limits: {`${jointLimits.theta2_min.toFixed(2)} to ${jointLimits.theta2_max.toFixed(2)}`}</div>
              <div>Tree Nodes: {result?.tree?.length ?? 0}</div>
              <div>Path Nodes: {result?.path?.length ?? 0}</div>
              <div>Iterations: {result?.iterations ?? 0}</div>
              <div>Compute Time: {result ? `${result.time_ms.toFixed(1)} ms` : "-"}</div>
            </div>
          </div>
        </details>
      </section>

      <section className="plots-panel">
        <div className="plots-main">
          <WorkspacePlot obstacle={obstacle} currentQ={currentQ} startQ={startQ} goalQ={goalQ} linkLengths={linkLengths} />
          <CspacePlot cspace={activeCspace} tree={result?.tree} parents={result?.parents} path={result?.path} wrapAngles={effectiveWrapAngles} currentQ={currentQ} startQ={startQ} goalQ={goalQ} jointLimits={activeJointLimits} />
          <TorusPlot cspace={activeCspace} path={result?.path} startQ={startQ} goalQ={goalQ} currentQ={currentQ} jointLimits={activeJointLimits} />
        </div>

        <StartGoalPanel startQ={startQ} goalQ={goalQ} jointLimits={jointLimits} linkLengths={linkLengths} onStartChange={updateStartQ} onGoalChange={updateGoalQ} onJointLimitChange={updateJointLimitField} onLinkLengthChange={updateLinkLengthField} />
      </section>
    </main>
  );
}

export default App;
