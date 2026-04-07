export function getArmCoords(t1, t2, l1 = 1.0, l2 = 1.0) {
  const x1 = l1 * Math.cos(t1);
  const y1 = l1 * Math.sin(t1);
  const x2 = x1 + l2 * Math.cos(t1 + t2);
  const y2 = y1 + l2 * Math.sin(t1 + t2);
  return {
    x: [0, x1, x2],
    y: [0, y1, y2],
  };
}

export function worldToPixel(x, y, worldMin, worldMax, sizePx) {
  const scale = sizePx / (worldMax - worldMin);
  const px = (x - worldMin) * scale;
  const py = sizePx - (y - worldMin) * scale;
  return [px, py];
}

export function angleToPixel(theta1, theta2, sizePx, jointLimits) {
  const theta1Span = Math.max(1e-9, jointLimits.theta1_max - jointLimits.theta1_min);
  const theta2Span = Math.max(1e-9, jointLimits.theta2_max - jointLimits.theta2_min);

  const x = ((theta1 - jointLimits.theta1_min) / theta1Span) * sizePx;
  const y = sizePx - ((theta2 - jointLimits.theta2_min) / theta2Span) * sizePx;
  return [x, y];
}

export function shouldSkipWrappedSegment(p1, p2, wrapAngles, jointLimits) {
  if (!wrapAngles) {
    return false;
  }

  const theta1Span = Math.max(1e-9, jointLimits.theta1_max - jointLimits.theta1_min);
  const theta2Span = Math.max(1e-9, jointLimits.theta2_max - jointLimits.theta2_min);

  return Math.abs(p2[0] - p1[0]) > theta1Span / 2 || Math.abs(p2[1] - p1[1]) > theta2Span / 2;
}
