/** Wspólna matematyka 2D używana przez model auta (kolizje) i lidar (raycasting). */

/** Przecięcie promienia (origin, angle) z odcinkiem (x1,y1)-(x2,y2); zwraca odległość wzdłuż promienia albo null. */
export function rayIntersectSegment(ox, oy, angle, maxRange, x1, y1, x2, y2) {
  const rdx = Math.cos(angle);
  const rdy = Math.sin(angle);
  const sdx = x2 - x1;
  const sdy = y2 - y1;
  const denom = rdx * sdy - rdy * sdx;
  if (Math.abs(denom) < 1e-9) return null;

  const t = ((x1 - ox) * sdy - (y1 - oy) * sdx) / denom;
  const u = ((x1 - ox) * rdy - (y1 - oy) * rdx) / denom;
  if (t >= 0 && t <= maxRange && u >= 0 && u <= 1) return t;
  return null;
}

/** Odległość punktu (px,py) od odcinka (x1,y1)-(x2,y2) (rzut z obcięciem do [0,1]). */
export function pointSegmentDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq > 1e-9 ? ((px - x1) * dx + (py - y1) * dy) / lenSq : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}
