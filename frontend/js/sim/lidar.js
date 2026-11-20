/** Lidar: wachlarz promieni rzucanych od auta na ściany toru (raycasting). */
import { rayIntersectSegment } from "./geometry.js";

export const RAY_COUNT = 7;
export const RAY_SPREAD = Math.PI; // 180 stopni: -90..+90 względem kierunku jazdy
export const RAY_MAX_RANGE = 220;

export function computeLidar(car, walls) {
  const distances = [];
  const angles = [];
  for (let i = 0; i < RAY_COUNT; i++) {
    const offset = -RAY_SPREAD / 2 + (RAY_SPREAD * i) / (RAY_COUNT - 1);
    const angle = car.heading + offset;
    let best = RAY_MAX_RANGE;
    for (const wall of walls) {
      const t = rayIntersectSegment(car.x, car.y, angle, RAY_MAX_RANGE, wall.x1, wall.y1, wall.x2, wall.y2);
      if (t !== null && t < best) best = t;
    }
    distances.push(best);
    angles.push(angle);
  }
  // znormalizowane do [0,1]: 1 = czysto, 0 = tuż przy ścianie — to wejście sieci
  const normalized = distances.map((d) => d / RAY_MAX_RANGE);
  return { distances, normalized, angles };
}
