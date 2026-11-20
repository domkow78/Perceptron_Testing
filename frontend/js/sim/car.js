/** Model jazdy auta: "unicycle" — stała prędkość do przodu, jedyny kanał
 * sterowania to skręt (lewo/prawo), bez gazu/hamulca. */
import { pointSegmentDistance } from "./geometry.js";

export const CAR_SPEED = 90; // jednostki świata / s
export const CAR_TURN_RATE = 2.2; // rad/s przy pełnym skręcie
export const CAR_RADIUS = 10; // promień kolizji auta

export function createCar(start) {
  return { x: start.x, y: start.y, heading: start.heading };
}

/** steerSign: -1 = skręt w lewo, 0 = na wprost, 1 = skręt w prawo. */
export function stepCar(car, steerSign, dt) {
  // minus, bo canvas ma odwróconą oś Y względem świata (worldToCanvas) —
  // bez tego strzałka w lewo wizualnie skręcała auto w prawo i odwrotnie.
  car.heading -= steerSign * CAR_TURN_RATE * dt;
  car.heading = Math.atan2(Math.sin(car.heading), Math.cos(car.heading));
  car.x += Math.cos(car.heading) * CAR_SPEED * dt;
  car.y += Math.sin(car.heading) * CAR_SPEED * dt;
}

export function checkCrash(car, walls) {
  for (const wall of walls) {
    if (pointSegmentDistance(car.x, car.y, wall.x1, wall.y1, wall.x2, wall.y2) < CAR_RADIUS) {
      return true;
    }
  }
  return false;
}

export function resetCar(car, start) {
  car.x = start.x;
  car.y = start.y;
  car.heading = start.heading;
}
