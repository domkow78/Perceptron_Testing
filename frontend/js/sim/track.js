/** Generuje zamknięty tor jako dwie pętle ścian-odcinków (zewnętrzna i
 * wewnętrzna), w jednostkach świata (dowolne jednostki płaskie).
 *
 * Tor to owal z dwiema szykanami (po jednej na każdej "prostej" — krótki
 * S-zakręt: w prawo, zaraz potem w lewo). Bez szykan prosty owal wymagałby
 * skręcania tylko w JEDNĄ stronę przez całe okrążenie (suma skrętów na
 * dowolnej zamkniętej pętli to zawsze 360° w jedną stronę) — sieć uczyłaby
 * się stałej reguły "zawsze w lewo" zamiast patrzeć na lidar. Szykany
 * wymuszają przykłady obu kierunków skrętu w nagranej jeździe.
 *
 * Tor budowany jest metodą "żółwia" (turtle graphics): opisujemy pół
 * okrążenia jako sekwencję ruchów (prosta/łuk) o łącznym skręcie dokładnie
 * 180°, a potem przechodzimy dokładnie tę samą sekwencję drugi raz —
 * dwa takie "pół-okrążenia" automatycznie zamykają się w pętlę (obrót o
 * 180° dwa razy = 360°, a przesunięcie drugiej połowy jest dokładnie
 * przeciwne do pierwszej, bo to ten sam ruch widziany z odwróconego
 * kierunku).
 */

export function buildTrack({
  leadLength = 160,
  trailLength = 160,
  chicaneRadius = 90,
  chicaneAngleDeg = 35,
  turnRadius = 150,
  trackWidth = 90,
  segmentsPerTurn = 16,
} = {}) {
  const chicaneAngle = (chicaneAngleDeg * Math.PI) / 180;
  const anglePerSegment = Math.PI / segmentsPerTurn;

  // Pół okrążenia: prosta -> szykana (prawo-lewo, suma skrętu = 0) -> prosta
  // -> duży zakręt o 180°. Łączny skręt = 0 + 180° = 180°, jak wymaga trik z zamykaniem pętli.
  const halfLapMoves = [
    { type: "straight", length: leadLength },
    { type: "arc", radius: chicaneRadius, angle: -chicaneAngle },
    { type: "arc", radius: chicaneRadius, angle: chicaneAngle },
    { type: "straight", length: trailLength },
    { type: "arc", radius: turnRadius, angle: Math.PI },
  ];

  const origin = { x: 0, y: 0, heading: 0 };
  const firstHalf = walkMoves(origin, halfLapMoves, anglePerSegment);
  const afterFirstHalf = firstHalf[firstHalf.length - 1];
  const secondHalf = walkMoves(afterFirstHalf, halfLapMoves, anglePerSegment);

  // secondHalf[0] duplikuje afterFirstHalf (już jest na liście), a ostatni
  // punkt drugiej połowy powinien (z dokładnością do błędu zmiennoprzecinkowego)
  // pokrywać się z origin — usuwamy go, bo zamknięcie pętli robi modulo przy
  // budowaniu ścian.
  const centerline = firstHalf.concat(secondHalf.slice(1, -1));

  // wyśrodkowanie kształtu wokół (0,0), żeby ładnie mieścił się na canvasie
  const xs = centerline.map((p) => p.x);
  const ys = centerline.map((p) => p.y);
  const offsetX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const offsetY = (Math.min(...ys) + Math.max(...ys)) / 2;
  centerline.forEach((p) => {
    p.x -= offsetX;
    p.y -= offsetY;
  });

  const halfWidth = trackWidth / 2;
  const outerWalls = [];
  const innerWalls = [];
  const n = centerline.length;
  for (let i = 0; i < n; i++) {
    const a = centerline[i];
    const b = centerline[(i + 1) % n];
    const outerA = offsetPoint(a, halfWidth);
    const outerB = offsetPoint(b, halfWidth);
    const innerA = offsetPoint(a, -halfWidth);
    const innerB = offsetPoint(b, -halfWidth);
    outerWalls.push({ x1: outerA.x, y1: outerA.y, x2: outerB.x, y2: outerB.y });
    innerWalls.push({ x1: innerA.x, y1: innerA.y, x2: innerB.x, y2: innerB.y });
  }

  return {
    outerWalls,
    innerWalls,
    start: { x: centerline[0].x, y: centerline[0].y, heading: centerline[0].heading },
  };
}

/** Idzie "żółwiem" przez listę ruchów ({type:"straight",length} lub
 * {type:"arc",radius,angle} — angle dodatni = skręt w lewo), zwraca listę
 * próbkowanych punktów {x,y,heading} (łuki próbkowane co `anglePerSegment`). */
function walkMoves(start, moves, anglePerSegment) {
  const points = [{ x: start.x, y: start.y, heading: start.heading }];
  let x = start.x;
  let y = start.y;
  let heading = start.heading;

  for (const move of moves) {
    if (move.type === "straight") {
      x += Math.cos(heading) * move.length;
      y += Math.sin(heading) * move.length;
      points.push({ x, y, heading });
      continue;
    }

    const headingAtStart = heading;
    const s = Math.sign(move.angle);
    const steps = Math.max(2, Math.round(Math.abs(move.angle) / anglePerSegment));
    const cx = x + move.radius * Math.cos(headingAtStart + (s * Math.PI) / 2);
    const cy = y + move.radius * Math.sin(headingAtStart + (s * Math.PI) / 2);
    const alpha0 = headingAtStart - (s * Math.PI) / 2;

    for (let i = 1; i <= steps; i++) {
      const u = (move.angle * i) / steps;
      const alpha = alpha0 + u;
      x = cx + move.radius * Math.cos(alpha);
      y = cy + move.radius * Math.sin(alpha);
      heading = headingAtStart + u;
      points.push({ x, y, heading });
    }
  }
  return points;
}

/** Przesuwa wierzchołek linii środkowej o `distance` wzdłuż normalnej na zewnątrz toru. */
function offsetPoint(vertex, distance) {
  const normalAngle = vertex.heading - Math.PI / 2;
  return {
    x: vertex.x + distance * Math.cos(normalAngle),
    y: vertex.y + distance * Math.sin(normalAngle),
  };
}
