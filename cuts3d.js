// 3-D cutting-plane animation: a polytope with its integer points, the LP
// optimum, and valid cuts that remove the fractional optimum one at a time.
(() => {
  const canvas = document.getElementById("cut-canvas");
  if (!canvas) return;
  const label = document.getElementById("cut-label");
  const ctx = canvas.getContext("2d");

  // ---------- geometry ----------
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => scale(a, 1 / Math.hypot(a[0], a[1], a[2]));

  function solve3(r0, r1, r2, b) {
    const det = dot(r0, cross(r1, r2));
    if (Math.abs(det) < 1e-9) return null;
    // Cramer's rule via the adjugate rows.
    const c0 = cross(r1, r2), c1 = cross(r2, r0), c2 = cross(r0, r1);
    return scale(add(add(scale(c0, b[0]), scale(c1, b[1])), scale(c2, b[2])), 1 / det);
  }

  // Order points of a planar convex polygon with normal n by angle.
  function orderAround(pts, n) {
    const c = scale(pts.reduce(add, [0, 0, 0]), 1 / pts.length);
    const e = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = unit(cross(n, e)), w = cross(unit(n), u);
    return pts
      .map((p) => ({ p, t: Math.atan2(dot(sub(p, c), w), dot(sub(p, c), u)) }))
      .sort((a, b) => a.t - b.t)
      .map((o) => o.p);
  }

  // Vertices and faces of { x : a·x <= b for every (a, b) in H }.
  function polytope(H) {
    const V = [];
    for (let i = 0; i < H.length; i++)
      for (let j = i + 1; j < H.length; j++)
        for (let k = j + 1; k < H.length; k++) {
          const v = solve3(H[i].a, H[j].a, H[k].a, [H[i].b, H[j].b, H[k].b]);
          if (!v || !H.every((h) => dot(h.a, v) <= h.b + 1e-7)) continue;
          if (!V.some((u) => Math.hypot(...sub(u, v)) < 1e-6)) V.push(v);
        }
    const faces = [];
    for (const h of H) {
      const pts = V.filter((v) => Math.abs(dot(h.a, v) - h.b) < 1e-6);
      if (pts.length >= 3) faces.push({ n: h.a, pts: orderAround(pts, h.a) });
    }
    return { V, faces };
  }

  // ---------- the integer program and its cuts ----------
  const c = [1, 1.15, 1.3]; // maximize c·x
  const H0 = [
    { a: [-1, 0, 0], b: 0 }, { a: [0, -1, 0], b: 0 }, { a: [0, 0, -1], b: 0 },
    { a: [1, 0, 0], b: 2.6 }, { a: [0, 1, 0], b: 2.5 }, { a: [0, 0, 1], b: 2.4 },
    { a: [1, 1, 1], b: 5.6 }, { a: [1, -1, 1], b: 3.3 },
  ];
  const lattice = [], feasible = [];
  for (let x = 0; x <= 3; x++)
    for (let y = 0; y <= 3; y++)
      for (let z = 0; z <= 3; z++) {
        const p = [x, y, z];
        lattice.push(p);
        if (H0.every((h) => dot(h.a, p) <= h.b + 1e-9)) feasible.push(p);
      }
  const dirs = [c, [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
    [1, -1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const argmax = (P) => P.V.reduce((best, v) => (dot(c, v) > dot(c, best) ? v : best));
  const isInt = (v) => v.every((t) => Math.abs(t - Math.round(t)) < 1e-6);

  // Each step: the current polytope and optimum, and the deepest valid cut
  // a·x <= max over integer points of a·x that removes that optimum.
  const steps = [];
  let H = H0.slice();
  for (let s = 0; s < 6; s++) {
    const P = polytope(H), xs = argmax(P);
    const step = { P, xs };
    steps.push(step);
    if (isInt(xs)) break;
    let best = null;
    for (const d of dirs) {
      const b = Math.max(...feasible.map((p) => dot(d, p)));
      const depth = (dot(d, xs) - b) / Math.hypot(...d);
      if (depth > 1e-3 && (!best || depth > best.depth)) best = { a: d, b, depth };
    }
    if (!best) break;
    step.cut = best;
    step.piece = polytope(H.concat([{ a: scale(best.a, -1), b: -best.b }]));
    step.bStart = Math.max(...P.V.map((v) => dot(best.a, v))) + 0.7;
    H = H.concat([{ a: best.a, b: best.b }]);
  }

  // ---------- drawing ----------
  const center = [1.3, 1.25, 1.2];
  let yaw = -0.6, pitch = 0.42, w = 0, h = 0, s = 1, dpr = 1;
  let colors = {};

  function readColors() {
    const css = getComputedStyle(document.documentElement);
    const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    colors = {
      accent: get("--accent", "#0b6e4f"),
      text: get("--text", "#1f2328"),
      muted: get("--muted", "#5f6670"),
      cut: get("--cut", "#d9822b"),
    };
  }

  function rgba(hex, a) {
    const m = hex.replace("#", "");
    const n = parseInt(m.length === 3 ? m.split("").map((ch) => ch + ch).join("") : m, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    s = Math.min(w / 6.4, h / 5.6);
  }

  // Rotate around the vertical axis, tilt, then project with mild perspective.
  function project(p) {
    const q = sub(p, center);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = q[0] * cy - q[1] * sy, y1 = q[0] * sy + q[1] * cy, z1 = q[2];
    const depth = y1 * cp - z1 * sp, up = y1 * sp + z1 * cp;
    const k = 9 / (9 + depth);
    return { x: w / 2 + x1 * s * k, y: h * 0.52 - up * s * k, depth };
  }

  function facesFacing(P, front, shift) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return P.faces
      .filter((f) => {
        const n = f.n, ny = n[0] * sy + n[1] * cy;
        const nd = ny * cp - n[2] * sp; // depth component of the outward normal
        return front ? nd < 0 : nd >= 0;
      })
      .map((f) => ({ pts: f.pts.map((p) => project(shift ? add(p, shift) : p)) }));
  }

  function drawFaces(list, fill, stroke, lineWidth) {
    for (const f of list) {
      ctx.beginPath();
      f.pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      ctx.stroke();
    }
  }

  function drawPoints(P) {
    const inside = (p) => P.faces.every((f) => dot(f.n, p) <= dot(f.n, f.pts[0]) + 1e-9);
    const pts = lattice.map((p) => ({ p, q: project(p), in: inside(p) })).sort((a, b) => b.q.depth - a.q.depth);
    for (const o of pts) {
      ctx.beginPath();
      ctx.arc(o.q.x, o.q.y, o.in ? 3.2 : 1.6, 0, Math.PI * 2);
      ctx.fillStyle = o.in ? rgba(colors.text, 0.85) : rgba(colors.muted, 0.35);
      ctx.fill();
    }
  }

  function drawOptimum(v, pulse, color) {
    const q = project(v);
    ctx.beginPath();
    ctx.arc(q.x, q.y, 6 + 5 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = rgba(color, 0.18 * (1 - pulse));
    ctx.fill();
    ctx.beginPath();
    ctx.arc(q.x, q.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  // The cutting plane a·x = b, clipped to a box around the polytope.
  function drawPlane(a, b, alpha) {
    const lo = -0.3, hi = 2.9, pts = [];
    const corners = [];
    for (const x of [lo, hi]) for (const y of [lo, hi]) for (const z of [lo, hi]) corners.push([x, y, z]);
    for (let i = 0; i < 8; i++)
      for (let j = i + 1; j < 8; j++) {
        const d = sub(corners[j], corners[i]);
        if (d.filter((t) => t !== 0).length !== 1) continue;
        const den = dot(a, d);
        if (Math.abs(den) < 1e-12) continue;
        const t = (b - dot(a, corners[i])) / den;
        if (t >= 0 && t <= 1) pts.push(add(corners[i], scale(d, t)));
      }
    if (pts.length < 3) return;
    const poly = orderAround(pts, a).map(project);
    ctx.beginPath();
    poly.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.fillStyle = rgba(colors.cut, 0.22 * alpha);
    ctx.fill();
    ctx.strokeStyle = rgba(colors.cut, 0.9 * alpha);
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  const fmt = (v) => "(" + v.map((t) => (isInt([t]) ? String(Math.round(t)) : t.toFixed(2))).join(", ") + ")";
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  // ---------- timeline ----------
  const SHOW = 1.8, SWEEP = 1.5, SPLIT = 1.5, FINAL = 3.5;
  const timeline = [];
  let total = 0;
  steps.forEach((step, i) => {
    if (step.cut) {
      timeline.push({ i, phase: "show", t0: total, len: SHOW }); total += SHOW;
      timeline.push({ i, phase: "sweep", t0: total, len: SWEEP }); total += SWEEP;
      timeline.push({ i, phase: "split", t0: total, len: SPLIT }); total += SPLIT;
    } else {
      timeline.push({ i, phase: "final", t0: total, len: FINAL }); total += FINAL;
    }
  });
  const cutsUsed = steps.filter((st) => st.cut).length;

  function setLabel(text) {
    if (label && label.textContent !== text) label.textContent = text;
  }

  function frame(t) {
    const seg = timeline.find((g) => t >= g.t0 && t < g.t0 + g.len) || timeline[timeline.length - 1];
    const u = Math.min(1, (t - seg.t0) / seg.len);
    const step = steps[seg.i];
    const pulse = (t * 1.2) % 1;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const body = seg.phase === "split" ? steps[seg.i + 1].P : step.P;
    const fill = rgba(colors.accent, 0.13), edge = rgba(colors.accent, 0.85);
    drawFaces(facesFacing(body, false), rgba(colors.accent, 0.06), rgba(colors.accent, 0.35), 1);
    drawPoints(body);
    drawFaces(facesFacing(body, true), fill, edge, 1.5);

    if (seg.phase === "show") {
      drawOptimum(step.xs, pulse, colors.cut);
      setLabel(`LP optimum ${fmt(step.xs)} is fractional`);
    } else if (seg.phase === "sweep") {
      const b = step.bStart + (step.cut.b - step.bStart) * ease(u);
      drawOptimum(step.xs, pulse, colors.cut);
      drawPlane(step.cut.a, b, Math.min(1, u * 3));
      setLabel(`Cut ${seg.i + 1}: every integer point stays on one side`);
    } else if (seg.phase === "split") {
      const shift = scale(unit(step.cut.a), 0.9 * ease(u));
      const a = 1 - u;
      drawFaces(facesFacing(step.piece, false, shift), rgba(colors.cut, 0.06 * a), rgba(colors.cut, 0.35 * a), 1);
      drawFaces(facesFacing(step.piece, true, shift), rgba(colors.cut, 0.16 * a), rgba(colors.cut, 0.9 * a), 1.5);
      drawPlane(step.cut.a, step.cut.b, 1 - u);
      drawOptimum(add(step.xs, shift), 0, rgba(colors.cut, a));
      setLabel(`Cut ${seg.i + 1} removes the fractional optimum`);
    } else {
      drawOptimum(step.xs, pulse, colors.accent);
      setLabel(`Integer optimum ${fmt(step.xs)} after ${cutsUsed} cut${cutsUsed === 1 ? "" : "s"}`);
    }
  }

  // ---------- loop, drag to rotate, reduced motion ----------
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let start = performance.now(), dragYaw = 0, visible = true, raf = 0, dragging = null;

  function tick(now) {
    raf = 0;
    const t = ((now - start) / 1000) % total;
    yaw = -0.6 + dragYaw + (reduced ? 0 : (now - start) / 1000 * 0.22);
    frame(reduced ? timeline[timeline.length - 1].t0 : t);
    if (visible && !reduced) raf = requestAnimationFrame(tick);
  }
  const redraw = () => { if (!raf) raf = requestAnimationFrame(tick); };

  canvas.addEventListener("pointerdown", (e) => {
    dragging = { x: e.clientX, y: e.clientY, yaw: dragYaw, pitch };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    dragYaw = dragging.yaw + (e.clientX - dragging.x) * 0.01;
    pitch = Math.max(-0.2, Math.min(1.3, dragging.pitch + (e.clientY - dragging.y) * 0.01));
    if (reduced) redraw();
  });
  const endDrag = () => { dragging = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    if (visible) redraw();
  }).observe(canvas);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); redraw(); });
  window.addEventListener("resize", () => { resize(); redraw(); });

  readColors();
  resize();
  redraw();
})();
