// Two 3-D animations: distributed gradient descent (workers send gradients,
// never data, to an aggregator) and spatial branch and bound on a landscape with
// many valleys.
(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dark = window.matchMedia("(prefers-color-scheme: dark)");
  let colors = {};

  function hexToRgb(hex) {
    const m = hex.replace("#", "");
    const n = parseInt(m.length === 3 ? m.split("").map((ch) => ch + ch).join("") : m, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function readColors() {
    const css = getComputedStyle(document.documentElement);
    const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    colors = {
      accent: get("--accent", "#0b6e4f"),
      text: get("--text", "#1f2328"),
      muted: get("--muted", "#5f6670"),
      surface: get("--surface", "#ffffff"),
      cut: get("--cut", "#d9822b"),
    };
    // One color per worker, readable in both themes.
    colors.workers = dark.matches
      ? ["#7aa7f0", "#c58af0", "#e8c55a", "#f783ac"]
      : ["#3b6fc4", "#9b4dca", "#b8860b", "#d6336c"];
  }

  function rgba(hex, a) {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r},${g},${b},${a})`;
  }

  const mixRgb = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const css = (c, k = 1) => `rgb(${Math.round(c[0] * k)},${Math.round(c[1] * k)},${Math.round(c[2] * k)})`;
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  // Canvas sizing, visibility and the loop. render(ctx, w, h, t) gets t in
  // seconds, or null when the viewer prefers reduced motion.
  function animate(canvas, render) {
    const ctx = canvas.getContext("2d");
    let w = 0, h = 0, dpr = 1, visible = true, raf = 0;
    const start = performance.now();
    function resize() {
      dpr = window.devicePixelRatio || 1;
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    function tick(now) {
      raf = 0;
      readColors();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      render(ctx, w, h, reduced ? null : (now - start) / 1000);
      if (visible && !reduced) raf = requestAnimationFrame(tick);
    }
    const redraw = () => { if (!raf) raf = requestAnimationFrame(tick); };
    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible) redraw();
    }).observe(canvas);
    window.addEventListener("resize", () => { resize(); redraw(); });
    dark.addEventListener("change", redraw);
    resize();
    return redraw;
  }

  function setText(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
  }

  // ---------- distributed gradient descent ----------
  // Worker i keeps its data and the local objective
  //   f_i(w) = ½ (w − a_i)' Q_i (w − a_i).
  // Each round the aggregator sends w to the workers, each worker sends back
  // ∇f_i(w) (never its data), and the aggregator steps
  //   w ← w − η Σ p_i ∇f_i(w),  the gradient step on F = Σ p_i f_i.
  function distributed() {
    const canvas = document.getElementById("dist-canvas");
    if (!canvas) return;
    const label = document.getElementById("dist-label");
    const M = 4, G = 24, MAX_ROUNDS = 14, HOLD = 3;
    const PHASES = [["send", 0.7], ["compute", 0.8], ["return", 0.9], ["update", 1.1]];
    const FAST = 0.45; // rounds after the first two play faster
    const ARROW = 2;
    let run = null, yaw = -0.6, pitch = 0.6, dragYaw = 0, dragging = null;

    const mul = (Q, v) => [Q[0][0] * v[0] + Q[0][1] * v[1], Q[1][0] * v[0] + Q[1][1] * v[1]];

    function newRun(t0) {
      const a = [], Q = [], n = [];
      for (let i = 0; i < M; i++) {
        const dir = [[-1, 1], [-1, -1], [1, 1], [1, -1]][i];
        a.push([dir[0] * (0.35 + Math.random() * 0.4), dir[1] * (0.35 + Math.random() * 0.4)]);
        const th = Math.random() * Math.PI, c = Math.cos(th), s = Math.sin(th);
        const l1 = 0.6 + Math.random() * 1.2, l2 = 0.6 + Math.random() * 1.2;
        Q.push([[c * c * l1 + s * s * l2, c * s * (l1 - l2)], [c * s * (l1 - l2), s * s * l1 + c * c * l2]]);
        n.push(3 + Math.floor(Math.random() * 7));
      }
      const total = n.reduce((x, y) => x + y, 0);
      const p = n.map((v) => v / total);
      const grad = (i, w) => mul(Q[i], [w[0] - a[i][0], w[1] - a[i][1]]);
      const F = (w) => Q.reduce((acc, q, i) => {
        const d = [w[0] - a[i][0], w[1] - a[i][1]], qd = mul(q, d);
        return acc + p[i] * 0.5 * (d[0] * qd[0] + d[1] * qd[1]);
      }, 0);

      // H = Σ p_i Q_i; the minimizer solves H w = Σ p_i Q_i a_i.
      const H = Q.reduce((S, q, i) => [[S[0][0] + p[i] * q[0][0], S[0][1] + p[i] * q[0][1]], [S[1][0] + p[i] * q[1][0], S[1][1] + p[i] * q[1][1]]], [[0, 0], [0, 0]]);
      const rhs = Q.reduce((v, q, i) => { const t = mul(q, a[i]); return [v[0] + p[i] * t[0], v[1] + p[i] * t[1]]; }, [0, 0]);
      const det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
      const wstar = [(H[1][1] * rhs[0] - H[0][1] * rhs[1]) / det, (H[0][0] * rhs[1] - H[1][0] * rhs[0]) / det];
      const lmax = (H[0][0] + H[1][1]) / 2 + Math.sqrt((H[0][0] - H[1][1]) ** 2 / 4 + H[0][1] ** 2);
      const eta = 0.42 / lmax;

      let fmin = F(wstar), fmax = -Infinity;
      const heights = [];
      for (let i = 0; i <= G; i++) {
        heights.push([]);
        for (let j = 0; j <= G; j++) { const v = F([(i / G) * 2 - 1, (j / G) * 2 - 1]); heights[i].push(v); fmax = Math.max(fmax, v); }
      }
      const norm = (v) => (v - fmin) / (fmax - fmin);

      // Start in the corner farthest from the optimum.
      const W = [[wstar[0] > 0 ? -0.85 : 0.85, wstar[1] > 0 ? -0.85 : 0.85]], steps = [];
      for (let r = 0; r < MAX_ROUNDS; r++) {
        const w = W[r];
        const local = a.map((_, i) => { const g = grad(i, w); return [-eta * g[0], -eta * g[1]]; });
        const step = local.reduce((s, v, i) => [s[0] + p[i] * v[0], s[1] + p[i] * v[1]], [0, 0]);
        steps.push({ local, step });
        W.push([w[0] + step[0], w[1] + step[1]]);
        if (Math.hypot(W[r + 1][0] - wstar[0], W[r + 1][1] - wstar[1]) < 0.02) break;
      }

      const segs = [];
      let t = 0;
      steps.forEach((_, r) => PHASES.forEach(([ph, len]) => {
        const d = len * (r < 2 ? 1 : FAST);
        segs.push({ r, ph, t0: t, len: d });
        t += d;
      }));
      run = { a, p, n, F, wstar, W, steps, heights, norm, segs, t0, total: t + HOLD };
    }

    let w = 0, h = 0, s = 1, side = 60;
    function project(q) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = q[0] * cy - q[1] * sy, y1 = q[0] * sy + q[1] * cy, z1 = q[2];
      const depth = y1 * cp - z1 * sp, up = y1 * sp + z1 * cp;
      const k = 6 / (6 + depth);
      return { x: w / 2 + x1 * s * k, y: h * 0.56 - up * s * k, depth };
    }
    const lift = (v) => -0.5 + 0.75 * run.norm(v);
    const onSurface = (pt, dz = 0) => [pt[0], pt[1], lift(run.F(pt)) + dz];

    function arrow(ctx, from, to, color, width, alpha) {
      const A = project(from), B = project(to);
      const len = Math.hypot(B.x - A.x, B.y - A.y);
      if (len < 2 || alpha <= 0) return;
      const ux = (B.x - A.x) / len, uy = (B.y - A.y) / len, head = Math.min(9, len * 0.45);
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x - ux * head * 0.6, B.y - uy * head * 0.6);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(B.x, B.y);
      ctx.lineTo(B.x - ux * head - uy * head * 0.5, B.y - uy * head + ux * head * 0.5);
      ctx.lineTo(B.x - ux * head + uy * head * 0.5, B.y - uy * head - ux * head * 0.5);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    const redraw = animate(canvas, (ctx, cw, ch, t) => {
      w = cw; h = ch; side = w < 500 ? 34 : 64; s = Math.min((w - 2 * side) / 3.0, h / 2.5);
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      yaw = -0.6 + dragYaw + (t === null ? 0 : now * 0.12);
      const local = t === null ? run.segs[5].t0 + 0.1 : now - run.t0;
      const seg = run.segs.find((g) => local >= g.t0 && local < g.t0 + g.len);
      const u = seg ? Math.min(1, (local - seg.t0) / seg.len) : 1;
      const r = seg ? seg.r : run.steps.length;
      const done = !seg;

      // Model position: it moves during the second half of the update phase.
      let wm = run.W[r];
      if (seg && seg.ph === "update" && u > 0.45) {
        const v = ease((u - 0.45) / 0.55), A = run.W[r], B = run.W[r + 1];
        wm = [A[0] + (B[0] - A[0]) * v, A[1] + (B[1] - A[1]) * v];
      }

      // Surface and markers, depth-sorted.
      const accent = hexToRgb(colors.accent), surf = hexToRgb(colors.surface);
      const high = mixRgb(accent, surf, 0.8);
      const light = [-0.35, -0.55, 0.76];
      const items = [];
      const P3 = (i, j) => [(i / G) * 2 - 1, (j / G) * 2 - 1, lift(run.heights[i][j])];
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const q = [P3(i, j), P3(i + 1, j), P3(i + 1, j + 1), P3(i, j + 1)];
          const d1 = [q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]], d2 = [q[3][0] - q[1][0], q[3][1] - q[1][1], q[3][2] - q[1][2]];
          const nv = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
          const shade = 0.65 + 0.35 * Math.max(0, (nv[0] * light[0] + nv[1] * light[1] + nv[2] * light[2]) / Math.hypot(...nv));
          const hv = (run.norm(run.heights[i][j]) + run.norm(run.heights[i + 1][j + 1])) / 2;
          const pts = q.map(project);
          items.push({ depth: (pts[0].depth + pts[1].depth + pts[2].depth + pts[3].depth) / 4, pts, col: mixRgb(accent, high, Math.min(1, 0.15 + hv * 1.1)), shade });
        }
      run.a.forEach((ai, i) => {
        const q = project(onSurface(ai, 0.015));
        items.push({ depth: q.depth - 0.02, mark: "ring", q, color: colors.workers[i] });
      });
      const qs = project(onSurface(run.wstar, 0.015));
      items.push({ depth: qs.depth - 0.02, mark: "star", q: qs });
      items.sort((x, y) => y.depth - x.depth);

      for (const it of items) {
        if (it.mark === "ring") {
          ctx.beginPath();
          ctx.arc(it.q.x, it.q.y, 5, 0, Math.PI * 2);
          ctx.strokeStyle = it.color;
          ctx.lineWidth = 2.2;
          ctx.stroke();
        } else if (it.mark === "star") {
          ctx.beginPath();
          for (let k = 0; k < 10; k++) {
            const rr = k % 2 ? 3 : 7.5, ang = -Math.PI / 2 + (k * Math.PI) / 5;
            const x = it.q.x + rr * Math.cos(ang), y = it.q.y + rr * Math.sin(ang);
            k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.closePath();
          ctx.fillStyle = colors.cut;
          ctx.fill();
        } else {
          const p = it.pts;
          ctx.beginPath();
          ctx.moveTo(p[0].x, p[0].y);
          for (let k = 1; k < 4; k++) ctx.lineTo(p[k].x, p[k].y);
          ctx.closePath();
          ctx.fillStyle = css(it.col, it.shade);
          ctx.fill();
          ctx.strokeStyle = css(it.col, it.shade * 0.92);
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }

      // Path of the shared model.
      ctx.beginPath();
      run.W.slice(0, r + 1).concat([wm]).forEach((pt, k) => {
        const q = project(onSurface(pt, 0.02));
        k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
      });
      ctx.strokeStyle = rgba(colors.text, 0.55);
      ctx.lineWidth = 2;
      ctx.stroke();

      // Communication links, workers with their data, and the aggregator.
      // Workers sit on the left and right, the aggregator at the top.
      const S = { x: w / 2, y: 30 };
      const workerAt = [[side, 0.3], [side, 0.72], [w - side, 0.3], [w - side, 0.72]].map(([x, f]) => ({ x, y: h * f }));
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      workerAt.forEach((q) => {
        ctx.beginPath();
        ctx.moveTo(q.x, q.y);
        ctx.lineTo(S.x, S.y);
        ctx.strokeStyle = rgba(colors.muted, 0.35);
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // Arrows at the model: each worker's step −η∇f_i, then the weighted sum.
      const zm = lift(run.F(wm)) + 0.03;
      const from = [wm[0], wm[1], zm];
      if (seg && seg.ph !== "send") {
        const st = run.steps[r];
        const grow = seg.ph === "compute" ? ease(u) : 1;
        const fade = seg.ph === "update" ? Math.max(0, 1 - u / 0.4) : 1;
        // Arrows are drawn ARROW times the actual step so directions read clearly.
        st.local.forEach((v, i) => {
          const len = Math.hypot(v[0], v[1]) * ARROW, cap = ARROW * Math.min(1, 0.85 / len) * grow;
          arrow(ctx, from, [wm[0] + v[0] * cap, wm[1] + v[1] * cap, zm], colors.workers[i], 2.2, 0.95 * fade);
        });
        if (seg.ph === "update") {
          const A = run.W[r], B = run.W[r + 1], g = ARROW * ease(Math.min(1, u / 0.45));
          arrow(ctx, [A[0], A[1], lift(run.F(A)) + 0.03], [A[0] + (B[0] - A[0]) * g, A[1] + (B[1] - A[1]) * g, lift(run.F(A)) + 0.03], colors.cut, 3, u > 0.9 ? (1 - u) * 10 : 1);
        }
      }

      // The model point.
      const qm = project(from);
      ctx.beginPath();
      ctx.arc(qm.x, qm.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = colors.text;
      ctx.fill();
      ctx.strokeStyle = colors.surface;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Messages: the model goes out, gradients come back. Data never moves.
      if (seg && (seg.ph === "send" || seg.ph === "return")) {
        const v = ease(u);
        workerAt.forEach((q, i) => {
          const [x0, y0, x1, y1] = seg.ph === "send" ? [S.x, S.y, q.x, q.y] : [q.x, q.y, S.x, S.y];
          ctx.beginPath();
          ctx.arc(x0 + (x1 - x0) * v, y0 + (y1 - y0) * v, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = seg.ph === "send" ? colors.text : colors.workers[i];
          ctx.fill();
          ctx.strokeStyle = colors.surface;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      }

      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      workerAt.forEach((q, i) => {
        // The worker's data: one square per sample, staying here.
        const cols = 3;
        for (let k = 0; k < run.n[i]; k++) {
          ctx.fillStyle = rgba(colors.workers[i], 0.55);
          const dx = q.x < w / 2 ? 14 : -14 - cols * 6;
          ctx.fillRect(q.x + dx + (k % cols) * 6, q.y - 9 + Math.floor(k / cols) * 6, 4.5, 4.5);
        }
        ctx.beginPath();
        ctx.arc(q.x, q.y, 9, 0, Math.PI * 2);
        ctx.fillStyle = colors.workers[i];
        ctx.fill();
        ctx.fillStyle = colors.workers[i];
        ctx.fillText(`Worker ${i + 1}`, q.x, q.y + 23);
        ctx.font = "400 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = colors.muted;
        ctx.fillText(`p = ${run.p[i].toFixed(2)}`, q.x, q.y + 36);
        ctx.font = "600 11px Inter, system-ui, sans-serif";
      });
      // The aggregator: a hub that combines the incoming gradients, Σ pᵢ ∇fᵢ.
      const R = 15;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(S.x + Math.cos(a) * (R + 3), S.y + Math.sin(a) * (R + 3));
        ctx.lineTo(S.x + Math.cos(a) * (R + 7), S.y + Math.sin(a) * (R + 7));
        ctx.strokeStyle = rgba(colors.accent, 0.6);
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.stroke();
      }
      ctx.lineCap = "butt";
      ctx.beginPath();
      ctx.arc(S.x, S.y, R, 0, Math.PI * 2);
      ctx.fillStyle = colors.surface;
      ctx.fill();
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = colors.accent;
      ctx.font = "600 17px 'Source Serif 4', Georgia, serif";
      ctx.textBaseline = "middle";
      ctx.fillText("Σ", S.x, S.y + 1);
      ctx.textBaseline = "alphabetic";
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.fillStyle = colors.text;
      ctx.textAlign = "left";
      ctx.fillText("Aggregator", S.x + R + 12, S.y + 4);
      ctx.textAlign = "center";

      const fval = run.F(wm).toFixed(3);
      if (done) setText(label, `Converged to the minimizer of F = Σ pᵢ fᵢ in ${run.steps.length} rounds; no data left the workers`);
      else if (r >= 2) setText(label, `Round ${r + 1} · F(w) = ${fval} · only gradients travel, the data stays with the workers`);
      else setText(label, `Round ${r + 1} · ` + {
        send: "the aggregator sends the current model w to every worker",
        compute: "each worker computes its gradient ∇fᵢ(w) on its own data",
        return: "workers send back gradients only; their data never leaves",
        update: "the aggregator steps along the weighted sum: w ← w − η Σ pᵢ ∇fᵢ(w)",
      }[seg.ph]);
    });

    canvas.addEventListener("pointerdown", (e) => {
      dragging = { x: e.clientX, y: e.clientY, yaw: dragYaw, pitch };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      dragYaw = dragging.yaw + (e.clientX - dragging.x) * 0.01;
      pitch = Math.max(0.2, Math.min(1.3, dragging.pitch + (e.clientY - dragging.y) * 0.01));
      if (reduced) redraw();
    });
    const endDrag = () => { dragging = null; };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    redraw();
  }

  // ---------- spatial branch and bound on a 3-D landscape ----------
  // Minimize f on [0, 1]² with Lipschitz constant L. Each box samples f on a
  // 5×5 grid; its lower bound is the smallest sample minus L times the
  // largest distance to a sample. Boxes whose bound is above the best value
  // found are pruned; the box with the lowest bound is bisected next.
  function bnb() {
    const canvas = document.getElementById("bnb-canvas");
    if (!canvas) return;
    const label = document.getElementById("bnb-label");
    const N = 32, MAX_STEPS = 46, STEP = 0.42, HOLD = 3;
    let run = null, yaw = -0.7, pitch = 0.62, dragYaw = 0, dragging = null;

    function newRun(t0) {
      const wells = [];
      for (let i = 0; i < 6; i++) {
        wells.push({
          c: [0.12 + Math.random() * 0.76, 0.12 + Math.random() * 0.76],
          a: i === 0 ? 1.15 : 0.45 + Math.random() * 0.5,
          s: 0.08 + Math.random() * 0.07,
        });
      }
      const p = Math.random() * 6.28;
      const raw = (x, y) => -wells.reduce((acc, wl) => acc + wl.a * Math.exp(-((x - wl.c[0]) ** 2 + (y - wl.c[1]) ** 2) / (2 * wl.s * wl.s)), 0)
        + 0.06 * Math.sin(7 * x + p) * Math.cos(6 * y - p);
      const G = 128;
      let lo = Infinity, hi = -Infinity;
      const grid = [];
      for (let i = 0; i <= G; i++) {
        grid.push([]);
        for (let j = 0; j <= G; j++) {
          const v = raw(i / G, j / G);
          grid[i].push(v);
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
      }
      let slope = 0;
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++)
          slope = Math.max(slope, Math.hypot(grid[i + 1][j] - grid[i][j], grid[i][j + 1] - grid[i][j]) * G);
      const f = (x, y) => (raw(x, y) - lo) / (hi - lo);
      const L = (slope / (hi - lo)) * 1.1;

      const heights = [];
      for (let i = 0; i <= N; i++) { heights.push([]); for (let j = 0; j <= N; j++) heights[i].push(f(i / N, j / N)); }

      const mk = (x0, y0, x1, y1) => {
        let cx = 0, cy = 0, fc = Infinity;
        for (let a = 0; a < 5; a++)
          for (let b = 0; b < 5; b++) {
            const x = x0 + ((a + 0.5) / 5) * (x1 - x0), y = y0 + ((b + 0.5) / 5) * (y1 - y0), v = f(x, y);
            if (v < fc) { fc = v; cx = x; cy = y; }
          }
        return { x0, y0, x1, y1, cx, cy, fc, lb: fc - L * Math.hypot(x1 - x0, y1 - y0) / 10, pruned: false };
      };
      let boxes = [mk(0, 0, 1, 1)];
      let best = boxes[0];
      const hist = [];
      for (let step = 0; step <= MAX_STEPS; step++) {
        const open = boxes.filter((b) => !b.pruned && Math.max(b.x1 - b.x0, b.y1 - b.y0) > 1 / N + 1e-9);
        const pick = open.length ? open.reduce((a, b) => (b.lb < a.lb ? b : a)) : null;
        const done = !pick || step === MAX_STEPS || pick.lb >= best.fc - 0.01;
        hist.push({ boxes: boxes.map((b) => ({ ...b })), pick: done ? null : boxes.indexOf(pick), best, done });
        if (done) break;
        const wide = pick.x1 - pick.x0 >= pick.y1 - pick.y0;
        const kids = wide
          ? [mk(pick.x0, pick.y0, (pick.x0 + pick.x1) / 2, pick.y1), mk((pick.x0 + pick.x1) / 2, pick.y0, pick.x1, pick.y1)]
          : [mk(pick.x0, pick.y0, pick.x1, (pick.y0 + pick.y1) / 2), mk(pick.x0, (pick.y0 + pick.y1) / 2, pick.x1, pick.y1)];
        boxes = boxes.filter((b) => b !== pick).concat(kids);
        for (const kid of kids) if (kid.fc < best.fc) best = kid;
        for (const b of boxes) if (b.lb > best.fc) b.pruned = true;
      }
      run = { heights, hist, t0, total: hist.length * STEP + HOLD };
    }

    // World coordinates: x, y in [−1, 1], height in [−0.45, 0.35].
    const W3 = (i, j, v) => [(i / N) * 2 - 1, (j / N) * 2 - 1, v * 0.8 - 0.45];
    let w = 0, h = 0, s = 1;
    function project(p) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = p[0] * cy - p[1] * sy, y1 = p[0] * sy + p[1] * cy, z1 = p[2];
      const depth = y1 * cp - z1 * sp, up = y1 * sp + z1 * cp;
      const k = 6 / (6 + depth);
      return { x: w / 2 + x1 * s * k, y: h * 0.5 - up * s * k, depth };
    }

    const redraw = animate(canvas, (ctx, cw, ch, t) => {
      w = cw; h = ch; s = Math.min(w / 3.2, h / 2.05);
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      yaw = -0.7 + dragYaw + (t === null ? 0 : now * 0.18);
      const local = t === null ? 14 * STEP : now - run.t0;
      const st = run.hist[Math.min(run.hist.length - 1, Math.floor(local / STEP))];

      // Which box owns each mesh cell.
      const owner = Array.from({ length: N }, () => Array(N).fill(-1));
      st.boxes.forEach((b, idx) => {
        for (let i = Math.round(b.x0 * N); i < Math.round(b.x1 * N); i++)
          for (let j = Math.round(b.y0 * N); j < Math.round(b.y1 * N); j++) owner[i][j] = idx;
      });

      const accent = hexToRgb(colors.accent), surf = hexToRgb(colors.surface);
      const muted = hexToRgb(colors.muted), cut = hexToRgb(colors.cut);
      const high = mixRgb(accent, surf, 0.78);
      const light = [-0.35, -0.55, 0.76];
      const items = [];
      const H = run.heights;
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const p00 = W3(i, j, H[i][j]), p10 = W3(i + 1, j, H[i + 1][j]), p11 = W3(i + 1, j + 1, H[i + 1][j + 1]), p01 = W3(i, j + 1, H[i][j + 1]);
          const d1 = [p11[0] - p00[0], p11[1] - p00[1], p11[2] - p00[2]], d2 = [p01[0] - p10[0], p01[1] - p10[1], p01[2] - p10[2]];
          const n = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
          const nl = Math.hypot(...n);
          const shade = 0.62 + 0.38 * Math.max(0, (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / nl);
          const hv = (H[i][j] + H[i + 1][j] + H[i + 1][j + 1] + H[i][j + 1]) / 4;
          const o = owner[i][j], box = st.boxes[o];
          let col;
          if (box.pruned) col = mixRgb(muted, surf, 0.62);
          else if (o === st.pick) col = mixRgb(cut, surf, 0.25 + 0.4 * hv);
          else col = mixRgb(accent, high, Math.min(1, hv * 1.25));
          const pts = [p00, p10, p11, p01].map(project);
          // Box boundaries follow the surface: an edge is drawn where the
          // neighbouring cell belongs to another box.
          const edge = [
            j === 0 || owner[i][j - 1] !== o,
            i === N - 1 || owner[i + 1][j] !== o,
            j === N - 1 || owner[i][j + 1] !== o,
            i === 0 || owner[i - 1][j] !== o,
          ];
          items.push({ depth: (pts[0].depth + pts[1].depth + pts[2].depth + pts[3].depth) / 4, pts, col, shade, edge, pick: o === st.pick, pruned: box.pruned });
        }
      // Evaluated box centers and the best point, depth-sorted with the mesh.
      st.boxes.forEach((b) => {
        const v = run.heights[Math.min(N, Math.round(b.cx * N))][Math.min(N, Math.round(b.cy * N))];
        const q = project([b.cx * 2 - 1, b.cy * 2 - 1, v * 0.8 - 0.45 + 0.02]);
        items.push({ depth: q.depth - 0.03, dot: q, best: b === st.best || (b.cx === st.best.cx && b.cy === st.best.cy), pruned: b.pruned });
      });
      items.sort((a, b) => b.depth - a.depth);

      for (const it of items) {
        if (it.dot) {
          ctx.beginPath();
          ctx.arc(it.dot.x, it.dot.y, it.best ? 5.5 : 2.2, 0, Math.PI * 2);
          ctx.fillStyle = it.best ? colors.cut : rgba(colors.text, it.pruned ? 0.3 : 0.8);
          ctx.fill();
          if (it.best) { ctx.strokeStyle = colors.surface; ctx.lineWidth = 1.5; ctx.stroke(); }
          continue;
        }
        const p = it.pts;
        ctx.beginPath();
        ctx.moveTo(p[0].x, p[0].y);
        for (let q = 1; q < 4; q++) ctx.lineTo(p[q].x, p[q].y);
        ctx.closePath();
        ctx.fillStyle = css(it.col, it.shade);
        ctx.fill();
        ctx.strokeStyle = css(it.col, it.shade * 0.93);
        ctx.lineWidth = 0.6;
        ctx.stroke();
        ctx.beginPath();
        for (let q = 0; q < 4; q++) {
          if (!it.edge[q]) continue;
          ctx.moveTo(p[q].x, p[q].y);
          ctx.lineTo(p[(q + 1) % 4].x, p[(q + 1) % 4].y);
        }
        ctx.strokeStyle = it.pick ? colors.cut : it.pruned ? rgba(colors.muted, 0.55) : rgba(colors.accent, 0.95);
        ctx.lineWidth = it.pick ? 2 : 1.2;
        ctx.stroke();
      }

      const pruned = st.boxes.filter((b) => b.pruned).length, open = st.boxes.length - pruned;
      setText(label, st.done
        ? `Deepest valley found · ${pruned} of ${st.boxes.length} boxes pruned without being searched`
        : `Step ${run.hist.indexOf(st) + 1} · ${open} boxes open, ${pruned} pruned · best value ${st.best.fc.toFixed(3)}`);
    });

    canvas.addEventListener("pointerdown", (e) => {
      dragging = { x: e.clientX, y: e.clientY, yaw: dragYaw, pitch };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      dragYaw = dragging.yaw + (e.clientX - dragging.x) * 0.01;
      pitch = Math.max(0.15, Math.min(1.35, dragging.pitch + (e.clientY - dragging.y) * 0.01));
      if (reduced) redraw();
    });
    const endDrag = () => { dragging = null; };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    redraw();
  }

  readColors();
  distributed();
  bnb();
})();
