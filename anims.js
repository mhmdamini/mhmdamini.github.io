// Two animations: ADMM (pieces of a fit solved locally, then stitched
// together) and spatial branch and bound on a 3-D landscape with many valleys.
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
    // Block colors for the ADMM pieces, readable in both themes.
    colors.blocks = dark.matches
      ? ["#5cc8a0", "#7aa7f0", "#c58af0", "#e8c55a"]
      : ["#0b6e4f", "#3b6fc4", "#9b4dca", "#b8860b"];
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

  // Solve M x = b by Gaussian elimination with partial pivoting.
  function solve(M, b) {
    const n = b.length, A = M.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      for (let r = c + 1; r < n; r++) {
        const f = A[r][c] / A[c][c];
        for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
      }
    }
    const x = Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = A[r][n];
      for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
      x[r] = s / A[r][r];
    }
    return x;
  }

  // ---------- ADMM: fit locally, stitch at the seams ----------
  // Four blocks each own the data on a quarter of [0, 1] and fit a cubic.
  // Consensus ADMM forces neighbouring cubics to agree in value and slope at
  // each seam:  θ_b ← argmin ½‖A_b θ − y_b‖² + ρ/2 ‖C_b θ − z_b + u_b‖²,
  //             z ← average of C θ + u at each seam,  u ← u + C θ − z.
  function admm() {
    const canvas = document.getElementById("admm-canvas");
    if (!canvas) return;
    const label = document.getElementById("admm-label");
    const B = 4, HALF = 0.5 / B, RHO = 1.2, SLOPE_W = 0.12, MAX_ITERS = 60;
    const LOCAL = 2.2, RATE = 3, HOLD = 2.8;
    const phi = (t) => [1, t, t * t, t * t * t];
    const dphi = (t) => [0, 1, 2 * t, 3 * t * t].map((v) => (v / HALF) * SLOPE_W);
    const mid = (b) => (b + 0.5) / B;
    let run = null;

    function newRun(t0) {
      const ph = Math.random() * 6.28, fr = 1 + Math.random() * 0.6, amp2 = 0.2 + Math.random() * 0.3;
      const truth = (x) => 0.8 * Math.sin(2 * Math.PI * fr * x + ph) + amp2 * Math.cos(7 * x + ph);
      const blocks = [];
      for (let b = 0; b < B; b++) {
        const pts = [];
        for (let i = 0; i < 11; i++) {
          const x = b / B + ((i + 0.2 + Math.random() * 0.6) / 11) / B;
          pts.push({ x, y: truth(x) + (Math.random() - 0.5) * 0.25 });
        }
        // Rows of C: left seam (value, slope) at t = −1, right seam at t = +1.
        const C = [], seams = [];
        if (b > 0) { C.push(phi(-1), dphi(-1)); seams.push([b - 1, 0], [b - 1, 1]); }
        if (b < B - 1) { C.push(phi(1), dphi(1)); seams.push([b, 0], [b, 1]); }
        const A = pts.map((p) => phi((p.x - mid(b)) / HALF));
        const AtA = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => A.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-6 : 0)));
        const Aty = [0, 1, 2, 3].map((i) => A.reduce((s, r, k) => s + r[i] * pts[k].y, 0));
        blocks.push({ pts, C, seams, AtA, Aty, u: C.map(() => 0) });
      }

      const thetas = [], gaps = [];
      let theta = blocks.map((bl) => solve(bl.AtA, bl.Aty)); // purely local fits
      const z = Array.from({ length: B - 1 }, () => [0, 0]);
      const Ct = (bl, th) => bl.C.map((row) => row.reduce((s, v, i) => s + v * th[i], 0));
      const seamGap = (th) => Math.max(...z.map((_, s) => Math.abs(phi(1).reduce((a, v, i) => a + v * th[s][i], 0) - phi(-1).reduce((a, v, i) => a + v * th[s + 1][i], 0))));
      const updateZ = () => {
        const acc = z.map(() => [0, 0]);
        blocks.forEach((bl, b) => Ct(bl, theta[b]).forEach((v, r) => { const [s, k] = bl.seams[r]; acc[s][k] += (v + bl.u[r]) / 2; }));
        acc.forEach((v, s) => { z[s] = v; });
      };
      updateZ();
      thetas.push(theta); gaps.push(seamGap(theta));
      for (let k = 0; k < MAX_ITERS; k++) {
        theta = blocks.map((bl) => {
          const M = bl.AtA.map((row, i) => row.map((v, j) => v + RHO * bl.C.reduce((s, r) => s + r[i] * r[j], 0)));
          const rhs = bl.Aty.map((v, i) => v + RHO * bl.C.reduce((s, r, q) => s + r[i] * (z[bl.seams[q][0]][bl.seams[q][1]] - bl.u[q]), 0));
          return solve(M, rhs);
        });
        updateZ();
        blocks.forEach((bl, b) => Ct(bl, theta[b]).forEach((v, r) => { const [s, q] = bl.seams[r]; bl.u[r] += v - z[s][q]; }));
        thetas.push(theta); gaps.push(seamGap(theta));
        if (gaps[gaps.length - 1] < 0.004) break;
      }

      const ys = blocks.flatMap((bl) => bl.pts.map((p) => p.y));
      const iters = thetas.length - 1;
      run = { blocks, thetas, gaps, iters, t0, ylo: Math.min(...ys) - 0.35, yhi: Math.max(...ys) + 0.35,
        total: LOCAL + iters / RATE + HOLD };
    }

    animate(canvas, (ctx, w, h, t) => {
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      const local = t === null ? LOCAL + 3 / RATE : now - run.t0;
      const kf = Math.max(0, Math.min(run.iters, (local - LOCAL) * RATE));
      const k = Math.floor(kf), fr = k >= run.iters ? 0 : ease(kf - k);
      const th = run.thetas[k].map((a, b) => a.map((v, i) => v + (run.thetas[Math.min(k + 1, run.iters)][b][i] - v) * fr));

      const L = 14, R = 14, T = 26, Bm = 12;
      const X = (x) => L + x * (w - L - R);
      const Y = (y) => T + ((run.yhi - y) / (run.yhi - run.ylo)) * (h - T - Bm);
      const val = (b, x) => phi((x - mid(b)) / HALF).reduce((s, v, i) => s + v * th[b][i], 0);

      // Block strips with labels.
      for (let b = 0; b < B; b++) {
        if (b % 2) {
          ctx.fillStyle = rgba(colors.muted, 0.05);
          ctx.fillRect(X(b / B), 0, X((b + 1) / B) - X(b / B), h);
        }
        ctx.fillStyle = colors.blocks[b];
        ctx.font = "600 11px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`Block ${b + 1}`, X(mid(b)), 16);
      }
      // Seams.
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = rgba(colors.muted, 0.4);
      ctx.lineWidth = 1;
      for (let s = 1; s < B; s++) {
        ctx.beginPath();
        ctx.moveTo(X(s / B), T - 4);
        ctx.lineTo(X(s / B), h - Bm);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Data.
      run.blocks.forEach((bl, b) => bl.pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(X(p.x), Y(p.y), 2.6, 0, Math.PI * 2);
        ctx.fillStyle = rgba(colors.blocks[b], 0.55);
        ctx.fill();
      }));

      // Each block's local model.
      for (let b = 0; b < B; b++) {
        ctx.beginPath();
        for (let i = 0; i <= 40; i++) {
          const x = b / B + (i / 40) / B;
          i ? ctx.lineTo(X(x), Y(val(b, x))) : ctx.moveTo(X(x), Y(val(b, x)));
        }
        ctx.strokeStyle = colors.blocks[b];
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.stroke();
      }

      // Gaps at the seams, which ADMM closes.
      for (let s = 1; s < B; s++) {
        const x = s / B, y1 = val(s - 1, x), y2 = val(s, x);
        if (Math.abs(Y(y1) - Y(y2)) > 1.5) {
          ctx.beginPath();
          ctx.moveTo(X(x), Y(y1));
          ctx.lineTo(X(x), Y(y2));
          ctx.strokeStyle = colors.cut;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        for (const y of [y1, y2]) {
          ctx.beginPath();
          ctx.arc(X(x), Y(y), 3.5, 0, Math.PI * 2);
          ctx.fillStyle = colors.cut;
          ctx.fill();
        }
      }

      const gap = run.gaps[k] + (run.gaps[Math.min(k + 1, run.iters)] - run.gaps[k]) * fr;
      if (local < LOCAL) setText(label, `Each block fits only its own data: the seams don't match (gap ${run.gaps[0].toFixed(2)})`);
      else if (k < run.iters) setText(label, `ADMM round ${k + 1} · largest gap at a seam ${gap.toFixed(3)}`);
      else setText(label, `Stitched: one smooth model from ${B} local solves after ${run.iters} rounds`);
    })();
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
  admm();
  bnb();
})();
