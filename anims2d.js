// Two 2-D animations: distributed optimization (agents on a network reach
// consensus with gradient tracking) and global optimization (a Lipschitz
// lower bound certifies the global minimum, Piyavskii–Shubert style).
(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dark = window.matchMedia("(prefers-color-scheme: dark)");
  let colors = {};

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
  }

  function rgba(hex, a) {
    const m = hex.replace("#", "");
    const n = parseInt(m.length === 3 ? m.split("").map((ch) => ch + ch).join("") : m, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  // Canvas sizing, visibility and the animation loop. render(ctx, w, h, t)
  // gets t in seconds, or null when the viewer prefers reduced motion.
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
    redraw();
  }

  function setText(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
  }

  // ---------- distributed optimization ----------
  // min sum_i f_i(x), f_i(x) = ½ (x − a_i)' Q_i (x − a_i), where agent i only
  // knows f_i and talks to its neighbours. Gradient tracking (DIGing):
  //   x ← W x − α y,   y ← W y + ∇f(x_new) − ∇f(x_old).
  function distributed() {
    const canvas = document.getElementById("dist-canvas");
    if (!canvas) return;
    const label = document.getElementById("dist-label");
    const N = 6;
    const edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3], [1, 4]];
    const deg = Array(N).fill(0);
    edges.forEach(([i, j]) => { deg[i]++; deg[j]++; });
    // Metropolis weights: symmetric and doubly stochastic.
    const W = Array.from({ length: N }, () => Array(N).fill(0));
    edges.forEach(([i, j]) => { W[i][j] = W[j][i] = 1 / (1 + Math.max(deg[i], deg[j])); });
    W.forEach((row, i) => { row[i] = 1 - row.reduce((s, v) => s + v, 0); });

    const ALPHA = 0.04, MAX_ITERS = 200, RATE = 9, HOLD = 2.2;
    let run = null;

    const mul = (Q, v) => [Q[0][0] * v[0] + Q[0][1] * v[1], Q[1][0] * v[0] + Q[1][1] * v[1]];
    const mix = (M, Z) => Z.map((_, i) => M[i].reduce((s, wij, j) => [s[0] + wij * Z[j][0], s[1] + wij * Z[j][1]], [0, 0]));

    function newRun(t0) {
      const spin = Math.random() * Math.PI * 2;
      const a = [], Q = [];
      for (let i = 0; i < N; i++) {
        const ang = spin + (2 * Math.PI * i) / N + (Math.random() - 0.5) * 0.5;
        const r = 0.55 + Math.random() * 0.25;
        a.push([r * Math.cos(ang), 0.78 * r * Math.sin(ang)]);
        const th = Math.random() * Math.PI, c = Math.cos(th), s = Math.sin(th);
        const l1 = 0.4 + Math.random() * 1.8, l2 = 0.4 + Math.random() * 1.8;
        Q.push([[c * c * l1 + s * s * l2, c * s * (l1 - l2)], [c * s * (l1 - l2), s * s * l1 + c * c * l2]]);
      }
      const grad = (i, x) => mul(Q[i], [x[0] - a[i][0], x[1] - a[i][1]]);

      // The minimizer of the sum: (Σ Q_i) x = Σ Q_i a_i.
      const S = Q.reduce((M, q) => [[M[0][0] + q[0][0], M[0][1] + q[0][1]], [M[1][0] + q[1][0], M[1][1] + q[1][1]]], [[0, 0], [0, 0]]);
      const rhs = Q.reduce((v, q, i) => { const t = mul(q, a[i]); return [v[0] + t[0], v[1] + t[1]]; }, [0, 0]);
      const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
      const xstar = [(S[1][1] * rhs[0] - S[0][1] * rhs[1]) / det, (S[0][0] * rhs[1] - S[1][0] * rhs[0]) / det];

      let X = a.map((p) => p.slice()), Y = X.map((x, i) => grad(i, x));
      const hist = [X];
      for (let k = 0; k < MAX_ITERS; k++) {
        const WX = mix(W, X), WY = mix(W, Y);
        const Xn = WX.map((p, i) => [p[0] - ALPHA * Y[i][0], p[1] - ALPHA * Y[i][1]]);
        Y = WY.map((p, i) => { const gn = grad(i, Xn[i]), go = grad(i, X[i]); return [p[0] + gn[0] - go[0], p[1] + gn[1] - go[1]]; });
        X = Xn;
        hist.push(X);
        if (X.every((p) => Math.hypot(p[0] - xstar[0], p[1] - xstar[1]) < 0.004)) break;
      }

      // Level sets of the summed objective are ellipses around x*.
      const tr = S[0][0] + S[1][1], disc = Math.sqrt(Math.max(0, (S[0][0] - S[1][1]) ** 2 / 4 + S[0][1] ** 2));
      const lam1 = tr / 2 + disc, lam2 = tr / 2 - disc;
      const phi = Math.atan2(lam1 - S[0][0], S[0][1] || 1e-12);
      const reach = Math.max(...a.map((p) => { const d = [p[0] - xstar[0], p[1] - xstar[1]]; const q = mul(S, d); return d[0] * q[0] + d[1] * q[1]; }));
      run = { a, xstar, hist, lam1, lam2, phi, reach, t0, iters: hist.length - 1 };
      run.total = run.iters / RATE + HOLD;
    }

    animate(canvas, (ctx, w, h, t) => {
      readColors();
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      const kf = t === null ? 6 : Math.min(run.iters, (now - run.t0) * RATE);
      const k = Math.floor(kf), fr = k >= run.iters ? 0 : kf - k;
      const A = run.hist[k], B = run.hist[Math.min(k + 1, run.iters)];
      const pos = A.map((p, i) => [p[0] + (B[i][0] - p[0]) * fr, p[1] + (B[i][1] - p[1]) * fr]);

      const s = Math.min(w / 2.3, h / 1.9);
      const P = (p) => [w / 2 + p[0] * s, h / 2 - p[1] * s];

      // Contours of the global objective.
      const [cx, cy] = P(run.xstar);
      ctx.lineWidth = 1;
      for (let j = 1; j <= 5; j++) {
        const level = run.reach * 0.95 * (j / 5) ** 2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, Math.sqrt(level / run.lam1) * s, Math.sqrt(level / run.lam2) * s, -run.phi, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(colors.muted, 0.22);
        ctx.stroke();
      }

      // Each agent's own minimizer.
      run.a.forEach((p) => {
        const [x, y] = P(p);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(colors.muted, 0.6);
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });

      // Trails.
      ctx.lineWidth = 1.5;
      for (let i = 0; i < N; i++) {
        ctx.beginPath();
        const from = Math.max(0, k - 24);
        for (let j = from; j <= k; j++) {
          const [x, y] = P(run.hist[j][i]);
          j === from ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        const [x, y] = P(pos[i]);
        ctx.lineTo(x, y);
        ctx.strokeStyle = rgba(colors.accent, 0.3);
        ctx.stroke();
      }

      // Network links and the messages travelling along them.
      edges.forEach(([i, j], e) => {
        const [x1, y1] = P(pos[i]), [x2, y2] = P(pos[j]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = rgba(colors.muted, 0.45);
        ctx.lineWidth = 1;
        ctx.stroke();
        if (k < run.iters) {
          const u = (k + e) % 2 ? fr : 1 - fr;
          ctx.beginPath();
          ctx.arc(x1 + (x2 - x1) * u, y1 + (y2 - y1) * u, 2.2, 0, Math.PI * 2);
          ctx.fillStyle = rgba(colors.cut, 0.85);
          ctx.fill();
        }
      });

      // The optimum of the sum.
      ctx.beginPath();
      ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
      ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
      ctx.strokeStyle = colors.cut;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Agents.
      pos.forEach((p) => {
        const [x, y] = P(p);
        ctx.beginPath();
        ctx.arc(x, y, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = colors.accent;
        ctx.fill();
        ctx.strokeStyle = colors.surface;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      const dist = Math.max(...pos.map((p) => Math.hypot(p[0] - run.xstar[0], p[1] - run.xstar[1])));
      setText(label, k >= run.iters
        ? `All ${N} agents agree on the global optimum after ${run.iters} rounds`
        : `Round ${k} · farthest agent is ${dist.toFixed(3)} from the optimum`);
    });
  }

  // ---------- global optimization ----------
  // Minimize a multimodal f on [0, 1] with Lipschitz constant L. The sawtooth
  // max_i f(x_i) − L|x − x_i| is a lower bound; sample where it is lowest.
  function globalOpt() {
    const canvas = document.getElementById("glob-canvas");
    if (!canvas) return;
    const label = document.getElementById("glob-label");
    const STEP = 0.55, HOLD = 2.5, TOL = 0.02, MAX_SAMPLES = 40;
    let run = null;

    function newRun(t0) {
      const p1 = Math.random() * 6.28, p2 = Math.random() * 6.28, c = 0.25 + Math.random() * 0.5;
      const raw = (x) => 0.55 * Math.sin(12 * x + p1) + 0.3 * Math.sin(27 * x + p2) + 1.4 * (x - c) ** 2;
      let lo = Infinity, hi = -Infinity, slope = 0, prev = raw(0);
      const G = 2000;
      for (let i = 0; i <= G; i++) {
        const v = raw(i / G);
        lo = Math.min(lo, v); hi = Math.max(hi, v);
        if (i) slope = Math.max(slope, Math.abs(v - prev) * G);
        prev = v;
      }
      const f = (x) => (raw(x) - lo) / (hi - lo);
      const L = (slope / (hi - lo)) * 1.15;

      const pts = [0, 0.5, 1].map((x) => ({ x, f: f(x) }));
      const hist = [];
      for (;;) {
        pts.sort((p, q) => p.x - q.x);
        const intervals = [];
        for (let i = 0; i + 1 < pts.length; i++) {
          const A = pts[i], B = pts[i + 1];
          const xm = Math.min(B.x, Math.max(A.x, (A.x + B.x) / 2 + (A.f - B.f) / (2 * L)));
          intervals.push({ A, B, xm, lb: (A.f + B.f) / 2 - (L * (B.x - A.x)) / 2 });
        }
        const best = pts.reduce((p, q) => (q.f < p.f ? q : p));
        const next = intervals.reduce((p, q) => (q.lb < p.lb ? q : p));
        const gap = best.f - next.lb;
        const done = gap < TOL || pts.length >= MAX_SAMPLES;
        hist.push({ pts: pts.slice(), intervals, best, next, gap, done });
        if (done) break;
        pts.push({ x: next.xm, f: f(next.xm) });
      }
      run = { f, hist, ymin: -0.6, t0, total: hist.length * STEP + HOLD };
    }

    animate(canvas, (ctx, w, h, t) => {
      readColors();
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      const local = t === null ? 6.5 * STEP : now - run.t0;
      const idx = Math.min(run.hist.length - 1, Math.floor(local / STEP));
      const u = Math.min(1, local / STEP - idx);
      const st = run.hist[idx];

      const L = 14, R = 14, T = 16, Bm = 14;
      const X = (x) => L + x * (w - L - R);
      const Y = (v) => T + ((1.06 - v) / (1.06 - run.ymin)) * (h - T - Bm);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, T - 8, w, h - T - Bm + 12);
      ctx.clip();

      // Intervals whose lower bound is above the best value found are pruned.
      st.intervals.forEach((iv) => {
        if (iv.lb <= st.best.f) return;
        ctx.fillStyle = rgba(colors.muted, 0.09);
        ctx.fillRect(X(iv.A.x), T - 6, X(iv.B.x) - X(iv.A.x), h - T - Bm + 6);
      });

      // Region between the lower bound and f: where the minimum could still hide.
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = i / 300;
        i ? ctx.lineTo(X(x), Y(run.f(x))) : ctx.moveTo(X(x), Y(run.f(x)));
      }
      for (let i = st.intervals.length - 1; i >= 0; i--) {
        const iv = st.intervals[i];
        ctx.lineTo(X(iv.B.x), Y(iv.B.f));
        ctx.lineTo(X(iv.xm), Y(iv.lb));
        ctx.lineTo(X(iv.A.x), Y(iv.A.f));
      }
      ctx.closePath();
      ctx.fillStyle = rgba(colors.accent, 0.12);
      ctx.fill();

      // f itself.
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = i / 300;
        i ? ctx.lineTo(X(x), Y(run.f(x))) : ctx.moveTo(X(x), Y(run.f(x)));
      }
      ctx.strokeStyle = rgba(colors.text, 0.75);
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // The sawtooth lower bound.
      ctx.beginPath();
      st.intervals.forEach((iv, i) => {
        if (!i) ctx.moveTo(X(iv.A.x), Y(iv.A.f));
        ctx.lineTo(X(iv.xm), Y(iv.lb));
        ctx.lineTo(X(iv.B.x), Y(iv.B.f));
      });
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Best value found (upper bound) and lowest point of the lower bound.
      const dashed = (v, color) => {
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(L, Y(v));
        ctx.lineTo(w - R, Y(v));
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      };
      dashed(st.best.f, rgba(colors.cut, 0.7));
      if (!st.done) dashed(st.next.lb, rgba(colors.accent, 0.6));

      // Samples.
      st.pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(X(p.x), Y(p.f), 2.8, 0, Math.PI * 2);
        ctx.fillStyle = colors.text;
        ctx.fill();
      });
      ctx.beginPath();
      ctx.arc(X(st.best.x), Y(st.best.f), 6, 0, Math.PI * 2);
      ctx.strokeStyle = colors.cut;
      ctx.lineWidth = 2;
      ctx.stroke();

      // The next sample rises from the lower bound to f.
      if (!st.done) {
        const fx = run.f(st.next.xm), v = st.next.lb + (fx - st.next.lb) * ease(u);
        ctx.beginPath();
        ctx.setLineDash([2, 3]);
        ctx.moveTo(X(st.next.xm), Y(st.next.lb));
        ctx.lineTo(X(st.next.xm), Y(v));
        ctx.strokeStyle = rgba(colors.cut, 0.8);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(X(st.next.xm), Y(v), 4, 0, Math.PI * 2);
        ctx.fillStyle = colors.cut;
        ctx.fill();
      }

      ctx.restore();
      setText(label, st.done
        ? `Global minimum certified after ${st.pts.length} samples · gap ${st.gap.toFixed(3)}`
        : `Sample ${st.pts.length + 1} · gap between bounds ${st.gap.toFixed(3)}`);
    });
  }

  readColors();
  distributed();
  globalOpt();
})();
