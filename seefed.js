// SeEFed with an unknown graph and restricted communication: a drone (the
// mobile aggregator) reaches only the farms inside its radius, caches their
// models (never their data), rebuilds the visited farm's model as a convex mix
// of similar cached models, learns those similarity edges, and picks its next
// stop among nearby farms by Thompson sampling.
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
      border: get("--border", "#e4e2dd"),
      cut: get("--cut", "#d9822b"),
    };
    // One color per hidden similarity group, readable in both themes.
    colors.groups = dark.matches ? ["#7aa7f0", "#c58af0", "#e8c55a"] : ["#3b6fc4", "#9b4dca", "#b8860b"];
  }

  function rgba(hex, a) {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r},${g},${b},${a})`;
  }

  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const lerp = (a, b, t) => a.map((v, k) => v + (b[k] - v) * t);

  function gauss() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

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

  function seefed() {
    const canvas = document.getElementById("seefed-canvas");
    if (!canvas) return;
    const label = document.getElementById("seefed-label");
    const N = 8, R = 0.85, MAX_STEPS = 20, HOLD = 3.5, FAST = 0.42, HOVER = 0.62;
    const PHASES = [["scan", 1.0], ["aggregate", 1.1], ["choose", 1.0], ["fly", 0.9]];
    let run = null, yaw = -0.35, pitch = 0.82, dragYaw = 0, dragging = null;

    // Farms on a jittered grid; similarity groups are shuffled so that similar
    // farms are usually not physically close.
    function layout() {
      for (;;) {
        const cells = [];
        for (let gx = 0; gx < 4; gx++) for (let gy = 0; gy < 3; gy++) cells.push([gx, gy]);
        cells.sort(() => Math.random() - 0.5);
        const pos = cells.slice(0, N).map(([gx, gy]) => [
          -0.95 + gx * 0.63 + (Math.random() - 0.5) * 0.3,
          -0.7 + gy * 0.7 + (Math.random() - 0.5) * 0.3,
        ]);
        const near = pos.map((p, i) => pos.map((_, j) => j).filter((j) => j !== i && Math.hypot(p[0] - pos[j][0], p[1] - pos[j][1]) <= R));
        const seen = new Set([0]), stack = [0];
        while (stack.length) near[stack.pop()].forEach((j) => { if (!seen.has(j)) { seen.add(j); stack.push(j); } });
        if (seen.size === N && near.every((l) => l.length >= 2)) return { pos, near };
      }
    }

    function newRun(t0) {
      const { pos, near } = layout();
      const group = Array.from({ length: N }, (_, i) => i % 3).sort(() => Math.random() - 0.5);
      // Hidden similarity graph: each farm is a convex mix of one or two farms of its group.
      const parents = pos.map((_, i) => {
        const same = pos.map((_, j) => j).filter((j) => j !== i && group[j] === group[i]).sort(() => Math.random() - 0.5);
        return same.slice(0, Math.random() < 0.5 ? 1 : 2).map((j) => ({ j, w: 0.35 + Math.random() * 0.65 }));
      });
      const totalEdges = parents.reduce((s, p) => s + p.length, 0);

      // Thompson-sampling statistics: visits and mean reward per farm.
      const visits = new Array(N).fill(0), meanR = new Array(N).fill(0);
      const cache = new Set(), learned = new Map(), steps = [];
      let seed = Math.floor(Math.random() * N);
      for (let k = 0; k < MAX_STEPS; k++) {
        const visible = [seed, ...near[seed]];
        visible.forEach((j) => cache.add(j));
        const avail = parents[seed].filter((e) => cache.has(e.j));
        const wsum = avail.reduce((s, e) => s + e.w, 0);
        const lambda = avail.map((e) => ({ j: e.j, l: e.w / wsum }));
        const before = learned.size;
        lambda.forEach((e) => learned.set(`${e.j}>${seed}`, { from: e.j, to: seed, l: e.l }));
        // Reward: the loss improvement of this visit. It is large when the farm
        // gained new similar neighbors and small once its model has settled.
        const gain = learned.size > before ? 0.85 : avail.length < parents[seed].length ? 0.55 : 0.02;
        const r = Math.max(0, Math.min(1, gain + 0.06 * gauss()));
        visits[seed]++;
        meanR[seed] += (r - meanR[seed]) / visits[seed];
        // One posterior draw per reachable farm; unvisited farms are uncertain.
        const draws = near[seed].map((j) => {
          const m = (0.7 + visits[j] * meanR[j]) / (1 + visits[j]), sd = 0.25 / Math.sqrt(1 + visits[j]);
          return { j, v: Math.max(0.04, Math.min(1, m + sd * gauss())) };
        });
        const next = draws.reduce((a, b) => (b.v > a.v ? b : a)).j;
        steps.push({ seed, visible, cached: new Set(cache), lambda, learned: learned.size, draws, next });
        seed = next;
        if (learned.size === totalEdges && visits.every((v) => v > 0)) break;
      }

      const segs = [];
      let t = 0;
      steps.forEach((st, k) => PHASES.forEach(([ph, len]) => {
        if (ph === "aggregate" && !st.lambda.length) len *= 0.5;
        const d = len * (k < 2 ? 1 : FAST);
        segs.push({ k, ph, t0: t, len: d });
        t += d;
      }));
      run = { pos, near, group, totalEdges, steps, segs, t0, total: t + HOLD };
    }

    let w = 0, h = 0, s = 1;
    function project(q) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = q[0] * cy - q[1] * sy, y1 = q[0] * sy + q[1] * cy, z1 = q[2];
      const depth = y1 * cp - z1 * sp, up = y1 * sp + z1 * cp;
      const k = 6 / (6 + depth);
      return { x: w / 2 + x1 * s * k, y: h * 0.56 - up * s * k, depth };
    }
    const ground = (p) => project([p[0], p[1], 0]);

    function polygon(ctx, pts) {
      ctx.beginPath();
      pts.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
    }

    // A learned edge j → i: a curve on the ground with an arrowhead at i.
    function edge(ctx, from, to, width, color, alpha, grow = 1) {
      const A = ground(from), B = ground(to);
      const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2, dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy);
      if (len < 4 || alpha <= 0) return;
      const C = { x: mx - dy * 0.18, y: my + dx * 0.18 };
      const at = (u) => ({ x: (1 - u) ** 2 * A.x + 2 * (1 - u) * u * C.x + u * u * B.x, y: (1 - u) ** 2 * A.y + 2 * (1 - u) * u * C.y + u * u * B.y });
      const end = Math.min(grow, 1 - 11 / len);
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      for (let u = 0.05; u <= end + 1e-9; u += 0.05) { const p = at(Math.min(u, end)); ctx.lineTo(p.x, p.y); }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.stroke();
      if (grow >= 1) {
        const P = at(end), Q = at(end - 0.04), ux = P.x - Q.x, uy = P.y - Q.y, n = Math.hypot(ux, uy) || 1;
        const hx = (ux / n) * 7, hy = (uy / n) * 7;
        ctx.beginPath();
        ctx.moveTo(P.x + hx * 0.4, P.y + hy * 0.4);
        ctx.lineTo(P.x - hx - hy * 0.6, P.y - hy + hx * 0.6);
        ctx.lineTo(P.x - hx + hy * 0.6, P.y - hy - hx * 0.6);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drone(ctx, q, spin) {
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.6;
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b]) => {
        const rx = q.x + a * 11, ry = q.y + b * 6;
        ctx.beginPath();
        ctx.moveTo(q.x, q.y);
        ctx.lineTo(rx, ry);
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(rx, ry, 7, 2.2, spin * a * b, 0, Math.PI * 2);
        ctx.fillStyle = rgba(colors.muted, 0.25);
        ctx.fill();
        ctx.stroke();
      });
      ctx.beginPath();
      ctx.roundRect(q.x - 6, q.y - 4, 12, 8, 3);
      ctx.fillStyle = colors.accent;
      ctx.fill();
      ctx.strokeStyle = colors.surface;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    const redraw = animate(canvas, (ctx, cw, ch, t) => {
      w = cw; h = ch; s = Math.min(w / 3.3, h / 2.5);
      const now = t === null ? 0 : t;
      if (!run || now - run.t0 > run.total) newRun(now);
      yaw = -0.35 + dragYaw + (t === null ? 0 : Math.sin(now * 0.15) * 0.25);
      const local = t === null ? run.segs[Math.min(9, run.segs.length - 1)].t0 + 0.6 : now - run.t0;
      const seg = run.segs.find((g) => local >= g.t0 && local < g.t0 + g.len);
      const u = seg ? Math.min(1, (local - seg.t0) / seg.len) : 1;
      const done = !seg;
      const k = seg ? seg.k : run.steps.length - 1;
      const st = run.steps[k];
      const { pos, group } = run;

      // Ground.
      const corners = [[-1.35, -1.05], [1.35, -1.05], [1.35, 1.05], [-1.35, 1.05]].map(ground);
      polygon(ctx, corners);
      ctx.fillStyle = rgba(colors.accent, 0.07);
      ctx.fill();
      ctx.strokeStyle = rgba(colors.accent, 0.3);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.strokeStyle = rgba(colors.accent, 0.08);
      for (let g = -1.35 + 0.27; g < 1.35; g += 0.27) {
        const a = ground([g, -1.05]), b = ground([g, 1.05]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }

      // Physical links: which farms can hear each other through the drone.
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = rgba(colors.muted, 0.28);
      run.near.forEach((l, i) => l.forEach((j) => {
        if (j < i) return;
        const a = ground(pos[i]), b = ground(pos[j]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }));
      ctx.setLineDash([]);

      // Drone position: above the seed, moving to the next one while flying.
      const fly = seg && seg.ph === "fly" ? ease(u) : done ? 0 : 0;
      const dpos = lerp(pos[st.seed], pos[st.next], fly);

      // Communication radius around the drone.
      if (!done) {
        const rr = R * (seg.ph === "scan" ? ease(Math.min(1, u / 0.5)) : 1);
        const ring = [];
        for (let a = 0; a < 48; a++) ring.push(ground([dpos[0] + rr * Math.cos((a / 48) * Math.PI * 2), dpos[1] + rr * Math.sin((a / 48) * Math.PI * 2)]));
        polygon(ctx, ring);
        ctx.fillStyle = rgba(colors.accent, 0.1);
        ctx.fill();
        ctx.strokeStyle = rgba(colors.accent, 0.55);
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }

      // Learned similarity edges so far, then this step's new ones growing.
      const learnedBefore = new Map();
      for (let q = 0; q < k + (seg && seg.ph !== "scan" ? 1 : 0) + (done ? 1 : 0); q++)
        run.steps[q].lambda.forEach((e) => learnedBefore.set(`${e.j}>${run.steps[q].seed}`, { from: e.j, to: run.steps[q].seed, l: e.l }));
      const fresh = seg && seg.ph === "aggregate" ? new Set(st.lambda.map((e) => `${e.j}>${st.seed}`)) : new Set();
      learnedBefore.forEach((e, key) => {
        const col = colors.groups[group[e.to]];
        const grow = fresh.has(key) ? ease(u) : 1;
        edge(ctx, pos[e.from], pos[e.to], 1.2 + e.l * 4, col, fresh.has(key) ? 0.95 : 0.6, grow);
      });

      // Farms, back to front: a disk, a small data stack and the cache tick.
      const order = pos.map((p, i) => ({ i, q: ground(p) })).sort((a, b) => b.q.depth - a.q.depth);
      const cachedNow = seg && seg.ph === "scan" && u < 0.7 ? (k ? run.steps[k - 1].cached : new Set()) : st.cached;
      const inRange = new Set(done ? [] : st.visible);
      ctx.font = "600 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      order.forEach(({ i, q }) => {
        const col = colors.groups[group[i]];
        for (let b = 0; b < 3; b++) {
          ctx.fillStyle = rgba(col, 0.45);
          ctx.fillRect(q.x + 9, q.y - 9 + b * 5, 7, 3.5);
        }
        ctx.beginPath();
        ctx.arc(q.x, q.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.strokeStyle = inRange.has(i) ? colors.accent : colors.surface;
        ctx.lineWidth = inRange.has(i) ? 2.5 : 1.5;
        ctx.stroke();
        ctx.fillStyle = colors.surface;
        ctx.fillText(String(i + 1), q.x, q.y + 3.5);
        if (cachedNow.has(i)) {
          ctx.beginPath();
          ctx.arc(q.x - 9, q.y - 8, 3, 0, Math.PI * 2);
          ctx.fillStyle = colors.text;
          ctx.fill();
        }
      });

      // Models (not data) travel up to the drone during the scan.
      const top = [dpos[0], dpos[1], HOVER];
      if (seg && seg.ph === "scan" && u > 0.35) {
        const v = ease((u - 0.35) / 0.65);
        st.visible.forEach((j) => {
          const q = project(lerp([pos[j][0], pos[j][1], 0.02], top, v));
          ctx.beginPath();
          ctx.arc(q.x, q.y, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = colors.groups[group[j]];
          ctx.fill();
          ctx.strokeStyle = colors.surface;
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }

      // Thompson sampling: one posterior draw per reachable farm.
      if (seg && (seg.ph === "choose" || seg.ph === "fly")) {
        const v = seg.ph === "choose" ? ease(Math.min(1, u / 0.6)) : 1;
        const fade = seg.ph === "fly" ? 1 - u : 1;
        st.draws.forEach((d) => {
          const base = project([pos[d.j][0], pos[d.j][1], 0.03]), tip = project([pos[d.j][0], pos[d.j][1], 0.03 + d.v * 0.85 * v]);
          const win = d.j === st.next && (seg.ph === "fly" || u > 0.6);
          ctx.globalAlpha = fade;
          ctx.beginPath();
          ctx.moveTo(base.x, base.y - 9);
          ctx.lineTo(tip.x, tip.y - 9);
          ctx.strokeStyle = win ? colors.cut : rgba(colors.muted, 0.75);
          ctx.lineWidth = 6;
          ctx.lineCap = "butt";
          ctx.stroke();
          ctx.globalAlpha = 1;
        });
      }

      // The drone and its tether to the ground.
      const dq = project(top), dg = ground(dpos);
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(dg.x, dg.y); ctx.lineTo(dq.x, dq.y);
      ctx.strokeStyle = rgba(colors.muted, 0.5);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      drone(ctx, dq, now * 9);

      // Legend.
      ctx.font = "400 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = colors.muted;
      ctx.fillText("color = hidden similarity group", 12, 18);
      ctx.beginPath(); ctx.arc(15, 31, 3, 0, Math.PI * 2); ctx.fillStyle = colors.text; ctx.fill();
      ctx.fillStyle = colors.muted;
      ctx.fillText(`model cached (${cachedNow.size}/${N})`, 23, 34);

      const nLearned = learnedBefore.size;
      if (done) setText(label, (nLearned === run.totalEdges ? `Learned all ${run.totalEdges} similarity edges in ${run.steps.length} stops` : `After ${run.steps.length} stops: learned ${nLearned} of ${run.totalEdges} similarity edges`) + "; similar clients are often far apart, and no data left a client");
      else if (k >= 2) setText(label, `Stop ${k + 1} · client ${st.seed + 1} · ${st.visible.length} clients in range · learned ${nLearned} of ${run.totalEdges} similarity edges`);
      else setText(label, `Stop ${k + 1} · ` + {
        scan: "the drone reaches only clients inside its radius; their models, not data, join its cache",
        aggregate: st.lambda.length
          ? `client ${st.seed + 1}'s target is a convex mix of similar cached models: θᵍ = Σ λⱼ θⱼ`
          : `no similar client is cached yet, so client ${st.seed + 1} waits`,
        choose: "Thompson sampling: one random draw of each nearby client's usefulness",
        fly: `the tallest draw wins: fly to client ${st.next + 1}`,
      }[seg.ph]);
    });

    canvas.addEventListener("pointerdown", (e) => {
      dragging = { x: e.clientX, y: e.clientY, yaw: dragYaw, pitch };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      dragYaw = dragging.yaw + (e.clientX - dragging.x) * 0.01;
      pitch = Math.max(0.35, Math.min(1.35, dragging.pitch + (e.clientY - dragging.y) * 0.01));
      if (reduced) redraw();
    });
    const endDrag = () => { dragging = null; };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    redraw();
  }

  readColors();
  seefed();
})();
