/* Explanatory two-state rotation, using the real-block convention in lrnn. */
(() => {
  'use strict';
  const get = (id) => document.getElementById(id);
  const play = get('play');
  const next = get('step');
  const angle = get('angle');
  const total = 60;
  let tick = 0;
  let running = false;
  let frame = null;
  let lastUpdate = null;

  const path = (count, omega) => Array.from({length: count + 1}, (_, i) => {
    const x = 362 + (506 * i / total);
    const y = 149 - 88 * Math.cos(i * omega);
    return `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');

  function draw() {
    const omega = Number(angle.value);
    // J = [[cos(w), sin(w)], [-sin(w), cos(w)]], z(0) = [1, 0].
    const z1 = Math.cos(tick * omega);
    const z2 = -Math.sin(tick * omega);
    const x = 137 + 88 * z1;
    const y = 149 - 88 * z2;
    get('state-radius').setAttribute('x2', x);
    get('state-radius').setAttribute('y2', y);
    get('state-point').setAttribute('cx', x);
    get('state-point').setAttribute('cy', y);
    get('state-projection').setAttribute('x1', x);
    get('state-projection').setAttribute('y1', y);
    get('state-projection').setAttribute('x2', x);
    get('state-projection').setAttribute('y2', 149);
    get('state-label').textContent = `z₁ = ${z1.toFixed(2)}, z₂ = ${z2.toFixed(2)}`;
    get('wave-guide').setAttribute('d', path(total, omega));
    get('wave-trace').setAttribute('d', path(tick, omega));
    get('wave-point').setAttribute('cx', 362 + 506 * tick / total);
    get('wave-point').setAttribute('cy', 149 - 88 * z1);
    get('angle-value').textContent = `${omega.toFixed(2)} rad`;
    get('step-value').textContent = `Step ${tick} / ${total}`;
    next.disabled = tick >= total;
  }

  function stop() {
    running = false;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    lastUpdate = null;
    play.textContent = tick >= total ? 'Replay ▶' : 'Play ▶';
    play.setAttribute('aria-label', tick >= total ? 'Replay rotation' : 'Play rotation');
  }

  function animate(now) {
    if (!running) return;
    if (lastUpdate === null) lastUpdate = now;
    if (now - lastUpdate >= 140) {
      tick += 1;
      lastUpdate = now;
      draw();
      if (tick >= total) { stop(); return; }
    }
    frame = requestAnimationFrame(animate);
  }

  play.addEventListener('click', () => {
    if (running) { stop(); return; }
    if (tick >= total) { tick = 0; draw(); }
    running = true;
    play.textContent = 'Pause Ⅱ';
    play.setAttribute('aria-label', 'Pause rotation');
    frame = requestAnimationFrame(animate);
  });
  next.addEventListener('click', () => { tick = Math.min(tick + 1, total); stop(); draw(); });
  get('reset').addEventListener('click', () => { tick = 0; stop(); draw(); });
  angle.addEventListener('input', () => { tick = 0; stop(); draw(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  draw();

  const copy = get('copy-citation');
  if (navigator.clipboard && window.isSecureContext) {
    copy.hidden = false;
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(get('bibtex').textContent);
        get('copy-status').textContent = 'BibTeX copied.';
      } catch {
        get('copy-status').textContent = 'Copy is unavailable. Select and copy the citation below.';
      }
    });
  }
})();
