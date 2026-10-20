// Info icons: the operation hints of the panels, kept out of the way until asked for — one floating box, shown on
// hover or keyboard focus of an .info button (data-tip, paragraphs split at "\n"), pinned by a click or tap, closed
// by Esc, a click elsewhere or scrolling.
export function initInfo() {
  const tip = document.getElementById('infotip');
  if (!tip) return;
  let cur = null, pinned = false;
  const render = (el) => {
    tip.replaceChildren(...String(el.dataset.tip || '').split(/\\n|\n/).map((t) => { const p = document.createElement('p'); p.textContent = t.trim(); return p; }));
  };
  const place = (el) => {
    tip.hidden = false;
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    let x = r.left + r.width / 2 - 22, y = r.bottom + 7;
    if (x + w > vw - 8) x = vw - 8 - w;
    if (x < 8) x = 8;
    if (y + h > vh - 8) y = r.top - 7 - h;
    tip.style.left = `${Math.round(x)}px`; tip.style.top = `${Math.round(Math.max(8, y))}px`;
  };
  const show = (el) => {
    if (cur && cur !== el) cur.classList.remove('on');
    cur = el; render(el); place(el); el.classList.add('on'); el.setAttribute('aria-describedby', 'infotip');
  };
  const hide = () => {
    if (cur) { cur.classList.remove('on'); cur.removeAttribute('aria-describedby'); }
    cur = null; pinned = false; tip.hidden = true;
  };
  const infoOf = (t) => (t && t.closest ? t.closest('.info') : null);
  document.addEventListener('pointerover', (e) => { const el = infoOf(e.target); if (el && !pinned) show(el); });
  document.addEventListener('pointerout', (e) => { const el = infoOf(e.target); if (el && el === cur && !pinned && !el.contains(e.relatedTarget)) hide(); });
  document.addEventListener('focusin', (e) => { const el = infoOf(e.target); if (el) show(el); });
  document.addEventListener('focusout', (e) => { const el = infoOf(e.target); if (el && el === cur && !pinned) hide(); });
  document.addEventListener('click', (e) => {
    const el = infoOf(e.target);
    if (el) { e.preventDefault(); e.stopPropagation(); if (pinned && cur === el) hide(); else { show(el); pinned = true; } return; }
    if (pinned) hide();
  }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && cur) hide(); });
  window.addEventListener('scroll', () => { if (cur) hide(); }, true);
  window.addEventListener('resize', () => { if (cur) hide(); });
}
