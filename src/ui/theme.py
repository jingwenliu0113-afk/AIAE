"""The interface's look, in one place both cores read.

Moved out of :mod:`src.ui.render` when BrickNet became the core. That module is
the *old* core's two-page interface: it is frozen, it has no entry point any
more, and the new core must not import its presentation code just to get a
stylesheet -- a dependency in that direction is how a frozen module ends up
being edited.

The look is the bright-toybox one: a cool near-white ice-blue ground, a single
cobalt as the only action colour, hairline borders, large radii and drawn
studs. It carries no third-party brand mark, wordmark, logo or figure of any
kind, and none is referenced: this project is not affiliated with any brick
manufacturer and the interface must not imply otherwise.

Colour is never the only carrier of meaning: every check prints ``pass`` /
``FAIL`` / ``n/a`` as text next to its shape, so the page reads the same
without hue.
"""

from __future__ import annotations

CSS = """
/* The bright-toybox design system, ported from the approved visual reference.
   Its three defining choices, all of which the earlier chunky style broke:
   weight 500 rather than 700, hairline 1px borders rather than 3px, and no
   dotted stud band on the containers -- the toy character comes from the
   brandmark, the board tab, the floating bricks and the part swatches. */
:root {
  --paper:#EDF5FF; --paper-2:#DCE9FB; --card:#FCFDFF;
  --ink:#17253B; --ink-soft:#52657F;
  --line:#A7BDDB; --line-deep:#A7BDDB; --rule:#C5D5EA; --shadow:#BED0E9;
  --accent:#2F68D2; --accent-deep:#19499B; --on-accent:#FCFDFF;
  --accent-soft:#8FB2EF; --accent-mid:#5688DF;
  --ok:#1B6B44; --ok-bg:#E2F1E9;
  --bad:#B3261E; --bad-bg:#FBEAE9;
  --warn:#8A4B00; --warn-bg:#FAEFDC;
  --na:#52657F;  --na-bg:#E7EEF8;
  --radius:24px; --radius-sm:14px; --radius-xs:10px; --lift:5px;
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:ui-rounded,"SF Pro Rounded","PingFang TC","Noto Sans TC",
    system-ui,-apple-system,"Segoe UI",Roboto,"Hiragino Sans",
    "Microsoft JhengHei",sans-serif;
  font-size:16px; line-height:1.55;
}
h1, h2, h3 { font-weight:500; margin:0; letter-spacing:-.02em; }
a { color:var(--accent-deep); }
:focus-visible { outline:3px solid var(--accent); outline-offset:3px;
  border-radius:4px; }
.skip {
  position:absolute; left:-9999px; top:0; background:var(--card);
  padding:12px 16px; border:1px solid var(--line);
  border-radius:var(--radius-sm); z-index:99;
  box-shadow:0 4px 0 var(--shadow);
}
.skip:focus { left:14px; top:14px; }

/* The dotted band is gone on purpose; the elements stay so the templates and
   their tests are untouched. */
.studs { display:none; }

.wrap { max-width:1040px; margin:0 auto; padding:28px 26px 44px; }
@media (max-width:760px) { .wrap { padding:24px 14px 32px; } }

/* --- header ------------------------------------------------------------ */
header.top { background:var(--card); border-bottom:1px solid var(--line); }
header.top .wrap { display:flex; align-items:center; justify-content:space-between;
  gap:14px 24px; flex-wrap:wrap; padding-top:14px; padding-bottom:16px; }
.brandline { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.brandline h1 { display:flex; align-items:center; gap:11px; font-size:19px; }
/* A plain blue brick as the mark: a body with an inset bottom shade, the same
   drawing the part swatches use. Nothing is drawn on its face -- two dots on a
   rounded rectangle read as eyes, and no part of this interface may look like
   a face. */
.brandline h1::before {
  content:""; flex:0 0 auto; width:44px; height:34px; border-radius:11px;
  background:var(--accent);
  box-shadow:inset 0 -7px 0 rgba(18,59,139,.22), 0 4px 0 var(--accent-deep);
}
.brandline .tag { color:var(--ink-soft); font-size:14px; }

/* --- the three-step nav, drawn as stud-topped tabs ---------------------- */
ol.steps { display:flex; gap:9px; list-style:none; margin:0;
  padding:0; flex-wrap:wrap; }
ol.steps li {
  position:relative; display:flex; align-items:center; min-height:42px;
  padding:8px 16px; border:1px solid var(--line); border-radius:12px;
  background:var(--paper-2); color:var(--ink-soft); font-size:15px;
  box-shadow:0 4px 0 var(--shadow);
}
ol.steps li::before, ol.steps li::after {
  content:""; position:absolute; top:-7px; width:13px; height:7px;
  border:1px solid var(--line); border-bottom:0; border-radius:5px 5px 0 0;
  background:inherit;
}
ol.steps li::before { left:16px; }
ol.steps li::after { right:16px; }
ol.steps li .n { display:none; }
ol.steps li[aria-current="step"] {
  background:var(--accent); border-color:var(--accent); color:var(--on-accent);
  transform:translateY(-2px); box-shadow:0 5px 0 var(--accent-deep);
}
ol.steps li[aria-current="step"]::before,
ol.steps li[aria-current="step"]::after { border-color:var(--accent); }

/* --- boards ------------------------------------------------------------ */
.card {
  position:relative; background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius); margin:26px 0;
  box-shadow:0 13px 30px rgba(57,91,140,.13), 0 6px 0 var(--shadow);
}
/* The board's own stud tab, centred on the top edge. */
.card::before {
  content:""; position:absolute; left:50%; top:-8px; width:104px; height:14px;
  transform:translateX(-50%); background:var(--accent);
  border-radius:8px 8px 3px 3px;
  box-shadow:inset 0 -4px 0 rgba(18,59,139,.18);
}
.card .body { padding:28px; }
@media (max-width:760px) { .card .body { padding:24px 18px; } }
.card h2 { font-size:25px; }
.card h3 { font-size:17px; margin:26px 0 8px; }
.card .lede { margin:10px 0 0; color:var(--ink-soft); font-size:16px; }
.card .lede + * { margin-top:20px; }

.grid2 { display:grid; gap:22px; grid-template-columns:1fr;
  align-items:start; }
@media (min-width:760px) {
  .grid2 { grid-template-columns:minmax(0,1.25fr) minmax(220px,.75fr); }
}

/* --- forms ------------------------------------------------------------- */
fieldset { border:1px solid var(--line); border-radius:var(--radius-sm);
  margin:0 0 18px; padding:6px 18px 18px; background:var(--card); }
fieldset[disabled] { opacity:.5; }
legend { font-weight:500; padding:0 8px; font-size:15px; }
label { display:block; font-weight:500; margin:22px 0 8px; font-size:14px; }
label:first-child { margin-top:0; }
.hint { color:var(--ink-soft); font-size:14px; font-weight:400;
  margin:8px 0 0; line-height:1.55; }
input[type=text], input[type=number], input[type=file], textarea, select {
  display:block; width:100%; font:inherit; color:var(--ink);
  background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius-sm); padding:13px 15px; min-height:48px;
  box-shadow:inset 0 2px 0 rgba(47,104,210,.05);
}
textarea { min-height:112px; resize:vertical; line-height:1.5; }
::placeholder { color:#6C788A; }
input[type=number] { max-width:14ch; }
.parts input[type=number] { max-width:none; }

.modes { display:grid; gap:11px; }
.modes label.mode {
  display:flex; gap:12px; align-items:flex-start; font-weight:400;
  margin:0; font-size:16px; min-height:48px; cursor:pointer;
  border:1px solid var(--line); border-radius:16px; padding:16px 17px;
  background:var(--card); box-shadow:0 4px 0 var(--shadow);
}
.modes label.mode:hover { border-color:var(--accent-soft); }
.modes input[type=radio], .modes input[type=checkbox] {
  width:20px; height:20px; margin-top:2px; accent-color:var(--accent);
  flex:0 0 auto; cursor:pointer; }
.modes .mode strong { display:block; font-weight:500; font-size:17px; }
.modes .mode span { color:var(--ink-soft); font-size:15px; line-height:1.45; }
.modes .mode strong + span { display:block; margin-top:6px; }

/* --- the part tray ------------------------------------------------------ */
/* ``minmax(0,1fr)`` rather than ``1fr``: a track sized 1fr will not shrink
   below its content's minimum, so in the two-page UI's narrow right-hand
   column the last two cells escaped the grid. The swatch shrinks with the
   track instead. */
.parts { display:grid; gap:11px;
  grid-template-columns:repeat(2,minmax(0,1fr)); }
@media (min-width:560px) {
  .parts { grid-template-columns:repeat(4,minmax(0,1fr)); }
}
.pair { display:grid; gap:14px; grid-template-columns:1fr; }
@media (min-width:560px) { .pair { grid-template-columns:1fr 1fr; } }
.part { padding:12px 10px; border:1px solid var(--line);
  border-radius:13px; background:var(--card); text-align:center; }
.part .head { display:flex; flex-direction:column; align-items:center; }
.part label { margin:0 0 8px; font-size:15px; order:2; }
.part input { padding:10px 8px; text-align:center; }
/* A brick body with an inset bottom shade. No studs on the face, so it can
   never read as eyes; the eight fill colours are the ones the 3-D preview
   paints with, so a part is the same colour in both places. */
.chip {
  display:block; order:1; width:62px; max-width:100%; height:31px;
  margin:4px auto 9px;
  border:1px solid var(--accent-deep); border-radius:7px;
  box-shadow:inset 0 -7px 0 rgba(18,59,139,.15);
}

/* --- buttons ----------------------------------------------------------- */
.actions { display:flex; gap:10px; flex-wrap:wrap; align-items:center;
  justify-content:flex-end; margin-top:26px; }
.actionbar .body { display:flex; gap:16px; align-items:center;
  flex-wrap:wrap; justify-content:space-between; }
.actionbar .why { margin:0; color:var(--ink-soft); font-size:15px;
  max-width:62ch; }
.btn {
  display:inline-flex; align-items:center; justify-content:center; gap:8px;
  cursor:pointer; font:inherit; font-weight:500; text-decoration:none;
  min-height:48px; padding:11px 20px; border-radius:13px;
  border:1px solid var(--accent); background:var(--accent);
  color:var(--on-accent); box-shadow:0 5px 0 var(--accent-deep);
  transition:transform .16s ease, box-shadow .16s ease;
}
.btn:active { transform:translateY(3px); box-shadow:0 1px 0 var(--accent-deep); }
.btn.ghost { background:var(--card); color:var(--ink);
  border-color:var(--line); box-shadow:0 4px 0 var(--shadow); }
.btn.ghost:active { box-shadow:0 1px 0 var(--shadow); }
.btn[aria-disabled="true"] { background:var(--na-bg); color:var(--na);
  border-color:var(--line); box-shadow:none; cursor:not-allowed;
  transform:none; }

/* --- notices ----------------------------------------------------------- */
.note { border:1px solid var(--line); border-radius:16px;
  padding:15px 17px; margin:0 0 18px; background:var(--card);
  font-size:15px; line-height:1.55; }
.note.bad { border-color:#E2B4B0; background:var(--bad-bg); }
.note.ok { border-color:#AFD4C0; background:var(--ok-bg); }
.note.warn { border-color:#E0C79A; background:var(--warn-bg); }
.note.flat { background:var(--paper-2); }
.note strong { display:block; font-weight:500; font-size:17px;
  margin-bottom:5px; }
.note.bad strong { color:var(--bad); }
.note.ok strong { color:var(--ok); }
.note.warn strong { color:var(--warn); }
.note ul { margin:8px 0 0; padding-left:20px; }
.note li { margin-bottom:4px; }

/* --- tables ------------------------------------------------------------ */
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch;
  border:1px solid var(--line); border-radius:16px; background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:15px;
  min-width:560px; }
caption { text-align:left; padding:13px 16px; color:var(--ink-soft);
  font-size:14px; }
th, td { text-align:left; padding:11px 16px;
  border-bottom:1px solid var(--rule); vertical-align:top; }
th { background:var(--paper-2); font-size:14px; font-weight:500;
  color:var(--ink-soft); white-space:nowrap; }
tbody tr:last-child td { border-bottom:0; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.mono, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
  monospace; font-size:14px; }
tr.picked td { background:var(--ok-bg); }

/* --- verdict pills ----------------------------------------------------- */
.pill { display:inline-flex; align-items:center; gap:6px; font-weight:500;
  font-size:13px; padding:4px 11px; border-radius:999px;
  border:1px solid currentColor; white-space:nowrap; }
.pill svg { width:12px; height:12px; }
.pill.pass { color:var(--ok); background:var(--ok-bg); }
.pill.fail { color:var(--bad); background:var(--bad-bg); }
.pill.na   { color:var(--na);  background:var(--na-bg); }

.checks { display:grid; gap:9px; grid-template-columns:1fr; margin:0; }
@media (min-width:720px) { .checks { grid-template-columns:1fr 1fr; } }
.checks div { display:flex; gap:10px; align-items:baseline; flex-wrap:wrap;
  border:1px solid var(--line); border-radius:12px; padding:10px 14px;
  background:var(--card); }
.checks .k { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:13px; }
.checks .g { color:var(--ink-soft); font-size:14px; }

.kv { display:grid; grid-template-columns:auto 1fr; gap:6px 16px;
  margin:0; font-size:15px; }
.kv dt { color:var(--ink-soft); white-space:nowrap; }
.kv dd { margin:0; overflow-wrap:anywhere; }

figure.preview { margin:0; }
figure.preview img { display:block; width:100%; height:auto; max-width:100%;
  border:1px solid var(--line); border-radius:18px; background:#EEF3FA; }
figure.preview figcaption { color:var(--ink-soft); font-size:14px;
  margin-top:10px; }

pre.text { margin:0; padding:15px 17px; background:var(--paper-2);
  border:1px solid var(--line); border-radius:var(--radius-sm);
  overflow-x:auto; font-size:14px; line-height:1.55; }

footer.foot { border-top:1px solid var(--line); background:var(--card);
  color:var(--ink-soft); font-size:14px; }
footer.foot .wrap { padding-top:22px; padding-bottom:26px; }
footer.foot p { margin:0 0 8px; }

@media (pointer:coarse) {
  .btn, .modes label.mode { min-height:48px; }
}
@media (prefers-reduced-motion:reduce) {
  * { transition:none !important; animation:none !important; }
  .btn:active { transform:none; }
}
"""

ICONS = {
    "pass": ('<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
             '<path d="M2 9l4 4 8-10" fill="none" stroke="currentColor" '
             'stroke-width="2.6" stroke-linecap="round" '
             'stroke-linejoin="round"/></svg>'),
    "fail": ('<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
             '<path d="M3 3l10 10M13 3L3 13" fill="none" '
             'stroke="currentColor" stroke-width="2.6" '
             'stroke-linecap="round"/></svg>'),
    "na": ('<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
           '<path d="M3 8h10" fill="none" stroke="currentColor" '
           'stroke-width="2.6" stroke-linecap="round"/></svg>'),
}
