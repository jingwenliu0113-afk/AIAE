"""The two pages, as HTML.

Templates live in this module rather than in a ``templates/`` directory on
purpose: the public snapshot allowlist publishes ``src/**/*.py`` and would not
carry a stray ``.html`` file, so a template kept outside Python would leave the
published UI silently broken.

The look is a chunky moulded-plastic-block style -- thick borders, deep flat
shadows, rounded corners, and a strip of studs drawn with CSS gradients.  It
carries no third-party brand mark, wordmark, logo or figure of any kind, and
none is referenced: this project is not affiliated with any brick manufacturer
and the interface must not imply otherwise.

The eight part swatches use :data:`src.rendering.preview.PART_COLOURS`, the
same values the CPU 3-D preview paints with, so a part is the same colour in
the form as it is in the image.  Colour is never the only carrier of meaning:
every check prints ``pass`` / ``FAIL`` / ``n/a`` as text next to a shape, so a
reader who cannot separate the hues reads the same three answers.
"""

from __future__ import annotations

from jinja2 import DictLoader, Environment, StrictUndefined

from src.data.bricks import PART_VOCAB, WORLD
from src.rendering.preview import PART_COLOURS

from src.ui.app import (CONNECTIVITY_LIMIT_ZH, MODE_COMPARE, MODE_LABELS,
                        MODE_PIPELINE, NO_MODEL_ZH, NOT_A_METRIC_ZH,
                        RETRIEVAL_LIMIT_ZH)

#: A Chinese gloss beside each check's authoritative English key.  The key is
#: what the JSON and the command line print; the gloss is a reading aid and
#: never replaces it.
CHECK_GLOSS = {
    "parse_success": "每行都是合法磚行",
    "known_parts": "零件都在八種詞彙內",
    "type_compliance": "沒有用到未備庫存的種類",
    "inventory_valid": "種類正確且沒有超領",
    "in_bounds": f"全部落在 {WORLD}×{WORLD}×{WORLD} 世界內",
    "collision_free": "沒有兩塊佔用同一格",
    "stud_only_connected": "相鄰層 footprint 交集下為單一元件",
    "touches_ground": "有磚位於 z = 0",
    "ldraw_serializable": "可序列化為 LDraw",
    "termination_accepted": "終止原因可接受（本介面不適用）",
    "deterministic_core_success": "以上全部成立",
}

SOLVER_GLOSS = {
    "OPTIMAL": "求得最佳解",
    "FEASIBLE": "求得可行解",
    "INFEASIBLE": "在此庫存下無解",
    "UNKNOWN": "時限內未定（逾時）",
    "MODEL_INVALID": "模型無效",
    "NOT_SOLVED": "未求解",
}

_CSS = """
:root {
  --paper:#F5F2EA; --paper-2:#EDE8DC; --card:#FFFFFF;
  --ink:#1C222C; --ink-soft:#4C5665;
  --line:#D8D1C2; --line-deep:#C0B7A4;
  --accent:#1D4ED8; --accent-deep:#1740AC; --on-accent:#FFFFFF;
  --ok:#1F6F3C; --ok-bg:#E7F4EA;
  --bad:#B3261E; --bad-bg:#FCEBEA;
  --warn:#8A4B00; --warn-bg:#FBF0E0;
  --na:#5B6472;  --na-bg:#EFF1F4;
  --radius:16px; --radius-sm:10px; --lift:4px;
  --stud:rgba(28,34,44,.10);
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",
    "PingFang TC","Hiragino Sans","Noto Sans TC","Microsoft JhengHei",
    "Heiti TC",sans-serif;
  font-size:16px; line-height:1.6;
}
a { color:var(--accent); }
:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
.skip {
  position:absolute; left:-9999px; top:0;
  background:var(--card); padding:12px 16px; border:3px solid var(--ink);
  border-radius:var(--radius-sm); z-index:99;
}
.skip:focus { left:12px; top:12px; }

/* --- the stud strip: eight circles, drawn, no image asset -------------- */
.studs {
  height:14px; border-radius:8px 8px 0 0;
  background-color:var(--paper-2);
  background-image:radial-gradient(circle at 10px 50%,
    var(--stud) 0 4.5px, transparent 5px);
  background-size:20px 100%; background-repeat:repeat-x;
}

.wrap { max-width:1120px; margin:0 auto; padding:20px 18px 64px; }

header.top {
  background:var(--card); border-bottom:3px solid var(--line-deep);
}
header.top > .studs {
  height:18px; border-radius:0; border-bottom:3px solid var(--line);
  background-size:24px 100%;
  background-image:radial-gradient(circle at 12px 55%,
    var(--stud) 0 5.5px, transparent 6px);
}
header.top .wrap { padding:2px 18px 18px; }
.brandline { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
  padding-top:16px; }
.brandline h1 { font-size:1.4rem; margin:0; letter-spacing:-.01em; }
.brandline .tag { color:var(--ink-soft); font-size:.9rem; }

/* --- step indicator ---------------------------------------------------- */
ol.steps { display:flex; gap:10px; list-style:none; margin:14px 0 0;
  padding:0; flex-wrap:wrap; }
ol.steps li {
  display:flex; align-items:center; gap:8px;
  border:3px solid var(--line); border-radius:999px;
  padding:6px 14px; background:var(--paper-2); color:var(--ink-soft);
  font-size:.9rem; min-height:40px;
}
ol.steps li .n {
  width:24px; height:24px; border-radius:50%; display:grid;
  place-items:center; background:var(--line); color:var(--ink);
  font-size:.8rem; font-weight:700;
}
ol.steps li[aria-current="step"] {
  background:var(--card); border-color:var(--ink); color:var(--ink);
  font-weight:600;
}
ol.steps li[aria-current="step"] .n { background:var(--accent);
  color:var(--on-accent); }

/* --- cards ------------------------------------------------------------- */
.card {
  background:var(--card); border:3px solid var(--line-deep);
  border-radius:var(--radius); margin:18px 0;
  box-shadow:0 var(--lift) 0 var(--line-deep), inset 0 2px 0 #fff;
}
.card > .studs { border-bottom:3px solid var(--line); }
.card .body { padding:18px; }
.card h2 { margin:0 0 4px; font-size:1.12rem; }
.card h3 { margin:20px 0 8px; font-size:1rem; }
.card .lede { margin:0 0 14px; color:var(--ink-soft); font-size:.92rem; }

.grid2 { display:grid; gap:18px; grid-template-columns:1fr;
  align-items:start; }
@media (min-width:880px) { .grid2 { grid-template-columns:1.05fr .95fr; } }

/* --- forms ------------------------------------------------------------- */
fieldset { border:3px solid var(--line); border-radius:var(--radius-sm);
  margin:0 0 16px; padding:14px 16px 16px; }
fieldset[disabled] { opacity:.55; }
legend { font-weight:700; padding:0 8px; font-size:.95rem; }
label { display:block; font-weight:600; margin-bottom:6px; font-size:.93rem; }
.hint { color:var(--ink-soft); font-size:.85rem; font-weight:400;
  margin:6px 0 0; }
input[type=text], input[type=number], textarea, select {
  width:100%; font:inherit; color:var(--ink); background:var(--paper);
  border:3px solid var(--line-deep); border-radius:var(--radius-sm);
  padding:10px 12px; min-height:44px;
}
textarea { min-height:120px; resize:vertical; line-height:1.6; }
input[type=number] { min-height:44px; max-width:14ch; }
.parts input[type=number] { max-width:none; }

.modes { display:grid; gap:10px; }
.modes label.mode {
  display:flex; gap:12px; align-items:flex-start; font-weight:400;
  border:3px solid var(--line); border-radius:var(--radius-sm);
  padding:12px 14px; background:var(--paper); cursor:pointer;
  min-height:44px;
}
.modes label.mode:hover { border-color:var(--line-deep); }
.modes input[type=radio] { width:20px; height:20px; margin-top:4px;
  accent-color:var(--accent); flex:0 0 auto; cursor:pointer; }
.modes .mode strong { display:block; font-weight:700; }
.modes .mode span { color:var(--ink-soft); font-size:.87rem; }

/* --- the eight parts --------------------------------------------------- */
.parts { display:grid; gap:10px; grid-template-columns:repeat(2,1fr); }
.pair { display:grid; gap:12px; grid-template-columns:1fr; }
@media (min-width:560px){ .pair { grid-template-columns:1fr 1fr; } }
@media (min-width:560px){ .parts { grid-template-columns:repeat(4,1fr);} }
.part { border:3px solid var(--line); border-radius:var(--radius-sm);
  padding:10px; background:var(--paper); }
.part .head { display:flex; align-items:center; margin-bottom:8px; }
.part label { margin:0; font-size:.9rem; }
.part input { padding:8px 10px; }
.chip {
  display:inline-block; vertical-align:-6px; margin-right:8px;
  flex:0 0 auto; width:36px; height:24px; border-radius:6px;
  border:2px solid rgba(28,34,44,.45);
  background-image:radial-gradient(circle at 9px 7px,
      rgba(255,255,255,.78) 0 3px, transparent 3.4px),
    radial-gradient(circle at 25px 7px,
      rgba(255,255,255,.78) 0 3px, transparent 3.4px);
  box-shadow:inset 0 -4px 0 rgba(28,34,44,.16);
}

/* --- buttons ----------------------------------------------------------- */
.actions { display:flex; gap:12px; flex-wrap:wrap; align-items:center;
  margin-top:6px; }
.actionbar .body { display:flex; gap:16px; align-items:center;
  flex-wrap:wrap; justify-content:space-between; }
.actionbar .why { margin:0; color:var(--ink-soft); font-size:.88rem;
  max-width:62ch; }
.btn {
  display:inline-flex; align-items:center; gap:8px; cursor:pointer;
  font:inherit; font-weight:700; text-decoration:none;
  min-height:48px; padding:12px 22px; border-radius:var(--radius-sm);
  border:3px solid var(--accent-deep); background:var(--accent);
  color:var(--on-accent); box-shadow:0 var(--lift) 0 var(--accent-deep);
  transition:transform .18s ease, box-shadow .18s ease;
}
.btn:hover { transform:translateY(2px); box-shadow:0 2px 0 var(--accent-deep); }
.btn:active { transform:translateY(4px); box-shadow:0 0 0 var(--accent-deep); }
.btn.ghost { background:var(--card); color:var(--ink);
  border-color:var(--line-deep); box-shadow:0 var(--lift) 0 var(--line-deep); }
.btn.ghost:hover { box-shadow:0 2px 0 var(--line-deep); }
.btn[aria-disabled="true"] { background:var(--na-bg); color:var(--na);
  border-color:var(--line); box-shadow:none; cursor:not-allowed;
  transform:none; }

/* --- notices ----------------------------------------------------------- */
.note { border:3px solid var(--line); border-left-width:10px;
  border-radius:var(--radius-sm); padding:12px 14px; margin:0 0 14px;
  background:var(--paper); font-size:.9rem; }
.note.bad { border-color:var(--bad); background:var(--bad-bg);
  color:var(--ink); }
.note.ok { border-color:var(--ok); background:var(--ok-bg); }
.note.warn { border-color:var(--warn); background:var(--warn-bg); }
.note.flat { border-color:var(--line-deep); background:var(--paper-2); }
.note strong { display:block; margin-bottom:4px; }
.note ul { margin:6px 0 0; padding-left:20px; }

/* --- tables ------------------------------------------------------------ */
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch;
  border:3px solid var(--line); border-radius:var(--radius-sm);
  background:var(--paper); }
table { border-collapse:collapse; width:100%; font-size:.9rem;
  min-width:560px; }
caption { text-align:left; padding:10px 12px; color:var(--ink-soft);
  font-size:.85rem; }
th, td { text-align:left; padding:9px 12px;
  border-bottom:2px solid var(--line); vertical-align:top; }
th { background:var(--paper-2); font-size:.82rem; letter-spacing:.02em;
  text-transform:uppercase; color:var(--ink-soft); white-space:nowrap; }
tbody tr:last-child td { border-bottom:0; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.mono, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
  monospace; font-size:.86rem; }
tr.picked td { background:var(--ok-bg); }

/* --- verdict pills ----------------------------------------------------- */
.pill { display:inline-flex; align-items:center; gap:6px; font-weight:700;
  font-size:.8rem; padding:3px 10px; border-radius:999px;
  border:2px solid currentColor; white-space:nowrap; }
.pill svg { width:12px; height:12px; }
.pill.pass { color:var(--ok); background:var(--ok-bg); }
.pill.fail { color:var(--bad); background:var(--bad-bg); }
.pill.na   { color:var(--na);  background:var(--na-bg); }

.checks { display:grid; gap:8px; grid-template-columns:1fr; margin:0; }
@media (min-width:720px){ .checks { grid-template-columns:1fr 1fr; } }
.checks div { display:flex; gap:10px; align-items:baseline;
  border:2px solid var(--line); border-radius:var(--radius-sm);
  padding:8px 12px; background:var(--paper); }
.checks .k { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.84rem; }
.checks .g { color:var(--ink-soft); font-size:.82rem; }

.kv { display:grid; grid-template-columns:auto 1fr; gap:4px 14px;
  margin:0; font-size:.9rem; }
.kv dt { color:var(--ink-soft); white-space:nowrap; }
.kv dd { margin:0; overflow-wrap:anywhere; }

figure.preview { margin:0; }
figure.preview img { display:block; width:100%; height:auto;
  max-width:100%; border:3px solid var(--line-deep);
  border-radius:var(--radius-sm); background:var(--card); }
figure.preview figcaption { color:var(--ink-soft); font-size:.85rem;
  margin-top:8px; }

pre.text { margin:0; padding:12px 14px; background:var(--paper);
  border:3px solid var(--line); border-radius:var(--radius-sm);
  overflow-x:auto; font-size:.86rem; line-height:1.55; }

footer.foot { border-top:3px solid var(--line-deep); background:var(--card);
  color:var(--ink-soft); font-size:.85rem; }
footer.foot .wrap { padding:18px; }
footer.foot p { margin:0 0 6px; }

@media (prefers-reduced-motion:reduce) {
  * { transition:none !important; animation:none !important; }
  .btn:hover, .btn:active { transform:none; }
}
"""

_ICONS = {
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

_BASE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{{ title }}｜BrickAgain</title>
<style>{{ css | safe }}</style>
</head>
<body>
<a class="skip" href="#main">跳到主要內容</a>
<header class="top">
  <div class="studs" aria-hidden="true"></div>
  <div class="wrap">
    <div class="brandline">
      <h1>BrickAgain 最小兩頁式介面</h1>
      <span class="tag">本機 · CPU · 離線 · 只綁 loopback</span>
    </div>
    <ol class="steps">
      <li{% if step == 1 %} aria-current="step"{% endif %}>
        <span class="n">1</span>需求與庫存</li>
      <li{% if step == 2 %} aria-current="step"{% endif %}>
        <span class="n">2</span>結果與交付</li>
    </ol>
  </div>
</header>
<main id="main" class="wrap">
{% block main %}{% endblock %}
</main>
<footer class="foot">
  <div class="wrap">
    <p>{{ not_a_metric }}</p>
    <p>{{ connectivity_limit }}</p>
    <p>{{ no_model }}</p>
    <p>本專題與任何積木製造商無關，介面不使用任何第三方商標、標誌或角色。</p>
  </div>
</footer>
</body>
</html>
"""

_PAGE1 = """{% extends "base.html" %}
{% block main %}
{% if error %}
<div class="note bad" role="alert">
  <strong>這次沒有執行，原因如下</strong>
  {{ error }}
</div>
{% endif %}

{% if not catalog_present %}
<div class="note warn" role="status">
  <strong>找不到目錄檔，送出後會被拒絕</strong>
  預期的 train-only 目錄是 <span class="mono">{{ catalog_display }}</span>。
  它是私有 processed 資料，不在公開 snapshot；公開 checkout 沒有它時，
  本介面會明確拒絕，不會改讀其他 split。
</div>
{% endif %}

<form method="post" action="/result" accept-charset="utf-8">
{# The per-process form key. A page on another origin can cause a submission
   but cannot read this value out of a page it is not allowed to read, so a
   submission without it is refused rather than run. #}
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
<div class="grid2">

  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>需求</h2>
      <p class="lede">用中文描述要組的作品，再選一種方法。</p>

      <label for="caption">文字需求（中文或英文皆可）</label>
      <textarea id="caption" name="caption" required
        maxlength="{{ max_caption }}"
        aria-describedby="caption-hint"
        placeholder="例如：一台低矮的小車，車身平整，上面有一排凸點。"
        >{{ form.caption }}</textarea>
      <p class="hint" id="caption-hint">{{ retrieval_limit }}</p>

      <h3>方法</h3>
      <div class="modes">
        <label class="mode">
          <input type="radio" name="mode" value="{{ mode_compare }}"
            {% if form.mode == mode_compare %}checked{% endif %}
            data-cpsat="off">
          <span>
            <strong>{{ mode_labels[mode_compare] }}</strong>
            <span>只在 train split 的既有作品中比對：詞彙 Top-N 之後，
              依精確缺件、完成比例與可組性重排。不重新鋪磚。</span>
          </span>
        </label>
        <label class="mode">
          <input type="radio" name="mode" value="{{ mode_pipeline }}"
            {% if form.mode == mode_pipeline %}checked{% endif %}
            data-cpsat="on">
          <span>
            <strong>{{ mode_labels[mode_pipeline] }}</strong>
            <span>取回 train 形狀後，交給既有 CP-SAT 依手動庫存重新鋪磚，
              再獨立復驗 exact cover、庫存、碰撞、邊界、接地與連通。</span>
          </span>
        </label>
      </div>

      <h3>檢索範圍</h3>
      <label for="top_n">Top-N（要考慮幾件 train 候選）</label>
      <input type="number" id="top_n" name="top_n" min="1"
        max="{{ max_top_n }}" step="1" value="{{ form.top_n }}"
        aria-describedby="topn-hint">
      <p class="hint" id="topn-hint">1 到 {{ max_top_n }}。
        F-pipeline 會依序嘗試，取第一個通過復驗的結果。</p>

      <fieldset id="cpsat" {% if form.mode != mode_pipeline %}disabled{% endif %}>
        <legend>CP-SAT 控制（只適用於{{ mode_labels[mode_pipeline] }}）</legend>
        <p class="hint" style="margin-top:0">
          選擇「{{ mode_labels[mode_compare] }}」時這兩欄不適用。填了會被
          <strong>具名拒絕</strong>，不會被靜默忽略。</p>
        <div class="pair">
          <div>
            <label for="time_limit">每個候選的 time limit（秒）</label>
            <input type="number" id="time_limit" name="time_limit"
              min="0" max="{{ max_time_limit }}" step="0.5"
              value="{{ form.time_limit }}">
          </div>
          <div>
            <label for="seed">seed（決定性）</label>
            <input type="number" id="seed" name="seed" min="0" step="1"
              value="{{ form.seed }}">
          </div>
        </div>
      </fieldset>
      <noscript>
        <p class="hint">未啟用 JavaScript 時，這兩欄不會自動停用；
          送出後仍由伺服器判定適用性並具名拒絕。</p>
      </noscript>
    </div>
  </section>

  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>手動庫存</h2>
      <p class="lede">八種正式零件。旋轉拼法會正規化到同一項庫存，
        例如 <span class="mono">4x1</span> 與
        <span class="mono">1x4</span> 是同一種。留白或 0 表示沒有這種零件。</p>

      <div class="parts">
        {% for part in parts %}
        <div class="part">
          <div class="head">
            <span class="chip" aria-hidden="true"
              style="background-color:{{ part_colours[part] }}"></span>
            <label for="qty_{{ part }}">{{ part }}</label>
          </div>
          <input type="number" id="qty_{{ part }}" name="qty_{{ part }}"
            min="0" step="1" inputmode="numeric"
            value="{{ form.grid.get(part, '') }}">
        </div>
        {% endfor %}
      </div>
      <p class="hint">色塊與 3D 預覽用的是同一組零件顏色。</p>

      <h3>進階：直接輸入庫存字串</h3>
      <label for="inventory_spec">庫存字串（可留白）</label>
      <input type="text" id="inventory_spec" name="inventory_spec"
        value="{{ form.inventory_spec }}"
        maxlength="{{ max_spec }}"
        placeholder="2x4:10,1x2:8"
        aria-describedby="spec-hint">
      <p class="hint" id="spec-hint">
        與上面八格是同一份庫存的兩種輸入方式，<strong>只能用一種</strong>；
        兩邊都填會被拒絕，本介面不自行猜測該相加或覆蓋。
        同一種零件的兩種旋轉拼法同時給出也會被拒絕。</p>
    </div>
  </section>

</div>

<section class="card actionbar">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <p class="why">送出後在同一台機器上以 CPU 執行：不連網、不載入權重、
      不使用 GPU。不適用的欄位會被具名拒絕，不會被靜默忽略。</p>
    <div class="actions">
      <button class="btn" type="submit">送出並查看結果</button>
    </div>
  </div>
</section>
</form>

<div class="note flat">
  <strong>這個介面不提供什麼</strong>
  <ul>
    <li>不提供模型生成，也沒有 <span class="mono">--generate</span> 入口。</li>
    <li>不提供 Phase 3 placement gate，該閘門從未正式評估。</li>
    <li>不提供正式評估，也不會讀取任何已封存的評估案例。</li>
  </ul>
</div>

<script>
/* Progressive enhancement only. The server is the authority on which fields
   apply: with this script absent, an inapplicable field is refused by name
   rather than silently ignored. */
(function () {
  var set = document.getElementById('cpsat');
  var radios = document.querySelectorAll('input[name="mode"]');
  if (!set || !radios.length) { return; }
  function sync() {
    var on = false;
    radios.forEach(function (r) {
      if (r.checked && r.getAttribute('data-cpsat') === 'on') { on = true; }
    });
    set.disabled = !on;
  }
  radios.forEach(function (r) { r.addEventListener('change', sync); });
  sync();
})();
</script>
{% endblock %}
"""

_PAGE2 = """{% extends "base.html" %}
{% block main %}

{% if ready %}
<div class="note ok" role="status">
  <strong>找到一件通過靜態交付檢查的結果</strong>
  下方的 LDraw 下載與 3D 預覽對應同一份磚清單。
  這是該件輸出的確定性檢查，<em>不是</em>成功率、不是
  <span class="mono">Structural Success@K</span>，也不是任何模型指標。
</div>
{% else %}
<div class="note warn" role="status">
  <strong>流程正常完成，但本次沒有可交付結果</strong>
  {{ not_ready_reason }}
  沒有產生預覽，也沒有開放下載——沒有通過檢查的結構不會被畫成看起來合法的圖片。
</div>
{% endif %}

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>方法與資料來源</h2>
    <p class="lede">provenance 先講，因為結果從哪來，決定下面的檢查值多少。</p>
    <dl class="kv">
      <dt>方法</dt><dd>{{ mode_labels[method.name] }}
        （<span class="mono">{{ method.name }}</span>）</dd>
      <dt>檢索方式</dt><dd>{{ method.retrieval }}</dd>
      <dt>檢索限制</dt><dd>{{ method.retrieval_limit }}</dd>
      <dt>形狀來源</dt><dd>{{ method.shape_source }}</dd>
      <dt>載入模型</dt><dd><span class="pill na">
        {{ icons['na'] | safe }} {{ method.model_loaded }}</span>
        ——本介面沒有解碼器，終止原因不適用</dd>
      <dt>Phase 3C</dt><dd>{{ method.phase_3c }}</dd>
      <dt>執行狀態</dt><dd><span class="mono">{{ status }}</span>
        ——{{ status_gloss }}</dd>
      <dt>目錄檔</dt><dd><span class="mono">{{ catalog.file }}</span>
        ，split=<span class="mono">{{ catalog.split }}</span>，
        {{ catalog.canonical_structures }} 件 canonical train 結構</dd>
      <dt>目錄 SHA-256</dt><dd class="mono">{{ catalog.sha256 }}</dd>
      <dt>凍結 split manifest SHA-256</dt>
        <dd class="mono">{{ catalog.split_manifest_sha256 }}</dd>
      <dt>文字需求</dt><dd>{{ request.caption }}</dd>
      <dt>手動庫存</dt><dd class="mono">{{ inventory_spec }}</dd>
      <dt>Top-N</dt><dd class="num">{{ request.top_n }}</dd>
      {% if cpsat %}
      <dt>CP-SAT</dt><dd>time limit {{ cpsat.time_limit }} 秒，
        seed {{ cpsat.seed }}</dd>
      {% endif %}
    </dl>
    <p class="hint">凍結的是 split manifest 與它的 SHA，不是目錄本身；
      每次執行都記下當次目錄 SHA 以供對帳。</p>
  </div>
</section>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    {% if method.name == mode_compare %}
    <h2>候選（依庫存證據重排）</h2>
    <p class="lede">先做確定性詞彙 Top-N，再依精確缺件、完成比例與可組性重排。
      <span class="mono">catalog_id</span> 是匿名識別碼，不是資料集識別碼。</p>
    <div class="scroll">
      <table>
        <caption>共 {{ rows | length }} 件候選；綠底是被選中的那一件。</caption>
        <thead><tr>
          <th>#</th><th>catalog_id</th><th class="num">詞彙分數</th>
          <th class="num">磚數</th><th>所需庫存</th>
          <th class="num">缺件</th><th class="num">完成比例</th>
          <th>可組</th><th>作品說明</th>
        </tr></thead>
        <tbody>
        {% for row in rows %}
        <tr{% if row.catalog_id == selected_id %} class="picked"{% endif %}>
          <td class="num">{{ loop.index }}</td>
          <td class="mono">{{ row.catalog_id }}</td>
          <td class="num">{{ '%.4f' % row.lexical_score }}</td>
          <td class="num">{{ row.n_bricks }}</td>
          <td class="mono">{{ row.required_inventory
            | dictsort | map('join', ':') | join(', ') }}</td>
          <td class="num">{{ row.missing_total }}
            {%- if row.missing_parts %}<br><span class="mono"
              >{{ row.missing_parts | dictsort
                 | map('join', ':') | join(', ') }}</span>{% endif %}</td>
          <td class="num">{{ '%.0f%%' % (row.inventory_completion * 100) }}</td>
          <td>{% if row.fully_buildable %}
              <span class="pill pass">{{ icons['pass'] | safe }} 可組</span>
            {% else %}
              <span class="pill fail">{{ icons['fail'] | safe }} 不足</span>
            {% endif %}</td>
          <td>{{ row.caption }}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="hint">「可組」同時要求庫存足夠、接觸地面且相鄰層連通。
      沒有可組候選時不會把最相似的說成可組。</p>

    {% else %}
    <h2>F-pipeline 各候選</h2>
    <p class="lede">依檢索順序嘗試，取第一個通過獨立復驗的結果。
      逾時與無解是兩種不同狀態，不會被寫成同一種。</p>
    <div class="scroll">
      <table>
        <caption>共嘗試 {{ rows | length }} 個候選；綠底是被選中的那一個。</caption>
        <thead><tr>
          <th>#</th><th>catalog_id</th><th>求解狀態</th>
          <th class="num">求解秒數</th><th class="num">candidate placements</th>
          <th>回傳鋪排</th><th>exact cover</th><th>庫存</th><th>碰撞</th>
          <th>邊界</th><th>接地</th><th>連通</th><th>可交付</th><th>失敗理由</th>
        </tr></thead>
        <tbody>
        {% for row in rows %}
        <tr{% if row.catalog_id == selected_id and row.delivery_ready
              %} class="picked"{% endif %}>
          <td class="num">{{ loop.index }}</td>
          <td class="mono">{{ row.catalog_id }}</td>
          <td><span class="mono">{{ row.solver_status }}</span><br>
            <span class="g">{{ solver_gloss.get(row.solver_status,
              row.solver_status) }}</span></td>
          <td class="num">{{ '%.3f' % row.wall_seconds }}</td>
          <td class="num">{{ row.candidate_placements }}</td>
          {% for key in attempt_flags %}
          <td>{% if row[key] %}
              <span class="pill pass">{{ icons['pass'] | safe }} pass</span>
            {% else %}
              <span class="pill fail">{{ icons['fail'] | safe }} FAIL</span>
            {% endif %}</td>
          {% endfor %}
          <td>{{ row.failure or '—' }}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="hint">求解秒數是這台機器這一次的實測，不是模型或求解器的本質速度。</p>
    {% endif %}
  </div>
</section>

{% if report %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>選中結果的靜態檢查</h2>
    <p class="lede">全部由
      <span class="mono">{{ report.scored_by }}</span> 計算並直接引用，
      本介面沒有第二份判定邏輯。</p>
    <div class="checks">
      {% for row in checks %}
      <div>
        <span class="pill {{ row.css }}">{{ icons[row.icon] | safe }}
          {{ row.label }}</span>
        <span class="k">{{ row.name }}</span>
        <span class="g">{{ row.gloss }}</span>
      </div>
      {% endfor %}
    </div>
    <p class="hint">
      <span class="mono">n/a</span> 不是 <span class="mono">FAIL</span>：
      本介面沒有跑解碼器，讀取終止原因的兩項檢查沒有答案，因此如實記為
      <span class="mono">null</span>，不寫成 false。
      交付判定用的是不含終止原因的九項靜態檢查：
      <span class="mono">{{ delivery_checks | join(' ') }}</span>。
    </p>
    <div class="note flat" style="margin-top:14px">
      {{ connectivity_limit }}
      額外的 <span class="mono">unsupported</span> 計數是 scorer 自己的描述性
      統計（本次 {{ report.unsupported.unsupported_brick_count }} 塊），
      同樣不是穩定性結果。
    </div>

    <h3>庫存使用量與剩餘</h3>
    <div class="scroll">
      <table>
        <caption>負數會如實印成負數並標記 OVERDRAWN，不會夾到 0。</caption>
        <thead><tr><th>零件</th><th class="num">備料</th>
          <th class="num">使用</th><th class="num">剩餘</th><th>狀態</th>
        </tr></thead>
        <tbody>
        {% for row in inventory_rows %}
        <tr>
          <td><span class="chip" aria-hidden="true"
              style="background-color:{{ part_colours[row.part] }}"></span>
            <span class="mono">{{ row.part }}</span></td>
          <td class="num">{{ row.stocked }}</td>
          <td class="num">{{ row.used }}</td>
          <td class="num">{{ row.left }}</td>
          <td>{% if row.left < 0 %}
              <span class="pill fail">{{ icons['fail'] | safe }} OVERDRAWN</span>
            {% else %}
              <span class="pill pass">{{ icons['pass'] | safe }} 足夠</span>
            {% endif %}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="hint">超領總量：{{ report.inventory.count_overflow_amount }}
      {% if report.inventory.type_violations %}
      ；用到未備庫存的種類：<span class="mono"
        >{{ report.inventory.type_violations | join(', ') }}</span>{% endif %}
    </p>

    <h3>磚清單（{{ report.result.n_bricks }} 塊）</h3>
    <pre class="text">{{ report.result.text.rstrip() }}</pre>
    <p class="hint">磚清單印的是每塊積木的實際擺放方向，庫存表印的是正規化後的
      項目：例如幾何上的 <span class="mono">2x1</span> 與
      <span class="mono">1x2</span> 是同一項庫存，因此兩張表的拼法可能不同，
      數量仍然對得上。</p>
  </div>
</section>

{% if ready %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>CPU 3D 幾何預覽與交付</h2>
    <figure class="preview">
      <img src="/artifact/{{ handle }}/preview.png"
        width="{{ preview_width }}" height="{{ preview_height }}"
        alt="選中結果的 3D 幾何預覽：以 {{ report.result.n_bricks }} 個
          軸對齊長方體畫出每一塊積木，顏色對應零件種類。
          這是幾何檢視，不是寫實渲染。">
      <figcaption>以 Matplotlib Agg 在 CPU 上把每塊積木畫成軸對齊長方體。
        這是幾何檢視，不是寫實渲染，也沒有物理或穩定性分析。
        涉及碰撞的積木會標成洋紅色。</figcaption>
    </figure>
    <div class="actions">
      <a class="btn" href="/artifact/{{ handle }}/model.ldr" download>
        下載 LDraw（.ldr）</a>
      <a class="btn ghost" href="/">回到第一頁，換一組需求</a>
    </div>
    <p class="hint">下載內容與上方預覽出自同一份磚清單，
      由已對齊 BrickGPT 參考向量的既有 writer 產生。
      檔案只存在本機記憶體與這次連線，不寫入專案的
      <span class="mono">artifacts/</span>。</p>
  </div>
</section>
{% endif %}

{% else %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>沒有選中結果</h2>
    <p class="lede">因此沒有靜態檢查表、沒有預覽、也沒有下載。
      上方的候選證據仍完整保留，讓「差在哪裡」可讀。</p>
    <div class="actions">
      <a class="btn" href="/">回到第一頁，調整庫存或需求</a>
    </div>
  </div>
</section>
{% endif %}

<div class="note flat">
  <strong>這一頁不能怎麼用</strong>
  <ul>
    <li>{{ not_a_metric }}</li>
    <li>沒有 <span class="mono">Structural</span>／
      <span class="mono">Semantic</span>／<span class="mono">Full
      Success@K</span>，也不得把單件檢查改名成它們。</li>
    <li>{{ no_model }}</li>
  </ul>
</div>
{% endblock %}
"""

_ERROR = """{% extends "base.html" %}
{% block main %}
<div class="note bad" role="alert">
  <strong>{{ heading }}</strong>
  {{ detail }}
</div>
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>接下來可以做什麼</h2>
    <p class="lede">{{ advice }}</p>
    <div class="actions">
      <a class="btn" href="/">回到第一頁</a>
    </div>
  </div>
</section>
{% endblock %}
"""

#: The stylesheet and the three status icons, exposed so the full interface's
#: templates use *these* rather than a second copy. One look, one place.
CSS = _CSS
ICONS = _ICONS

_ENV = Environment(
    loader=DictLoader({
        "base.html": _BASE,
        "page1.html": _PAGE1,
        "page2.html": _PAGE2,
        "error.html": _ERROR,
    }),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

#: The per-attempt boolean columns of the F-pipeline table, in report order.
ATTEMPT_FLAGS = ("solver_returned_tiling", "exact_cover_verified",
                 "inventory_verified", "collision_free", "in_bounds",
                 "touches_ground", "stud_only_connected", "delivery_ready")

_STATUS_GLOSS = {
    "buildable_existing_work_found": "在 Top-N 內找到可完整組裝的既有作品",
    "no_buildable_existing_work_in_retrieved_set":
        "取回的候選都缺件或不符合靜態條件；沒有可組的既有作品",
    "success": "有候選通過 CP-SAT 鋪排與全部獨立復驗",
    "tiling_found_but_not_delivery_ready":
        "求解器有回傳鋪排，但沒有一個通過全部復驗",
    "no_valid_build": "沒有任何候選在此庫存下得到鋪排",
}

_COMMON = {
    "css": _CSS,
    "icons": _ICONS,
    "parts": PART_VOCAB,
    "part_colours": PART_COLOURS,
    "mode_compare": MODE_COMPARE,
    "mode_pipeline": MODE_PIPELINE,
    "mode_labels": MODE_LABELS,
    "retrieval_limit": RETRIEVAL_LIMIT_ZH,
    "connectivity_limit": CONNECTIVITY_LIMIT_ZH,
    "not_a_metric": NOT_A_METRIC_ZH,
    "no_model": NO_MODEL_ZH,
}


def blank_form() -> dict:
    """The page-one field state before anything has been submitted."""
    return {"caption": "", "mode": MODE_COMPARE, "grid": {},
            "inventory_spec": "", "top_n": 5, "time_limit": "2", "seed": "0"}


def form_state(fields) -> dict:
    """Echo a submission back so a refusal does not discard what was typed.

    The form key is deliberately not among the echoed fields.  The re-rendered
    page carries the server's own key, so a submitted one is never reflected.
    """
    def one(name, default=""):
        values = fields.get(name) or []
        return values[0] if values else default

    grid = {}
    for part in PART_VOCAB:
        raw = one(f"qty_{part}").strip()
        if raw:
            grid[part] = raw
    mode = one("mode", MODE_COMPARE)
    return {
        "caption": one("caption"),
        "mode": mode if mode in (MODE_COMPARE, MODE_PIPELINE) else MODE_COMPARE,
        "grid": grid,
        "inventory_spec": one("inventory_spec"),
        "top_n": one("top_n", "5"),
        "time_limit": one("time_limit", "2"),
        "seed": one("seed", "0"),
    }


def render_page_one(*, csrf_token: str, form: dict | None = None,
                    error: str | None = None, catalog_present: bool = True,
                    catalog_display: str = "") -> str:
    """Page one: the brief, the manual stock and the method.

    ``csrf_token`` has no default on purpose.  A default would render a form
    whose every submission is refused, which is the kind of quietly broken
    page this project would rather not be able to produce.
    """
    if not isinstance(csrf_token, str) or not csrf_token:
        from src.ui.app import UiError
        raise UiError(
            "page one may not be rendered without the form key it has to "
            "carry; a form that cannot be submitted is not a page")
    from src.ui.app import (MAX_CAPTION_CHARS, MAX_INVENTORY_SPEC_CHARS,
                            MAX_TIME_LIMIT, MAX_TOP_N)
    return _ENV.get_template("page1.html").render(
        title="需求與庫存", step=1, form=form or blank_form(), error=error,
        csrf_token=csrf_token,
        catalog_present=catalog_present, catalog_display=catalog_display,
        max_caption=MAX_CAPTION_CHARS, max_spec=MAX_INVENTORY_SPEC_CHARS,
        max_top_n=MAX_TOP_N, max_time_limit=MAX_TIME_LIMIT, **_COMMON)


def _check_rows(report: dict) -> list[dict]:
    order = list(CHECK_GLOSS)
    rows = []
    for name in order:
        if name not in report["checks"]:
            continue
        value = report["checks"][name]
        if value is None:
            css, icon, label = "na", "na", "n/a"
        elif value:
            css, icon, label = "pass", "pass", "pass"
        else:
            css, icon, label = "fail", "fail", "FAIL"
        rows.append({"name": name, "gloss": CHECK_GLOSS[name], "css": css,
                     "icon": icon, "label": label})
    return rows


def _not_ready_reason(payload: dict, report: dict | None) -> str:
    status = payload["result"]["status"]
    if report is None:
        return _STATUS_GLOSS.get(status, status) + "。"
    failed = [name for name, value in report["checks"].items()
              if value is False]
    if failed:
        return ("選出的結構沒有通過靜態交付檢查，未通過的是："
                + "、".join(sorted(failed)) + "。")
    return _STATUS_GLOSS.get(status, status) + "。"


def render_page_two(result) -> str:
    """Page two: the method actually used, the evidence, and the outputs."""
    payload, report = result.payload, result.report
    body = payload["result"]
    rows = (body["inventory_reranked"]
            if payload["method"]["name"] == MODE_COMPARE
            else body["attempts"])

    inventory_rows = []
    if report is not None:
        stocked = report["request"]["inventory"]
        used = report["inventory"]["used"]
        left = report["inventory"]["remaining"]
        inventory_rows = [
            {"part": part, "stocked": count, "used": used.get(part, 0),
             "left": left[part]} for part, count in stocked.items()]

    cpsat = None
    if result.request.is_pipeline:
        cpsat = {"time_limit": result.request.time_limit,
                 "seed": result.request.seed}

    return _ENV.get_template("page2.html").render(
        title="結果與交付", step=2,
        ready=result.ready, handle=result.handle,
        preview_width=result.artifacts.preview_width if result.artifacts else 0,
        preview_height=(result.artifacts.preview_height
                        if result.artifacts else 0),
        method=payload["method"], catalog=payload["catalog"],
        request=payload["request"], inventory_spec=result.request.inventory_spec,
        status=body["status"],
        status_gloss=_STATUS_GLOSS.get(body["status"], body["status"]),
        rows=rows, selected_id=body["selected_catalog_id"],
        attempt_flags=ATTEMPT_FLAGS, solver_gloss=SOLVER_GLOSS,
        report=report, checks=_check_rows(report) if report else [],
        inventory_rows=inventory_rows, cpsat=cpsat,
        delivery_checks=report["delivery"]["checks_used"] if report else [],
        not_ready_reason=_not_ready_reason(payload, report),
        **_COMMON)


def render_error(*, heading: str, detail: str, advice: str,
                 title: str = "無法完成") -> str:
    """A refusal or a defect, as a page. Never a traceback."""
    return _ENV.get_template("error.html").render(
        title=title, step=1, heading=heading, detail=detail, advice=advice,
        **_COMMON)
