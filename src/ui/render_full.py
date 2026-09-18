"""The full interface's four pages, as HTML.

Kept apart from :mod:`src.ui.render`, which holds the two-page version and is
not modified: that version's pages, its refusals and its 195 regression tests
go on meaning exactly what they meant.  What is shared is shared properly --
the stylesheet and the three status icons are imported from there, so there is
one look and one place to change it.

Templates live in Python for the same reason they do next door: the public
snapshot's allowlist publishes ``src/**/*.py``, and a template kept outside
Python would leave the published interface silently broken.

Two rendering decisions are load bearing:

**Boxes are drawn as an SVG overlay, not burned into the image.**  The
photograph is served as it was uploaded and an inline ``<svg>`` with a
``viewBox`` of the image's own pixel size sits over it.  So the image bytes a
person sees are the bytes they sent -- nothing has been re-encoded, and a box
cannot be drawn in the wrong place by a scaling mistake without the whole
overlay being visibly wrong.

**Colour is never the only carrier of meaning.**  Every check prints
``pass`` / ``FAIL`` / ``n/a`` as text beside its shape, and every box carries
its number as a label, so the page reads the same without hue.
"""

from __future__ import annotations

from jinja2 import DictLoader, Environment, StrictUndefined

from src.colour.palette import COLOUR_ORDER, PALETTE
from src.data.bricks import PART_VOCAB, WORLD
from src.rendering.preview import PART_COLOURS
from src.ui.full import (CAPTURE_ASSUMPTION_ZH, METHOD_LABELS,
                         PHOTO_MODE_LABELS, RECOGNISE_CV, RECOGNISE_LEARNED,
                         RECOGNITION_LIMIT_ZH, METHOD_PIPELINE,
                         METHOD_PROJECT, METHOD_RAG, PHOTO_MULTI,
                         PHOTO_SINGLE)
from src.ui.render import CHECK_GLOSS, CSS, ICONS, SOLVER_GLOSS

STEPS = ("說想法", "放積木", "看作品")

_EXTRA_CSS = """
/* --- the opening block ------------------------------------------------- */
.intro { display:grid; grid-template-columns:1fr; gap:22px;
  align-items:center; margin:4px 0 6px; }
@media (min-width:760px) {
  .intro { grid-template-columns:minmax(0,1.25fr) minmax(240px,.75fr);
    gap:28px; }
}
/* No ``ch`` cap here: one CJK character is about 1.8ch wide, so the
   reference's 11ch cut ``今天想蓋什麼？`` in half. The grid column is the
   bound. */
.intro .big { font-size:clamp(32px,4.6vw,56px); line-height:1.12;
  letter-spacing:-.03em; text-wrap:balance; }
.intro p { max-width:32rem; margin:14px 0 0; color:var(--ink-soft);
  font-size:16px; line-height:1.55; }

/* Three bricks. Studs on the top edge are the only thing drawn on them:
   no face, no figure, no expression. */
.brickscene { position:relative; width:238px; height:164px; margin:8px auto 0; }
.brickscene span { position:absolute; display:block; height:48px;
  border:2px solid var(--accent-deep); border-radius:14px;
  background:var(--accent);
  box-shadow:inset 0 -10px 0 rgba(18,59,139,.16), 0 8px 0 var(--shadow); }
.brickscene span::before, .brickscene span::after {
  content:""; position:absolute; top:-12px; width:34px; height:13px;
  border:2px solid var(--accent-deep); border-bottom:0;
  border-radius:9px 9px 2px 2px; background:var(--accent-soft); }
.brickscene span::before { left:18px; }
.brickscene span::after { right:18px; }
.brickscene span:nth-child(1) { top:12px; right:10px; width:124px;
  transform:rotate(8deg); background:var(--accent-soft); }
.brickscene span:nth-child(2) { top:67px; left:0; width:162px;
  transform:rotate(-6deg); }
.brickscene span:nth-child(3) { right:0; bottom:2px; width:188px;
  transform:rotate(3deg); background:var(--accent-mid); }
@media (max-width:760px) {
  .brickscene { transform:scale(.8); margin-top:-10px; margin-bottom:-12px; }
}

/* --- the quick-start aside --------------------------------------------- */
.quick { padding:18px; border-radius:18px; background:var(--paper-2); }
.quick h3 { font-size:17px; margin:0; }
.suggest { display:grid; gap:9px; margin-top:12px; }
.suggest button { min-height:42px; padding:9px 13px; font:inherit;
  border:1px solid var(--line); border-radius:12px; background:var(--card);
  color:var(--ink); text-align:left; cursor:pointer;
  box-shadow:0 3px 0 var(--shadow); }
.suggest button:hover { border-color:var(--accent-soft); }
.suggest button:active { transform:translateY(2px); box-shadow:none; }

/* The tray panel is the full interface's own: the two-page UI keeps the
   plain grid it was designed around. Scoped to ``div`` so the assembly
   page's ``ol.parts`` list is untouched. */
div.parts { padding:15px; border:1px solid var(--line); border-radius:18px;
  background:var(--paper-2); box-shadow:inset 0 3px 0 rgba(47,104,210,.06); }

/* --- the two ways of counting ------------------------------------------ */
.choice-row { display:grid; grid-template-columns:1fr; gap:14px;
  margin:22px 0; }
@media (min-width:760px) { .choice-row { grid-template-columns:1fr 1.35fr; } }
.choice { display:block; min-height:108px; padding:17px;
  border:1px solid var(--line); border-radius:16px; background:var(--card);
  color:var(--ink); text-align:left; text-decoration:none;
  box-shadow:0 5px 0 var(--shadow); }
.choice strong { display:block; font-size:17px; font-weight:500; }
.choice span { display:block; margin-top:6px; color:var(--ink-soft);
  font-size:15px; line-height:1.45; }
.choice[aria-current="true"] { background:var(--paper-2);
  border-color:var(--accent);
  box-shadow:0 0 0 4px rgba(47,104,210,.15), 0 5px 0 var(--shadow); }

/* --- the plus/minus counter, added by script only ---------------------- */
.counter { display:grid; grid-template-columns:34px 1fr 34px;
  align-items:center; gap:5px; margin-top:10px; }
.counter button { width:34px; height:34px; padding:0; font:inherit;
  font-size:18px; line-height:1; border:1px solid var(--line);
  border-radius:10px; background:var(--card); color:var(--ink);
  cursor:pointer; }
.counter button:hover { border-color:var(--accent-soft); }
.counter input { margin:0; }
@media (pointer:coarse) {
  .counter { grid-template-columns:44px 1fr 44px; }
  .counter button { width:44px; height:44px; }
  .suggest button { min-height:44px; }
}

/* --- everything technical sits behind one of these --------------------- */
details.adv { margin:20px 0 0; color:var(--ink-soft); }
details.adv > summary { display:flex; align-items:center; gap:8px;
  min-height:38px; padding:6px 0; cursor:pointer; font-weight:500;
  font-size:15px; color:var(--accent-deep); list-style:none; }
details.adv > summary::-webkit-details-marker { display:none; }
details.adv > summary::after { content:""; flex:0 0 auto; width:8px;
  height:8px; margin-top:-3px; border-right:2px solid currentColor;
  border-bottom:2px solid currentColor; transform:rotate(45deg);
  transition:transform .16s ease; }
details.adv[open] > summary::after { transform:rotate(225deg);
  margin-top:2px; }
details.adv > .inner { padding:6px 0 4px; }
details.adv > .inner > :first-child { margin-top:0; }
details.adv .note:last-child, details.adv .hint:last-child { margin-bottom:0; }

/* --- the result board -------------------------------------------------- */
.result-grid { display:grid; grid-template-columns:1fr; gap:22px;
  align-items:center; }
@media (min-width:760px) {
  .result-grid { grid-template-columns:1.15fr .85fr; }
}
.result-copy h2 { font-size:clamp(24px,3.6vw,31px); }
.result-copy p { color:var(--ink-soft); line-height:1.55; }
.ready { display:inline-flex; align-items:center; gap:7px; margin:0 0 12px;
  font-weight:500; color:var(--accent-deep); }
.ready svg { width:18px; height:18px; }
.ready.no { color:var(--warn); }
.tally { display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 0; }
.tally span { border:1px solid var(--line); border-radius:13px;
  background:var(--paper-2); padding:9px 15px; font-size:14px;
  color:var(--ink-soft); }
.tally b { display:block; font-weight:500; font-size:22px; color:var(--ink);
  line-height:1.2; }

/* --- the detection overlay --------------------------------------------- */
.shot { position:relative; display:inline-block; max-width:100%;
  border:1px solid var(--line); border-radius:18px; overflow:hidden;
  background:#EEF3FA; }
.shot img { display:block; max-width:100%; height:auto; }
.shot svg { position:absolute; inset:0; width:100%; height:100%; }
.shot svg rect { fill:none; stroke:#2F68D2; stroke-width:4;
  vector-effect:non-scaling-stroke; }
.shot svg rect.low { stroke:#8A4B00; stroke-dasharray:10 6; }
.shot svg rect.gone { stroke:#B3261E; stroke-dasharray:3 7; }
.shot svg text { fill:#FFFFFF; font-weight:500; font-size:26px;
  paint-order:stroke; stroke:#17253B; stroke-width:6; }

/* --- the per-item correction table ------------------------------------- */
table.edit td, table.edit th { vertical-align:top; }
table.edit input, table.edit select { min-height:42px; padding:9px 11px; }
table.edit input[type=number] { max-width:9ch; }
table.edit input[type=text] { max-width:17ch; }
table.edit select { min-width:11ch; }
table.edit label { margin:0; }
.tag { display:inline-block; border:1px solid var(--line);
  border-radius:999px; padding:2px 10px; font-size:13px;
  background:var(--paper-2); }
.tag.model { background:var(--na-bg); }
.tag.mixed { background:var(--warn-bg); color:var(--warn); }
.tag.operator { background:var(--ok-bg); color:var(--ok); }
.was { color:var(--ink-soft); font-size:13px; }

/* --- assembly-step navigation ------------------------------------------ */
.stepbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  margin:18px 0; }
.stepbar .count { font-weight:500; color:var(--ink-soft); }
/* ``.parts`` is a grid in the shared sheet; the assembly page's list wants
   the multi-column flow it always had, so say so explicitly. */
ol.parts { display:block; columns:2; margin:0; padding-left:1.4em; }
@media (min-width:720px) { ol.parts { columns:3; } }
.swatchrow { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.swatchrow span { display:inline-flex; align-items:center; gap:7px;
  border:1px solid var(--line); border-radius:999px; padding:4px 12px;
  font-size:14px; background:var(--card); }
.swatchrow i { width:16px; height:12px; border-radius:3px;
  border:1px solid var(--accent-deep); display:inline-block; }
"""

_BASE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{{ title }}｜BrickAgain</title>
<style>{{ css | safe }}{{ extra_css | safe }}</style>
</head>
<body>
<a class="skip" href="#main">跳到主要內容</a>
<header class="top">
  <div class="studs" aria-hidden="true"></div>
  <div class="wrap">
    <div class="brandline">
      <h1>BrickAgain</h1>
      <span class="tag">在你自己的電腦上執行</span>
    </div>
    <ol class="steps">
      {% for name in steps %}
      <li{% if loop.index == step %} aria-current="step"{% endif %}>
        <span class="n">{{ loop.index }}</span>{{ name }}</li>
      {% endfor %}
    </ol>
  </div>
</header>
<main id="main" class="wrap">
{% block main %}{% endblock %}
</main>
<footer class="foot">
  <div class="wrap">
    <p>照片和結果都放在記憶體裡，關掉就沒了，不會存進專案資料夾。</p>
    <details class="adv">
      <summary>技術資訊與限制說明</summary>
      <div class="inner">
        <p>本頁任何數字都不是指標，不可與已封存的 Phase 2 評估比較，
          也不得改名成 Structural／Semantic／Full Success@K。</p>
        <p>連通性是相鄰層 footprint 交集，接觸地面是有磚位於 z = 0；
          兩者都是靜態幾何，不是物理支撐，也不是穩定性分析。</p>
        <p>服務只綁 loopback，並檢查 Host 與 Origin；表單金鑰為每個行程隨機產生。</p>
      </div>
    </details>
    <p>本專題與任何積木製造商無關，介面不使用任何第三方商標、標誌或角色。</p>
  </div>
</footer>
</body>
</html>
"""

_START = """{% extends "base.html" %}
{% block main %}
{% if error %}
<div class="note bad" role="alert">
  <strong>這次沒有執行</strong>
  {{ error }}
</div>
{% endif %}
{% if notice %}
<div class="note warn" role="status">{{ notice }}</div>
{% endif %}

<div class="intro">
  <div>
    <h2 class="big">今天想蓋什麼？</h2>
    <p>告訴我你的點子。系統會配合手上的積木，找出適合的作品。</p>
  </div>
  <div class="brickscene" role="img"
    aria-label="三塊藍色積木組成的抽象圖形">
    <span></span><span></span><span></span>
  </div>
</div>

<form method="post" action="/result" accept-charset="utf-8">
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
{% if photo_handle %}
<input type="hidden" name="photo_handle" value="{{ photo_handle }}">
{% endif %}

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>說說你的點子</h2>
    <div class="grid2">
      <div>
        <label for="caption">作品描述</label>
        <textarea id="caption" name="caption" required
          maxlength="{{ max_caption }}"
          placeholder="例如：我想蓋一座小城堡，要有兩座塔和一個大門。"
          >{{ form.caption }}</textarea>
        <p class="hint">可以順便寫「最多幾顆」或想要的顏色。
          看得懂但這次用不上的條件，結果頁會逐條列出，不會靜默忽略。</p>
      </div>
      <aside class="quick" aria-label="快速開始" hidden>
        <h3>快速開始</h3>
        <div class="suggest">
          <button type="button"
            data-example="我想蓋一座小城堡，要有兩座塔和一個大門。">小城堡</button>
          <button type="button"
            data-example="我想蓋一台低矮的小火車。">小火車</button>
          <button type="button"
            data-example="我想蓋一座可以跨過小河的橋。">一座橋</button>
        </div>
      </aside>
    </div>
    <details class="adv">
      <summary>換一種設計方式</summary>
      <div class="inner">
        <div class="modes">
          {% for value in methods %}
          <label class="mode">
            <input type="radio" name="method" value="{{ value }}"
              {% if form.method == value %}checked{% endif %}>
            <span><strong>{{ method_labels[value] }}</strong>
              <span>{{ method_notes[value] }}</span></span>
          </label>
          {% endfor %}
        </div>
      </div>
    </details>
  </div>
</section>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>把積木放上工作桌</h2>
    <p class="lede">可以拍照幫忙辨識，也可以直接填寫數量。</p>
    {% if photo_handle %}
    <div class="note ok" role="status">
      <strong>已帶入照片數到的積木</strong>
      <span class="mono">{{ form.inventory_spec }}</span>
    </div>
    {% endif %}
    <div class="choice-row">
      <a class="choice" href="#photo-board">
        <strong>拍照幫我數</strong>
        <span>適合平放、沒有互相遮住的積木。</span></a>
      <div class="choice" aria-current="true">
        <strong>自己填數量</strong>
        <span>用下方的加減按鈕，快速整理手上的積木。</span></div>
    </div>
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
    <p class="hint">1x2 和 2x1 是同一種積木，直的橫的算在一起，不必分開填。
      目錄裡的作品多半要幾十顆，數量填大一點比較容易找到組得起來的。</p>

    <details class="adv">
      <summary>其他積木與進階設定</summary>
      <div class="inner">
        <label for="inventory_spec">庫存字串（與上面八格擇一）</label>
        <input type="text" id="inventory_spec" name="inventory_spec"
          value="{{ form.inventory_spec }}" maxlength="{{ max_spec }}"
          placeholder="2x4:10,1x2:8">
        <p class="hint">旋轉拼法正規化到同一項；兩種拼法同時給出會被拒絕，
          不會相加。八格與字串同時填寫也會被拒絕。</p>

        <h3>配色（可選）</h3>
        <label for="colour_stock">顏色庫存字串</label>
        <input type="text" id="colour_stock" name="colour_stock"
          value="{{ form.colour_stock }}" maxlength="{{ max_spec }}"
          placeholder="2x4:red:6,1x2:blue:4">
        <div class="swatchrow">
          {% for entry in palette %}
          <span><i style="background-color:{{ entry.hex }}"></i>
            <span class="mono">{{ entry.colour_id }}</span>
            {{ entry.label_zh }}</span>
          {% endfor %}
        </div>
        <p class="hint">留白就不配色，結構仍以 LDraw 預設色輸出。
          某形狀的顏色總量不足時會具名拒絕，不會虛構顏色。</p>

        <h3>其他</h3>
        <div class="pair">
          <div>
            <label for="top_n">Top-N</label>
            <input type="number" id="top_n" name="top_n" min="1"
              max="{{ max_top_n }}" step="1" value="{{ form.top_n }}">
          </div>
          <div>
            <label for="max_per_step">組裝每步最多幾顆</label>
            <input type="number" id="max_per_step" name="max_per_step" min="1"
              max="{{ max_per_step_cap }}" step="1"
              value="{{ form.max_per_step }}">
          </div>
        </div>
        <fieldset id="cpsat"
          {% if form.method != mode_pipeline %}disabled{% endif %}>
          <legend>CP-SAT 控制（只適用於{{ method_labels[mode_pipeline] }}）</legend>
          <label for="time_limit">time limit（秒）</label>
          <input type="number" id="time_limit" name="time_limit" min="0"
            max="{{ max_time_limit }}" step="0.5"
            value="{{ form.time_limit }}">
        </fieldset>
        <fieldset id="seedbox"
          {% if form.method == mode_rag %}disabled{% endif %}>
          <legend>seed（適用於 {{ method_labels[mode_pipeline] }} 與 {{
            method_labels[mode_project] }}）</legend>
          <label for="seed">seed（決定性）</label>
          <input type="number" id="seed" name="seed" min="0" step="1"
            value="{{ form.seed }}">
          <p class="hint">檢索是確定性的，所以 seed 對「{{
            method_labels[mode_rag] }}」沒有作用；填了會被<strong>具名拒絕</strong>，
            不會被靜默忽略。這個欄位有自己的區塊，是因為它同時屬於 CP-SAT 與
            解碼器——先前它放在 CP-SAT 區塊裡，選了正式模型時整個區塊被停用，
            seed 就被靜默丟掉了。</p>
        </fieldset>
        <fieldset id="decoder"
          {% if form.method != mode_project %}disabled{% endif %}>
          <legend>解碼器控制（只適用於{{ method_labels[mode_project] }}）</legend>
          <label class="mode" style="border:0;box-shadow:none;padding:0">
            <input type="checkbox" name="placement"
              {% if form.placement %}checked{% endif %}>
            <span>開啟 placement gate</span>
          </label>
          <p class="hint">{{ placement_notice }}</p>
        </fieldset>
        <p class="hint">不適用的欄位會被具名拒絕，不會被靜默忽略。
          頁面上的 JavaScript 只是漸進增強；<strong>伺服器才是權威</strong>，
          關掉 JavaScript 的結果是拒絕，不是靜默接受。</p>
        <noscript>
          <p class="hint">未啟用 JavaScript 時這兩組欄位不會自動停用；
            送出後仍由伺服器判定適用性並具名拒絕。</p>
        </noscript>
        <div class="note flat">
          <strong>這個介面不做什麼</strong>
          <ul>
            <li>不重新訓練、不調參、不重選 <span class="mono">final_H2</span>。</li>
            <li>不執行 Phase 3C，不讀取任何已封存的評估案例，不產生 Success@K。</li>
            <li>不把沒有通過靜態檢查的結構提供下載。</li>
          </ul>
        </div>
      </div>
    </details>

    <div class="actions">
      <a class="btn ghost" href="/reset">全部重新開始</a>
      <button class="btn" type="submit">開始設計</button>
    </div>
  </div>
</section>
</form>

<section class="card" id="photo-board">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>拍照幫我數</h2>
    <p class="lede">這一步可以跳過。把積木平放拍一張，系統先數一遍，
      你再逐一改成正確的數量。</p>
    <form method="post" action="/photo" enctype="multipart/form-data"
      accept-charset="utf-8">
      <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
      <label for="photo">選一張照片（PNG 或 JPEG，最大 {{ max_upload_mb }} MB）</label>
      <input type="file" id="photo" name="photo" accept="image/png,image/jpeg"
        required>
      <label id="modelabel">這張照片裡有</label>
      <div class="modes" role="group" aria-labelledby="modelabel">
        {% for value in photo_modes %}
        <label class="mode">
          <input type="radio" name="photo_mode" value="{{ value }}"
            {% if loop.first %}checked{% endif %}>
          <span><strong>{{ photo_mode_labels[value] }}</strong></span>
        </label>
        {% endfor %}
      </div>
      <details class="adv">
        <summary>辨識方法與已知限制</summary>
        <div class="inner">
          <div class="modes">
            <label class="mode">
              <input type="radio" name="recognise" value="{{ recognise_cv }}"
                checked>
              <span><strong>傳統 CV baseline</strong>
                <span>輪廓、長寬比與 stud 週期，全部可稽核，不需要權重。</span></span>
            </label>
            <label class="mode">
              <input type="radio" name="recognise"
                value="{{ recognise_learned }}"
                {% if not checkpoint_present %}disabled{% endif %}>
              <span><strong>學習模型（transfer ResNet-18）</strong>
                <span>{% if checkpoint_present %}使用
                  <span class="mono">{{ checkpoint_display }}</span>。
                {% else %}沒有可用的 checkpoint，因此停用。{% endif %}</span></span>
            </label>
          </div>
          <p class="hint">{{ capture_assumption }}</p>
          <p class="hint">{{ recognition_limit }}</p>
        </div>
      </details>
      <div class="actions">
        <button class="btn" type="submit">開始辨識</button>
      </div>
    </form>
  </div>
</section>

<script>
/* Progressive enhancement, in three parts.

   The fieldset sync is the load-bearing one: the server decides which fields
   apply, and with this script absent an inapplicable field is refused by name
   rather than silently ignored.

   The quick-start panel and the plus/minus buttons are conveniences. Both are
   *revealed or created* by script, so with JavaScript off nothing dead is on
   the page: the panel stays hidden and the eight number inputs are exactly
   what they always were. */
(function () {
  var cpsat = document.getElementById('cpsat');
  var decoder = document.getElementById('decoder');
  var seedbox = document.getElementById('seedbox');
  var radios = document.querySelectorAll('input[name="method"]');
  if (cpsat && decoder && seedbox && radios.length) {
    var sync = function () {
      var picked = null;
      radios.forEach(function (r) { if (r.checked) { picked = r.value; } });
      cpsat.disabled = picked !== {{ mode_pipeline | tojson }};
      decoder.disabled = picked !== {{ mode_project | tojson }};
      seedbox.disabled = picked === {{ mode_rag | tojson }};
    };
    radios.forEach(function (r) { r.addEventListener('change', sync); });
    sync();
  }

  var caption = document.getElementById('caption');
  var quick = document.querySelector('.quick');
  if (caption && quick) {
    quick.hidden = false;
    quick.querySelectorAll('[data-example]').forEach(function (button) {
      button.addEventListener('click', function () {
        caption.value = button.getAttribute('data-example');
        caption.focus();
      });
    });
  }

  var bump = function (input, delta) {
    var now = parseInt(input.value, 10);
    if (isNaN(now) || now < 0) { now = 0; }
    var next = now + delta;
    input.value = String(next < 0 ? 0 : next);
  };
  document.querySelectorAll('.parts .part').forEach(function (cell) {
    var input = cell.querySelector('input[type="number"]');
    var label = cell.querySelector('label');
    if (!input || !label) { return; }
    var row = document.createElement('div');
    row.className = 'counter';
    var less = document.createElement('button');
    less.type = 'button';
    less.textContent = '\\u2212';
    less.setAttribute('aria-label', '減少 ' + label.textContent + ' 的數量');
    var more = document.createElement('button');
    more.type = 'button';
    more.textContent = '\\uFF0B';
    more.setAttribute('aria-label', '增加 ' + label.textContent + ' 的數量');
    input.parentNode.insertBefore(row, input);
    row.appendChild(less);
    row.appendChild(input);
    row.appendChild(more);
    less.addEventListener('click', function () { bump(input, -1); });
    more.addEventListener('click', function () { bump(input, 1); });
  });
})();
</script>
{% endblock %}
"""

_PHOTO = """{% extends "base.html" %}
{% block main %}
{% if error %}
<div class="note bad" role="alert"><strong>這次沒有套用</strong>
  {{ error }}</div>
{% endif %}

<div class="note {{ 'warn' if analysis.unidentified else 'ok' }}" role="status">
  <strong>辨識完成：找到 {{ analysis.found }} 個項目，其中
    {{ analysis.unidentified }} 個沒有被命名</strong>
  數出來的只是建議。下面每一格都可以改，沒認出來的那幾個要你自己決定是什麼；
  系統不會替你猜一個最接近的。
</div>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>照片裡看到的東西</h2>
    <p class="lede">框線畫在照片上面，照片本身沒有被改過。
      橘色虛線代表不太確定，紅色虛線代表已經刪掉。</p>
    <div class="shot">
      <img src="/photo/{{ handle }}/image" width="{{ analysis.width }}"
        height="{{ analysis.height }}"
        alt="上傳的照片，{{ analysis.width }}×{{ analysis.height }} 像素，
          上面標了 {{ items | length }} 個偵測框">
      <svg viewBox="0 0 {{ analysis.width }} {{ analysis.height }}"
        preserveAspectRatio="none" aria-hidden="true">
        {% for item in items %}
        <rect x="{{ item.adopted_box[0] }}" y="{{ item.adopted_box[1] }}"
          width="{{ item.adopted_box[2] - item.adopted_box[0] }}"
          height="{{ item.adopted_box[3] - item.adopted_box[1] }}"
          class="{% if item.deleted %}gone{% elif item.adopted_part == unknown %}low{% endif %}"/>
        <text x="{{ item.adopted_box[0] + 8 }}"
          y="{{ item.adopted_box[1] + 32 }}">{{ item.index }}</text>
        {% endfor %}
      </svg>
    </div>
    <details class="adv">
      <summary>這張照片的技術資訊</summary>
      <div class="inner">
        <p class="hint">檔名 <span class="mono">{{ filename }}</span>，
          {{ analysis.width }}×{{ analysis.height }}，
          模式 {{ photo_mode_labels[analysis.mode] }}，
          辨識方法 <span class="mono">{{ analysis.method }}</span>。</p>
        <p class="hint">{{ capture_assumption }}</p>
        <p class="hint">{{ recognition_limit }}</p>
      </div>
    </details>
  </div>
</section>

<form method="post" action="/photo/{{ handle }}/correct" accept-charset="utf-8">
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>人工修正</h2>
    <p class="lede">改成正確的就好。你改過的地方會被記下來，
      系統原本猜的答案也會留著，不會被蓋掉。</p>
    <div class="scroll">
      <table class="edit">
        <caption>共 {{ items | length }} 個項目。</caption>
        <thead><tr>
          <th>#</th><th>來源</th><th>系統猜的</th><th>其他可能</th>
          <th>零件</th><th>數量</th><th>顏色</th><th>框 (x0,y0,x1,y1)</th>
          <th>刪除</th>
        </tr></thead>
        <tbody>
        {% for item in items %}
        <tr>
          <td class="num">{{ item.index }}</td>
          <td><span class="tag {{ item.source.split('+')[-1] if '+' in item.source else item.source }}"
            >{{ item.source }}</span>
            {% if item.changed_fields %}<br><span class="was"
              >改了 {{ item.changed_fields | join('、') }}</span>{% endif %}</td>
          <td><span class="mono">{{ item.predicted_part }}</span><br>
            <span class="was">信心 {{ '%.3f' % item.predicted_confidence }}</span>
            {% if item.predicted_colour %}<br><span class="was"
              >顏色 {{ item.predicted_colour }}</span>{% endif %}</td>
          <td class="mono">{{ item.predicted_top3 | join(' ') or '—' }}</td>
          <td>
            <select name="part_{{ item.index }}">
              <option value="">（不改）</option>
              <option value="{{ unknown }}">unknown</option>
              {% for part in parts %}
              <option value="{{ part }}"
                {% if item.edited_part == part %}selected{% endif %}
                >{{ part }}</option>
              {% endfor %}
            </select>
          </td>
          <td><input type="number" name="count_{{ item.index }}" min="0"
            step="1" max="{{ max_count }}"
            value="{{ item.edited_count if item.edited_count is not none else '' }}"
            placeholder="1"></td>
          <td>
            <select name="colour_{{ item.index }}">
              <option value="">（不改）</option>
              {% for name in colours %}
              <option value="{{ name }}"
                {% if item.edited_colour == name %}selected{% endif %}
                >{{ name }}</option>
              {% endfor %}
            </select>
          </td>
          <td><input type="text" name="box_{{ item.index }}"
            value="{{ item.adopted_box | join(',') }}"
            placeholder="x0,y0,x1,y1"></td>
          <td><input type="checkbox" name="delete_{{ item.index }}"
            {% if item.deleted %}checked{% endif %}></td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

    <h3>照片裡漏掉的可以自己補</h3>
    <div class="pair">
      <div>
        <label for="add_part">零件</label>
        <select id="add_part" name="add_part">
          <option value="">（不新增）</option>
          {% for part in parts %}<option value="{{ part }}">{{ part }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="add_count">數量</label>
        <input type="number" id="add_count" name="add_count" min="1"
          max="{{ max_count }}" step="1" value="1">
      </div>
    </div>
    <label for="add_colour">顏色（可留白）</label>
    <select id="add_colour" name="add_colour">
      <option value="">（不指定）</option>
      {% for name in colours %}<option value="{{ name }}">{{ name }}</option>
      {% endfor %}
    </select>
    <p class="hint">自己補的項目沒有系統預測，會如實記成
      <span class="mono">operator</span>。</p>

    <div class="actions">
      <a class="btn ghost" href="/">回上一步</a>
      <button class="btn" type="submit">套用修正</button>
    </div>
  </div>
</section>
</form>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>修正前／修正後庫存</h2>
    <p class="lede">修正前共 {{ before_total }} 塊，修正後共 {{ after_total }} 塊。
      只有「修正後」的數字會被拿去用。</p>
    <div class="scroll">
      <table>
        <caption>只有「採用值」會進入庫存；仍是 unknown 的項目不計入任何零件。</caption>
        <thead><tr><th>零件</th><th class="num">修正前</th>
          <th class="num">修正後</th><th>變化</th></tr></thead>
        <tbody>
        {% for row in inventory_rows %}
        <tr>
          <td><span class="chip" aria-hidden="true"
            style="background-color:{{ part_colours[row.part] }}"></span>
            <span class="mono">{{ row.part }}</span></td>
          <td class="num">{{ row.before }}</td>
          <td class="num">{{ row.after }}</td>
          <td>{% if row.after == row.before %}<span class="g">—</span>
            {% else %}<span class="pill na">{{ '%+d' % (row.after - row.before) }}</span>
            {% endif %}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="hint">人工改動 {{ corrected.edited_items }} 個項目。
      {% if corrected.unresolved_items %}仍未辨識的項目：
      <span class="mono">{{ corrected.unresolved_items | join(', ') }}</span>
      ——它們不計入庫存。{% endif %}</p>
    {% if corrected.colour_parts %}
    <p class="hint">顏色庫存：<span class="mono">{{ colour_spec }}</span>
      {% if not corrected.fully_coloured %}（不完整：有項目沒有顏色，
      因此配色器不會用這份顏色庫存）{% endif %}</p>
    {% endif %}
  </div>
</section>

<form method="post" action="/result" accept-charset="utf-8">
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
<input type="hidden" name="photo_handle" value="{{ handle }}">
<input type="hidden" name="inventory_spec" value="{{ adopted_spec }}">
{% if corrected.fully_coloured %}
<input type="hidden" name="colour_stock" value="{{ colour_spec }}">
{% endif %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>用這些積木繼續</h2>
    {% if adopted_spec %}
    <p class="lede">要用的積木是 <span class="mono">{{ adopted_spec }}</span>。
      這份庫存和手打進去的走完全一樣的檢查。</p>
    <label for="caption2">作品描述</label>
    <textarea id="caption2" name="caption" required maxlength="{{ max_caption }}"
      placeholder="例如：我想蓋一台低矮的小火車。"></textarea>
    <details class="adv">
      <summary>換一種設計方式</summary>
      <div class="inner">
        <div class="modes">
          {% for value in methods %}
          <label class="mode">
            <input type="radio" name="method" value="{{ value }}"
              {% if loop.first %}checked{% endif %}>
            <span><strong>{{ method_labels[value] }}</strong></span>
          </label>
          {% endfor %}
        </div>
      </div>
    </details>
    <div class="actions">
      <button class="btn" type="submit">開始設計</button>
    </div>
    {% else %}
    <p class="lede">修正後一個積木都沒有，所以沒辦法繼續。
      請至少給一個項目指定零件與數量。</p>
    {% endif %}
  </div>
</section>
</form>
{% endblock %}
"""

_RESULT = """{% extends "base.html" %}
{% block main %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <div class="result-grid">
      {% if finished %}
      <figure class="preview">
        <img src="/artifact/{{ handle }}/preview.png"
          width="{{ finished.preview_width }}"
          height="{{ finished.preview_height }}"
          alt="選中結果的 3D 幾何預覽：以 {{ report.result.n_bricks }} 個軸對齊
            長方體畫出每一塊積木。這是幾何檢視，不是寫實渲染。">
        <figcaption>形狀示意圖，不是照片，也沒有計算會不會倒。</figcaption>
      </figure>
      {% endif %}
      <div class="result-copy">
        {% if ready %}
        <p class="ready">{{ icons['pass'] | safe }}你的積木足夠</p>
        <h2>{{ report.request.caption }}</h2>
        <p>需要 {{ report.result.n_bricks }} 顆積木，數量確認足夠，
          也沒有兩塊積木重疊。</p>
        <div class="tally">
          <span>總共<b>{{ report.result.n_bricks }} 顆</b></span>
          {% if finished and finished.plan %}
          <span>分成<b>{{ finished.plan.n_steps }} 步</b></span>
          {% endif %}
        </div>
        {% else %}
        <p class="ready no">{{ icons['fail'] | safe }}這次沒做出來</p>
        <h2>沒有找到組得起來的作品</h2>
        <p>{{ why.lead }}</p>
        {% if why.missing %}
        <p>最接近的一件要用 {{ why.needed }} 顆積木，你手上還差
          {{ why.short }} 顆：</p>
        <div class="swatchrow">
          {% for part, count in why.missing %}
          <span><i style="background-color:{{ part_colours[part] }}"></i>
            <span class="mono">{{ part }}</span> 還差 {{ count }} 顆</span>
          {% endfor %}
        </div>
        {% endif %}
        <p class="hint">{{ why.advice }}</p>
        {% endif %}
        <div class="actions" style="justify-content:flex-start">
          {% if finished and finished.plan %}
          <a class="btn" href="/steps/{{ handle }}/1">開始組裝</a>
          <a class="btn ghost" href="/">換一個</a>
          {% else %}
          <a class="btn" href="/">換一個</a>
          <a class="btn ghost" href="/reset">全部重新開始</a>
          {% endif %}
        </div>
        {% if finished and finished.plan_problem %}
        <div class="note warn" style="margin-top:18px">
          <strong>這件作品排不出組裝步驟</strong>
          有積木被上面的積木吊著，從下往上疊不出來，所以沒有一步一步的教學。
          原始說明：<span class="mono">{{ finished.plan_problem }}</span>
        </div>
        {% endif %}
        <details class="adv">
          <summary>下載模型與技術資訊</summary>
          <div class="inner">
            {% if finished %}
            <div class="actions" style="justify-content:flex-start;margin-top:0">
              <a class="btn ghost" href="/artifact/{{ handle }}/model.ldr"
                download>下載 LDraw（.ldr）</a>
            </div>
            <p class="hint">以實際組裝順序寫入 <span class="mono">0 STEP</span>。
              檔案只存在本機記憶體與這次連線，不寫入專案的
              <span class="mono">artifacts/</span>。</p>
            {% endif %}
            {% if ready %}
            <p class="hint">找到一件通過靜態交付檢查的結果。預覽、配色、組裝步驟
              與 LDraw 下載全部出自同一份磚清單。這是該件輸出的確定性檢查，
              <em>不是</em>成功率，也不是任何模型指標。</p>
            {% else %}
            <p class="hint">流程正常完成，但本次沒有可交付結果：
              沒有預覽、沒有配色、沒有組裝步驟，也沒有下載。
              {{ not_ready_reason }}</p>
            {% endif %}
            <dl class="kv">
              <dt>方法</dt><dd>{{ method_labels[result.method] }}
                （<span class="mono">{{ result.method }}</span>）</dd>
              <dt>執行狀態</dt><dd><span class="mono">{{ result.status }}</span></dd>
              {% for key, value in provenance_rows %}
              <dt>{{ key }}</dt><dd class="mono">{{ value }}</dd>
              {% endfor %}
              <dt>庫存來源</dt><dd>{{ inventory_origin }}</dd>
              <dt>手動／採用庫存</dt><dd class="mono">{{ inventory_spec }}</dd>
            </dl>
            <p class="hint">以上是這次執行的方法與 provenance。</p>
            {% if provenance_notices %}
            {% for notice in provenance_notices %}
            <div class="note warn" style="margin-top:14px">{{ notice }}</div>
            {% endfor %}
            {% endif %}
          </div>
        </details>
      </div>
    </div>
  </div>
</section>

{% if finished %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>顏色</h2>
    {% if finished.colour_problem %}
    <div class="note bad"><strong>沒有配色</strong>
      {{ finished.colour_problem }}</div>
    <p class="hint">結構、預覽與組裝步驟仍然產生：LDraw 使用預設色，
      預覽與步驟圖改用「一種形狀一種顏色」的辨識用圖例。
      這時圖上的顏色<strong>不是</strong>檔案裡的顏色。
      顏色庫存不足時具名拒絕，不會虛構顏色。</p>
    {% elif finished.assignment %}
    <p class="lede">確定性配色：由下層往上、依偏好順序指派，
      每一次執行結果相同，且不超過任何顏色庫存。</p>
    <div class="swatchrow">
      {% for row in colour_rows %}
      <span><i style="background-color:{{ row.hex }}"></i>
        <span class="mono">{{ row.part }}</span>
        {{ row.colour_id }} ×{{ row.count }}</span>
      {% endfor %}
    </div>
    <p class="hint">偏好顏色用到 {{ finished.assignment.preferred_count }} 塊，
      其餘 {{ finished.assignment.non_preferred_count }} 塊退到其他顏色。
      LDraw、3D 預覽與每一張步驟圖都用<strong>同一份</strong>配色結果，
      顏色值取自同一張調色盤，因此畫面上的顏色就是檔案裡的顏色。</p>
    {% else %}
    <p class="lede">你沒有指定顏色，所以圖上是「一種形狀一種顏色」的辨識用配色。
      形狀是對的，顏色只是幫你分辨，下載的檔案用的是預設色。</p>
    {% endif %}
  </div>
</section>
{% endif %}

{% if conditions %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>從你寫的那句話讀到什麼</h2>
    <ul>
      {% for line in conditions.lines %}<li>{{ line }}</li>{% endfor %}
    </ul>
    {% if conditions.unresolved %}
    <div class="note warn">
      <strong>以下條件被看到但沒有套用，已具名回報</strong>
      <ul>{% for item in conditions.unresolved %}
        <li><span class="mono">{{ item.field }}</span>
          「{{ item.text }}」：{{ item.reason }}</li>{% endfor %}</ul>
    </div>
    {% endif %}
    {% if conditions.not_applied %}
    <p class="hint">理解了但不影響檢索的條件：
      {% for item in conditions.not_applied %}
      <span class="mono">{{ item.field }}</span>（{{ item.reason }}）
      {% endfor %}</p>
    {% endif %}
    {% if explanation %}
    <details class="adv">
      <summary>為什麼挑這一件</summary>
      <div class="inner">
        <p class="hint">每一句都對應一個可查的數值；沒有語言模型參與。</p>
        {% for line in explanation.header %}<p class="hint">{{ line }}</p>{% endfor %}
        {% for candidate in explanation.candidates %}
        <h3>候選 {{ loop.index }}
          <span class="mono">{{ candidate.evidence.catalog_id }}</span></h3>
        <ul>{% for line in candidate.sentences %}<li>{{ line }}</li>{% endfor %}</ul>
        {% endfor %}
        <div class="note flat"><strong>{{ explanation.selection }}</strong></div>
        {% if explanation.colour_note %}
        <p class="hint">{{ explanation.colour_note }}</p>{% endif %}
        {% for note in explanation.notes %}<p class="hint">{{ note }}</p>{% endfor %}
      </div>
    </details>
    {% endif %}
  </div>
</section>
{% endif %}

{% if attempts or report %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>技術資訊</h2>
    <p class="lede">這一區是研究紀錄用的，看不懂可以略過。</p>
    {% if attempts %}
    <details class="adv">
      <summary>F-pipeline 各候選</summary>
      <div class="inner">
        <p class="hint">依檢索順序嘗試，取第一個通過獨立復驗的結果。
          逾時與無解是兩種不同狀態。</p>
        <div class="scroll">
          <table>
            <thead><tr><th>#</th><th>catalog_id</th><th>求解狀態</th>
              <th class="num">秒</th><th>回傳鋪排</th><th>exact cover</th>
              <th>庫存</th><th>碰撞</th><th>邊界</th><th>接地</th><th>連通</th>
              <th>可交付</th><th>失敗理由</th></tr></thead>
            <tbody>
            {% for row in attempts %}
            <tr>
              <td class="num">{{ loop.index }}</td>
              <td class="mono">{{ row.catalog_id }}</td>
              <td><span class="mono">{{ row.solver_status }}</span><br>
                <span class="g">{{ solver_gloss.get(row.solver_status,
                  row.solver_status) }}</span></td>
              <td class="num">{{ '%.3f' % row.wall_seconds }}</td>
              {% for key in attempt_flags %}
              <td>{% if row[key] %}<span class="pill pass">
                  {{ icons['pass'] | safe }} pass</span>
                {% else %}<span class="pill fail">
                  {{ icons['fail'] | safe }} FAIL</span>{% endif %}</td>
              {% endfor %}
              <td>{{ row.failure or '—' }}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </details>
    {% endif %}
    {% if report %}
    <details class="adv">
      <summary>選中結果的靜態檢查</summary>
      <div class="inner">
        <p class="hint">全部由 <span class="mono">{{ report.scored_by }}</span>
          計算並直接引用；本介面沒有第二份判定邏輯。</p>
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
        <p class="hint">交付判定用的是不含終止原因的九項：
          <span class="mono">{{ delivery_checks | join(' ') }}</span>。</p>
      </div>
    </details>
    <details class="adv">
      <summary>庫存使用量與磚清單</summary>
      <div class="inner">
        <div class="scroll">
          <table>
            <caption>負數如實印成負數並標記 OVERDRAWN，不會夾到 0。</caption>
            <thead><tr><th>零件</th><th class="num">備料</th>
              <th class="num">使用</th><th class="num">剩餘</th>
              <th>狀態</th></tr></thead>
            <tbody>
            {% for row in inventory_rows %}
            <tr>
              <td><span class="chip" aria-hidden="true"
                style="background-color:{{ part_colours[row.part] }}"></span>
                <span class="mono">{{ row.part }}</span></td>
              <td class="num">{{ row.stocked }}</td>
              <td class="num">{{ row.used }}</td>
              <td class="num">{{ row.left }}</td>
              <td>{% if row.left < 0 %}<span class="pill fail">
                  {{ icons['fail'] | safe }} OVERDRAWN</span>
                {% else %}<span class="pill pass">
                  {{ icons['pass'] | safe }} 足夠</span>{% endif %}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
        </div>
        <h3>磚清單（{{ report.result.n_bricks }} 塊）</h3>
        <pre class="text">{{ report.result.text.rstrip() }}</pre>
        <p class="hint">磚清單印的是實際擺放方向，庫存表印的是正規化後的項目，
          因此兩張表的拼法可能不同，數量仍然對得上。</p>
      </div>
    </details>
    {% endif %}
  </div>
</section>
{% endif %}
{% endblock %}
"""

_STEPS = """{% extends "base.html" %}
{% block main %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>組裝步驟 {{ number }} / {{ total }}</h2>
    <p class="lede">{{ description }}</p>
    <div class="stepbar">
      {% if number > 1 %}
      <a class="btn ghost" href="/steps/{{ handle }}/{{ number - 1 }}">上一步</a>
      {% endif %}
      <span class="count">{{ number }} / {{ total }}</span>
      {% if number < total %}
      <a class="btn" href="/steps/{{ handle }}/{{ number + 1 }}">下一步</a>
      {% endif %}
      <a class="btn ghost" href="/result/{{ handle }}">回到作品</a>
    </div>
    <figure class="preview">
      <img src="/steps/{{ handle }}/{{ number }}/image"
        alt="第 {{ number }} 步的累積結構，共 {{ build.total_bricks }} 塊積木">
      <figcaption>疊到這一步的樣子，目前共 {{ build.total_bricks }} 塊。</figcaption>
    </figure>
  </div>
</section>

<div class="grid2">
  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>這一步加入</h2>
      <ul>
        {% for part, count in build.added_parts.items() %}
        <li><span class="mono">{{ part }}</span> ×{{ count }}</li>
        {% endfor %}
      </ul>
      <p class="hint">每一步最多加 {{ plan.max_per_step }} 顆。
        新加的積木都會壓在已經放好的積木上面；也可以先分開疊幾堆，
        最後再用長條積木接起來。</p>
      <p class="hint">目前分成 {{ build.components }} 堆。中間可以有很多堆，
        只有做完的時候要連成一整件。這裡講的「連起來」只是形狀有沒有疊在一起，
        <strong>不是物理支撐</strong>，也沒有算會不會倒。</p>
      <details class="adv">
        <summary>這一步的靜態檢查</summary>
        <div class="inner">
          <div class="checks">
            {% for row in step_checks %}
            <div>
              <span class="pill {{ row.css }}">{{ icons[row.icon] | safe }}
                {{ row.label }}</span>
              <span class="k">{{ row.name }}</span>
              <span class="g">{{ row.gloss }}</span>
            </div>
            {% endfor %}
          </div>
        </div>
      </details>
    </div>
  </section>

  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>累積零件表</h2>
      <ol class="parts">
        {% for part, count in build.cumulative_parts.items() %}
        <li><span class="mono">{{ part }}</span> ×{{ count }}</li>
        {% endfor %}
      </ol>
      <p class="hint">目前用掉 {{ build.total_bricks }} 塊，
        整件作品共 {{ plan.n_bricks }} 塊。</p>
      {% if build.stock_remaining %}
      <h3>還沒用到的積木</h3>
      <p class="mono">{{ build.stock_remaining | dictsort
        | map('join', ':') | join(', ') }}</p>
      {% endif %}
      <details class="adv">
        <summary>全部步驟一覽</summary>
        <div class="inner">
          <ol>
            {% for line in descriptions %}
            <li{% if loop.index == number %} style="font-weight:500"{% endif %}
              >{{ line }}</li>
            {% endfor %}
          </ol>
        </div>
      </details>
    </div>
  </section>
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
    <div class="actions" style="justify-content:flex-start">
      <a class="btn" href="/">回到第一步</a>
      <a class="btn ghost" href="/reset">全部重新開始</a>
    </div>
  </div>
</section>
{% endblock %}
"""

_ENV = Environment(
    loader=DictLoader({
        "base.html": _BASE,
        "start.html": _START,
        "photo.html": _PHOTO,
        "result.html": _RESULT,
        "steps.html": _STEPS,
        "error.html": _ERROR,
    }),
    autoescape=True, undefined=StrictUndefined,
    trim_blocks=True, lstrip_blocks=True,
)

ATTEMPT_FLAGS = ("solver_returned_tiling", "exact_cover_verified",
                 "inventory_verified", "collision_free", "in_bounds",
                 "touches_ground", "stud_only_connected", "delivery_ready")

METHOD_NOTES = {
    METHOD_RAG: ("多語 embedding 語意檢索 train split 的既有作品，"
                 "再以精確庫存與靜態結構條件重排並產生有根據說明。"),
    METHOD_PIPELINE: ("取回 train 形狀後，交給既有 CP-SAT 依庫存重新鋪磚，"
                      "再獨立復驗 exact cover、庫存、碰撞、邊界、接地與連通。"),
    METHOD_PROJECT: ("以封存的 final_H2 做一次展示解碼。硬庫存 gate 生效，"
                     "placement gate 預設關閉。不是批次、不是評估。"),
}

STEP_CHECK_GLOSS = {
    "collision_free": "沒有兩塊佔用同一格",
    "in_bounds": f"全部落在 {WORLD}×{WORLD}×{WORLD} 世界內",
    "touches_ground": "有磚位於 z = 0",
    "within_stock": "累積用量沒有超出庫存",
}

_COMMON = {
    "css": CSS,
    "extra_css": _EXTRA_CSS,
    "icons": ICONS,
    "steps": STEPS,
    "parts": PART_VOCAB,
    "part_colours": PART_COLOURS,
    "colours": COLOUR_ORDER,
    "palette": PALETTE,
    "methods": (METHOD_RAG, METHOD_PIPELINE, METHOD_PROJECT),
    "method_labels": METHOD_LABELS,
    "mode_pipeline": METHOD_PIPELINE,
    "mode_project": METHOD_PROJECT,
    "mode_rag": METHOD_RAG,
    "method_notes": METHOD_NOTES,
    "photo_modes": (PHOTO_SINGLE, PHOTO_MULTI),
    "photo_mode_labels": PHOTO_MODE_LABELS,
    "recognise_cv": RECOGNISE_CV,
    "recognise_learned": RECOGNISE_LEARNED,
    "capture_assumption": CAPTURE_ASSUMPTION_ZH,
    "recognition_limit": RECOGNITION_LIMIT_ZH,
    "solver_gloss": SOLVER_GLOSS,
    "attempt_flags": ATTEMPT_FLAGS,
    "unknown": "unknown",
}


def blank_form() -> dict:
    """The field state before anything has been submitted.

    ``time_limit`` starts **empty**, not at its default. It only applies to the
    F-pipeline, and an inapplicable field carrying a value is refused by name --
    so pre-filling it made the default method refuse every first submission. A
    real browser walk-through found that; the server was right and the form was
    wrong.
    """
    return {"caption": "", "method": METHOD_RAG, "grid": {},
            "inventory_spec": "", "colour_stock": "", "top_n": 10,
            "time_limit": "", "seed": "", "max_per_step": 1,
            "placement": False}


def form_state(fields) -> dict:
    """Echo a submission back so a refusal does not discard what was typed."""
    def one(name, default=""):
        values = fields.get(name) or []
        return values[0] if values else default

    grid = {}
    for part in PART_VOCAB:
        raw = one(f"qty_{part}").strip()
        if raw:
            grid[part] = raw
    method = one("method", METHOD_RAG)
    return {
        "caption": one("caption"),
        "method": method if method in METHOD_LABELS else METHOD_RAG,
        "grid": grid,
        "inventory_spec": one("inventory_spec"),
        "colour_stock": one("colour_stock"),
        "top_n": one("top_n", "10"),
        "time_limit": one("time_limit"),
        "seed": one("seed"),
        "max_per_step": one("max_per_step", "1"),
        "placement": bool(fields.get("placement")),
    }


def render_start(*, csrf_token: str, form=None, error=None, notice=None,
                 photo_handle=None, checkpoint_present: bool = False,
                 checkpoint_display: str = "") -> str:
    from src.ui.app import (MAX_CAPTION_CHARS, MAX_INVENTORY_SPEC_CHARS,
                            MAX_TIME_LIMIT, MAX_TOP_N, UiError)
    from src.ui.model_entry import PLACEMENT_NOTICE
    from src.ui.upload import MAX_UPLOAD_BYTES

    if not isinstance(csrf_token, str) or not csrf_token:
        raise UiError(
            "page one may not be rendered without the form key it has to "
            "carry; a form that cannot be submitted is not a page")
    return _ENV.get_template("start.html").render(
        title="說想法", step=1, form=form or blank_form(), error=error,
        notice=notice, csrf_token=csrf_token, photo_handle=photo_handle,
        checkpoint_present=checkpoint_present,
        checkpoint_display=checkpoint_display,
        max_caption=MAX_CAPTION_CHARS, max_spec=MAX_INVENTORY_SPEC_CHARS,
        max_top_n=MAX_TOP_N, max_time_limit=MAX_TIME_LIMIT,
        max_per_step_cap=8, max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
        placement_notice=PLACEMENT_NOTICE, **_COMMON)


def render_photo(*, csrf_token: str, handle: str, filename: str, analysis,
                 items, before, corrected, error=None) -> str:
    from src.ui.app import MAX_CAPTION_CHARS
    from src.ui.corrections import MAX_COUNT, colour_stock_spec, inventory_spec

    rows = []
    for part in PART_VOCAB:
        first = before.parts.get(part, 0)
        second = corrected.parts.get(part, 0)
        if first or second:
            rows.append({"part": part, "before": first, "after": second})
    return _ENV.get_template("photo.html").render(
        title="放積木", step=2, csrf_token=csrf_token, handle=handle,
        filename=filename, analysis=analysis.as_dict(), items=items,
        corrected=corrected.as_dict(), inventory_rows=rows,
        before_total=before.total, after_total=corrected.total,
        adopted_spec=inventory_spec(corrected.parts),
        colour_spec=colour_stock_spec(corrected.colour_parts),
        max_count=MAX_COUNT, error=error,
        max_caption=MAX_CAPTION_CHARS, **_COMMON)


def check_rows(checks: dict) -> list[dict]:
    """The static checks as rows, with the three answers as text."""
    out = []
    for name, gloss in CHECK_GLOSS.items():
        if name not in checks:
            continue
        value = checks[name]
        if value is None:
            css, icon, label = "na", "na", "n/a"
        elif value:
            css, icon, label = "pass", "pass", "pass"
        else:
            css, icon, label = "fail", "fail", "FAIL"
        out.append({"name": name, "gloss": gloss, "css": css, "icon": icon,
                    "label": label})
    return out


def why_not_ready(result, explanation) -> dict:
    """A plain-language reason for a run that delivered nothing.

    Every number here was already computed by the retrieval explanation. This
    recomputes nothing and judges nothing: it lifts the shortfall out of the
    collapsed technical block, where a person reading "沒有找到組得起來的作品"
    could not see the one thing that tells them what to do next -- which parts
    were short, and by how many.
    """
    out = {"lead": "", "missing": [], "needed": 0, "short": 0, "advice": ""}
    if result.status == "no_semantic_candidate":
        out["lead"] = "目錄裡沒有找到文字上接近的作品，所以沒有東西可以比對庫存。"
        out["advice"] = "換一種說法，或把想蓋的東西寫得更具體一點，再試一次。"
        return out

    best = None
    for entry in (explanation or {}).get("candidates") or []:
        evidence = entry.get("evidence") or {}
        if "missing_total" not in evidence:
            continue
        if best is None or evidence["missing_total"] < best["missing_total"]:
            best = evidence

    if best is None:
        out["lead"] = "系統試過取回來的候選，但沒有一件能用現在的積木組出來。"
        out["advice"] = "把積木數量填多一點，或把想法寫得更簡單一點，再試一次。"
        return out

    if best.get("missing_parts"):
        out["missing"] = sorted(best["missing_parts"].items())
        out["needed"] = best.get("required_total") or 0
        out["short"] = best["missing_total"]
        out["lead"] = "積木不夠。"
        out["advice"] = ("把缺的補齊就能組這一件；"
                         "或在「其他積木與進階設定」裡把候選數量調大，多找幾件。")
        return out

    out["lead"] = "積木數量是夠的，但取回來的作品沒有通過結構檢查（接地或連通）。"
    out["advice"] = "換一種說法再試一次，或把候選數量調大，多找幾件。"
    return out


def render_result(*, result, handle, inventory_spec: str,
                  inventory_origin: str, finished=None,
                  not_ready_reason: str = "") -> str:
    from src.ui.app import load_delivery

    report = result.report
    inventory_rows = []
    if report is not None:
        stocked = report["request"]["inventory"]
        used = report["inventory"]["used"]
        left = report["inventory"]["remaining"]
        inventory_rows = [
            {"part": part, "stocked": count, "used": used.get(part, 0),
             "left": left[part]} for part, count in stocked.items()]

    provenance_rows = []
    notices = []
    for key, value in (result.provenance or {}).items():
        if key.endswith("notice") and value:
            notices.append(value)
            continue
        if value is None or isinstance(value, (dict, list)):
            continue
        provenance_rows.append((key, value))

    conditions = None
    explanation = None
    if result.explanation is not None:
        explanation = result.explanation
        block = result.evidence.get("conditions") or {}
        conditions = {
            "lines": _condition_lines(block),
            "unresolved": block.get("unresolved") or [],
            "not_applied": result.evidence.get(
                "conditions_not_applied_to_retrieval") or [],
        }

    colour_rows = []
    if finished is not None and finished.assignment is not None:
        from src.colour.palette import BY_ID

        counted: dict[tuple[str, str], int] = {}
        for brick in finished.assignment.bricks:
            key = (brick.part, brick.colour_id)
            counted[key] = counted.get(key, 0) + 1
        colour_rows = [
            {"part": part, "colour_id": name, "count": count,
             "hex": BY_ID[name].hex}
            for (part, name), count in sorted(counted.items())]

    return _ENV.get_template("result.html").render(
        title="看作品", step=3, result=result, handle=handle,
        why=(why_not_ready(result, explanation) if not result.ready else None),
        ready=result.ready, report=report,
        checks=check_rows(report["checks"]) if report else [],
        delivery_checks=list(load_delivery().DELIVERY_CHECKS),
        inventory_rows=inventory_rows, inventory_spec=inventory_spec,
        inventory_origin=inventory_origin,
        provenance_rows=provenance_rows, provenance_notices=notices,
        conditions=conditions, explanation=explanation,
        attempts=(result.evidence.get("attempts")
                  if result.method == METHOD_PIPELINE else None),
        finished=finished, colour_rows=colour_rows,
        not_ready_reason=not_ready_reason, **_COMMON)


def _condition_lines(block: dict) -> list[str]:
    out = []
    if block.get("category"):
        out.append(f"類別：{block['category']}")
    if block.get("max_parts") is not None:
        out.append(f"最大零件數：{block['max_parts']}")
    if block.get("preferred_colours"):
        out.append("偏好顏色：" + "、".join(block["preferred_colours"]))
    if block.get("allow_colour_substitution") is not None:
        out.append("允許替代顏色" if block["allow_colour_substitution"]
                   else "不允許替代顏色")
    if block.get("mode"):
        out.append(f"模式：{block['mode']}")
    return out or ["沒有抽取到任何結構化條件；整段文字只用於語意檢索"]


def render_steps(*, handle: str, plan, number: int, descriptions) -> str:
    step = plan.steps[number - 1]
    rows = []
    for name, gloss in STEP_CHECK_GLOSS.items():
        value = getattr(step, name)
        if value is None:
            css, icon, label = "na", "na", "n/a"
        elif value:
            css, icon, label = "pass", "pass", "pass"
        else:
            css, icon, label = "fail", "fail", "FAIL"
        rows.append({"name": name, "gloss": gloss, "css": css, "icon": icon,
                     "label": label})
    return _ENV.get_template("steps.html").render(
        title="組裝步驟", step=3, handle=handle, plan=plan.as_dict(),
        number=number, total=plan.n_steps, step_checks=rows,
        build=step.as_dict(), descriptions=list(descriptions),
        description=descriptions[number - 1], **_COMMON)


def render_error(*, heading: str, detail: str, advice: str,
                 title: str = "無法完成", step: int = 1) -> str:
    return _ENV.get_template("error.html").render(
        title=title, step=step, heading=heading, detail=detail, advice=advice,
        **_COMMON)
