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

STEPS = ("庫存與需求", "照片辨識與修正", "結果與交付", "組裝步驟")

_EXTRA_CSS = """
/* --- the detection overlay ------------------------------------------- */
.shot { position:relative; display:inline-block; max-width:100%;
  border:3px solid var(--line-deep); border-radius:var(--radius-sm);
  overflow:hidden; background:var(--paper-2); }
.shot img { display:block; max-width:100%; height:auto; }
.shot svg { position:absolute; inset:0; width:100%; height:100%; }
.shot svg rect { fill:none; stroke:#1D4ED8; stroke-width:4;
  vector-effect:non-scaling-stroke; }
.shot svg rect.low { stroke:#8A4B00; stroke-dasharray:10 6; }
.shot svg rect.gone { stroke:#B3261E; stroke-dasharray:3 7; }
.shot svg text { fill:#FFFFFF; font-weight:700; font-size:26px;
  paint-order:stroke; stroke:#1C222C; stroke-width:6; }

/* --- the per-item correction table ----------------------------------- */
table.edit td, table.edit th { vertical-align:top; }
table.edit input[type=number] { max-width:9ch; }
table.edit input[type=text] { max-width:16ch; }
table.edit select { min-width:11ch; }
.tag { display:inline-block; border:2px solid var(--line-deep);
  border-radius:999px; padding:1px 9px; font-size:.78rem;
  background:var(--paper-2); }
.tag.model { background:var(--na-bg); }
.tag.mixed { background:var(--warn-bg); color:var(--warn); }
.tag.operator { background:var(--ok-bg); color:var(--ok); }
.was { color:var(--ink-soft); font-size:.8rem; }

/* --- step navigation -------------------------------------------------- */
.stepbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  margin:14px 0; }
.stepbar .count { font-weight:700; }
ol.parts { columns:2; margin:0; padding-left:1.4em; }
@media (min-width:720px) { ol.parts { columns:3; } }
.swatchrow { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }
.swatchrow span { display:inline-flex; align-items:center; gap:6px;
  border:2px solid var(--line); border-radius:999px; padding:2px 10px;
  font-size:.82rem; background:var(--paper); }
.swatchrow i { width:14px; height:14px; border-radius:3px;
  border:1px solid rgba(28,34,44,.5); display:inline-block; }
"""

_BASE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{{ title }}｜BrickAgain 完整介面</title>
<style>{{ css | safe }}{{ extra_css | safe }}</style>
</head>
<body>
<a class="skip" href="#main">跳到主要內容</a>
<header class="top">
  <div class="studs" aria-hidden="true"></div>
  <div class="wrap">
    <div class="brandline">
      <h1>BrickAgain 完整介面</h1>
      <span class="tag">本機 · 離線 · 只綁 loopback</span>
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
    <p>本頁任何數字都不是指標，不可與已封存的 Phase 2 評估比較，
      也不得改名成 Structural／Semantic／Full Success@K。</p>
    <p>連通性是相鄰層 footprint 交集，接觸地面是有磚位於 z = 0；
      兩者都是靜態幾何，不是物理支撐，也不是穩定性分析。</p>
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
  <strong>這次沒有執行，原因如下</strong>
  {{ error }}
</div>
{% endif %}
{% if notice %}
<div class="note warn" role="status">{{ notice }}</div>
{% endif %}

<div class="grid2">
  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>照片辨識（可選）</h2>
      <p class="lede">上傳一張照片，先辨識再修正，修正後的庫存會帶進下一步。
        沒有照片就直接用右邊的手動庫存。</p>
      <form method="post" action="/photo" enctype="multipart/form-data"
        accept-charset="utf-8">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        <label for="photo">照片檔（PNG 或 JPEG，上限 {{ max_upload_mb }} MB）</label>
        <input type="file" id="photo" name="photo" accept="image/png,image/jpeg"
          required>
        <h3>這張照片是</h3>
        <div class="modes">
          {% for value in photo_modes %}
          <label class="mode">
            <input type="radio" name="photo_mode" value="{{ value }}"
              {% if loop.first %}checked{% endif %}>
            <span><strong>{{ photo_mode_labels[value] }}</strong></span>
          </label>
          {% endfor %}
        </div>
        <h3>辨識方法</h3>
        <div class="modes">
          <label class="mode">
            <input type="radio" name="recognise" value="{{ recognise_cv }}"
              checked>
            <span><strong>傳統 CV baseline</strong>
              <span>輪廓、長寬比與 stud 週期，全部可稽核，不需要權重。</span></span>
          </label>
          <label class="mode">
            <input type="radio" name="recognise" value="{{ recognise_learned }}"
              {% if not checkpoint_present %}disabled{% endif %}>
            <span><strong>學習模型（transfer ResNet-18）</strong>
              <span>{% if checkpoint_present %}使用
                <span class="mono">{{ checkpoint_display }}</span>。
              {% else %}沒有可用的 checkpoint，因此停用。{% endif %}</span></span>
          </label>
        </div>
        <p class="hint">{{ capture_assumption }}</p>
        <p class="hint">{{ recognition_limit }}</p>
        <div class="actions">
          <button class="btn" type="submit">辨識這張照片</button>
        </div>
      </form>
    </div>
  </section>

  <section class="card">
    <div class="studs" aria-hidden="true"></div>
    <div class="body">
      <h2>手動庫存與需求</h2>
      <form method="post" action="/result" accept-charset="utf-8">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        {% if photo_handle %}
        <input type="hidden" name="photo_handle" value="{{ photo_handle }}">
        <div class="note ok" role="status">
          <strong>已帶入修正後的庫存</strong>
          <span class="mono">{{ form.inventory_spec }}</span>
        </div>
        {% endif %}

        <label for="caption">中文需求（或英文）</label>
        <textarea id="caption" name="caption" required
          maxlength="{{ max_caption }}"
          placeholder="例如：我想做一台 30 顆以內的藍色小車，顏色可以替換。"
          >{{ form.caption }}</textarea>
        <p class="hint">會抽取類別、最大零件數、偏好顏色、是否允許替代與模式；
          看得懂但無法套用的條件會**具名回報**，不會靜默猜測。</p>

        <h3>方法</h3>
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

        <h3>手動庫存</h3>
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
        <label for="inventory_spec" style="margin-top:12px">庫存字串（與上面八格擇一）</label>
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
          某形狀的顏色總量不足時會**具名拒絕**，不會虛構顏色。</p>

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
          <label>
            <input type="checkbox" name="placement"
              {% if form.placement %}checked{% endif %}>
            開啟 placement gate
          </label>
          <p class="hint">{{ placement_notice }}</p>
        </fieldset>
        <p class="hint">不適用的欄位會被**具名拒絕**，不會被靜默忽略。
          頁面上的 JavaScript 只是漸進增強；<strong>伺服器才是權威</strong>，
          關掉 JavaScript 的結果是拒絕，不是靜默接受。</p>
        <noscript>
          <p class="hint">未啟用 JavaScript 時這兩組欄位不會自動停用；
            送出後仍由伺服器判定適用性並具名拒絕。</p>
        </noscript>

        <div class="actions">
          <button class="btn" type="submit">執行</button>
          <a class="btn ghost" href="/reset">全部重新開始</a>
        </div>
      </form>
    </div>
  </section>
</div>

<div class="note flat">
  <strong>這個介面不做什麼</strong>
  <ul>
    <li>不重新訓練、不調參、不重選 <span class="mono">final_H2</span>。</li>
    <li>不執行 Phase 3C，不讀取任何已封存的評估案例，不產生 Success@K。</li>
    <li>不把沒有通過靜態檢查的結構提供下載。</li>
  </ul>
</div>

<script>
/* Progressive enhancement only. The server decides which fields apply: with
   this script absent, an inapplicable field is refused by name rather than
   silently ignored. */
(function () {
  var cpsat = document.getElementById('cpsat');
  var decoder = document.getElementById('decoder');
  var seedbox = document.getElementById('seedbox');
  var radios = document.querySelectorAll('input[name="method"]');
  if (!cpsat || !decoder || !seedbox || !radios.length) { return; }
  function sync() {
    var picked = null;
    radios.forEach(function (r) { if (r.checked) { picked = r.value; } });
    cpsat.disabled = picked !== {{ mode_pipeline | tojson }};
    decoder.disabled = picked !== {{ mode_project | tojson }};
    seedbox.disabled = picked === {{ mode_rag | tojson }};
  }
  radios.forEach(function (r) { r.addEventListener('change', sync); });
  sync();
})();
</script>
{% endblock %}
"""

_PHOTO = """{% extends "base.html" %}
{% block main %}
{% if error %}
<div class="note bad" role="alert"><strong>這次沒有套用，原因如下</strong>
  {{ error }}</div>
{% endif %}

<div class="note {{ 'warn' if analysis.unidentified else 'ok' }}" role="status">
  <strong>辨識完成：找到 {{ analysis.found }} 個項目，其中
    {{ analysis.unidentified }} 個沒有被命名</strong>
  {{ recognition_limit }}
</div>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>照片與偵測框</h2>
    <p class="lede">框線畫在照片上方，照片本身沒有被重新編碼。
      虛線橘框是低信心，虛線紅框是已刪除。</p>
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
    <p class="hint">檔名 <span class="mono">{{ filename }}</span>，
      {{ analysis.width }}×{{ analysis.height }}，
      模式 {{ photo_mode_labels[analysis.mode] }}，
      辨識方法 <span class="mono">{{ analysis.method }}</span>。</p>
    <p class="hint">{{ capture_assumption }}</p>
  </div>
</section>

<form method="post" action="/photo/{{ handle }}/correct" accept-charset="utf-8">
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>人工修正</h2>
    <p class="lede">模型預測、你的修改與最後採用值分開保存。
      改了什麼都看得見，原始預測永遠不會被覆寫。</p>
    <div class="scroll">
      <table class="edit">
        <caption>共 {{ items | length }} 個項目。</caption>
        <thead><tr>
          <th>#</th><th>來源</th><th>模型預測</th><th>Top-3</th>
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

    <h3>新增一個項目</h3>
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
    <p class="hint">新增的項目沒有模型預測，會如實記為
      <span class="mono">operator</span>。</p>

    <div class="actions">
      <button class="btn" type="submit">套用修正</button>
      <a class="btn ghost" href="/">回到第一步</a>
    </div>
  </div>
</section>
</form>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>修正前／修正後庫存</h2>
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
            {% else %}<span class="pill warn">{{ '%+d' % (row.after - row.before) }}</span>
            {% endif %}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="hint">修正前共 {{ before_total }} 塊，修正後共 {{ after_total }} 塊；
      人工改動 {{ corrected.edited_items }} 個項目。
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
    <h2>用這份庫存繼續</h2>
    {% if adopted_spec %}
    <p class="lede">庫存 <span class="mono">{{ adopted_spec }}</span> 會交給既有的
      <span class="mono">parse_inventory</span>，和手打的庫存走同一條驗證路徑。</p>
    <label for="caption2">中文需求</label>
    <textarea id="caption2" name="caption" required maxlength="{{ max_caption }}"
      placeholder="例如：我想做一台小車。"></textarea>
    <h3>方法</h3>
    <div class="modes">
      {% for value in methods %}
      <label class="mode">
        <input type="radio" name="method" value="{{ value }}"
          {% if loop.first %}checked{% endif %}>
        <span><strong>{{ method_labels[value] }}</strong></span>
      </label>
      {% endfor %}
    </div>
    <div class="actions">
      <button class="btn" type="submit">執行</button>
    </div>
    {% else %}
    <p class="lede">修正後的庫存是空的，因此無法繼續。
      請至少給一個項目指定零件與數量。</p>
    {% endif %}
  </div>
</section>
</form>
{% endblock %}
"""

_RESULT = """{% extends "base.html" %}
{% block main %}
{% if ready %}
<div class="note ok" role="status">
  <strong>找到一件通過靜態交付檢查的結果</strong>
  下方的預覽、配色、組裝步驟與 LDraw 下載全部出自同一份磚清單。
  有配色時，圖與檔案也用同一份配色；沒有配色時只有結構相同，顏色不同。
  這是該件輸出的確定性檢查，<em>不是</em>成功率，也不是任何模型指標。
</div>
{% else %}
<div class="note warn" role="status">
  <strong>流程正常完成，但本次沒有可交付結果</strong>
  {{ not_ready_reason }}
  沒有預覽、沒有配色、沒有組裝步驟，也沒有下載。
</div>
{% endif %}

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>方法與 provenance</h2>
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
    {% if provenance_notices %}
    {% for notice in provenance_notices %}
    <div class="note warn" style="margin-top:12px">{{ notice }}</div>
    {% endfor %}
    {% endif %}
  </div>
</section>

{% if conditions %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>從中文需求抽取到的條件</h2>
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
  </div>
</section>
{% endif %}

{% if explanation %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>有根據的推薦說明</h2>
    <p class="lede">每一句都對應一個可查的數值；沒有語言模型參與。</p>
    {% for line in explanation.header %}<p>{{ line }}</p>{% endfor %}
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
</section>
{% endif %}

{% if attempts %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>F-pipeline 各候選</h2>
    <p class="lede">依檢索順序嘗試，取第一個通過獨立復驗的結果。
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
</section>
{% endif %}

{% if report %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>選中結果的靜態檢查</h2>
    <p class="lede">全部由 <span class="mono">{{ report.scored_by }}</span>
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

    <h3>庫存使用量與剩餘</h3>
    <div class="scroll">
      <table>
        <caption>負數如實印成負數並標記 OVERDRAWN，不會夾到 0。</caption>
        <thead><tr><th>零件</th><th class="num">備料</th><th class="num">使用</th>
          <th class="num">剩餘</th><th>狀態</th></tr></thead>
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
</section>
{% endif %}

{% if finished %}
<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>配色</h2>
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
    <p class="lede">沒有提供顏色庫存，因此不配色；LDraw 使用預設色，
      預覽與步驟圖改用「一種形狀一種顏色」的辨識用圖例。
      三者是同一個結構，但<strong>不是</strong>同一組顏色。</p>
    {% endif %}
  </div>
</section>

<section class="card">
  <div class="studs" aria-hidden="true"></div>
  <div class="body">
    <h2>CPU 3D 幾何預覽與交付</h2>
    <figure class="preview">
      <img src="/artifact/{{ handle }}/preview.png"
        width="{{ finished.preview_width }}"
        height="{{ finished.preview_height }}"
        alt="選中結果的 3D 幾何預覽：以 {{ report.result.n_bricks }} 個軸對齊
          長方體畫出每一塊積木。這是幾何檢視，不是寫實渲染。">
      <figcaption>以 Matplotlib Agg 在 CPU 上繪製。
        這是幾何檢視，不是寫實渲染，也沒有物理或穩定性分析。</figcaption>
    </figure>
    <div class="actions">
      <a class="btn" href="/artifact/{{ handle }}/model.ldr" download>
        下載 LDraw（.ldr）</a>
      {% if finished.plan %}
      <a class="btn" href="/steps/{{ handle }}/1">看組裝步驟</a>
      {% endif %}
      <a class="btn ghost" href="/">換一組需求</a>
      <a class="btn ghost" href="/reset">全部重新開始</a>
    </div>
    <p class="hint">LDraw 以實際組裝順序寫入 <span class="mono">0 STEP</span>。
      檔案只存在本機記憶體與這次連線，不寫入專案的
      <span class="mono">artifacts/</span>。</p>
    {% if finished.plan_problem %}
    <div class="note bad"><strong>沒有組裝步驟</strong>
      {{ finished.plan_problem }}</div>
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
    <p class="lede">每一步只加入最多 {{ plan.max_per_step }} 顆。
      每個非地面積木加入時，都與已加入的相鄰下層積木有 footprint 交集；
      允許先建立多個接地子結構，之後再由橫樑連接。</p>
    <div class="stepbar">
      {% if number > 1 %}
      <a class="btn ghost" href="/steps/{{ handle }}/{{ number - 1 }}">上一步</a>
      {% endif %}
      <span class="count">{{ number }} / {{ total }}</span>
      {% if number < total %}
      <a class="btn" href="/steps/{{ handle }}/{{ number + 1 }}">下一步</a>
      {% endif %}
      <a class="btn ghost" href="/result/{{ handle }}">回到結果</a>
    </div>
    <figure class="preview">
      <img src="/steps/{{ handle }}/{{ number }}/image"
        alt="第 {{ number }} 步的累積結構，共 {{ build.total_bricks }} 塊積木">
      <figcaption>{{ description }}</figcaption>
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
      <h3>這一步之後的狀態</h3>
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
      <p class="hint">目前 {{ build.components }} 個子結構。
        中間步驟允許有多個；只有最終結構要求單一元件。
        連通性不是物理支撐，也不是穩定性。</p>
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
      <p class="hint">累積 {{ build.total_bricks }} 塊，
        整件作品共 {{ plan.n_bricks }} 塊。</p>
      {% if build.stock_remaining %}
      <h3>庫存剩餘</h3>
      <p class="mono">{{ build.stock_remaining | dictsort
        | map('join', ':') | join(', ') }}</p>
      {% endif %}
    </div>
  </section>
</div>

<div class="note flat">
  <strong>全部步驟</strong>
  <ol>
    {% for line in descriptions %}
    <li{% if loop.index == number %} style="font-weight:700"{% endif %}
      >{{ line }}</li>
    {% endfor %}
  </ol>
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
        title="庫存與需求", step=1, form=form or blank_form(), error=error,
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
        title="照片辨識與修正", step=2, csrf_token=csrf_token, handle=handle,
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
        title="結果與交付", step=3, result=result, handle=handle,
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
        title="組裝步驟", step=4, handle=handle, plan=plan.as_dict(),
        number=number, total=plan.n_steps, step_checks=rows,
        build=step.as_dict(), descriptions=list(descriptions),
        description=descriptions[number - 1], **_COMMON)


def render_error(*, heading: str, detail: str, advice: str,
                 title: str = "無法完成", step: int = 1) -> str:
    return _ENV.get_template("error.html").render(
        title=title, step=step, heading=heading, detail=detail, advice=advice,
        **_COMMON)
