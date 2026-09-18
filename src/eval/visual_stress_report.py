"""The V1 report: rendering only, and a refusal if it would overclaim.

Reads ``summary.json`` and ``per_image_results.json`` and formats them; it
derives no figure of its own.

It *is* inside the pinned source set, unlike its Phase 3C counterpart. The
manifest pins the import closure of ``scripts/60_visual_stress.py``, and
``--report`` reaches this module, so editing it moves
``source_manifest_digest`` and a finished run's manifest stops holding. That
is deliberate: the previous design pinned a hand-written list of nine files
and left the rest of the closure -- ``src/vision/metrics.py`` among them --
free to change without moving anything, which is the failure that made
generation gen02 unusable. Paying for a reworded heading with a new
generation is the cheaper of the two mistakes.

The one thing it does enforce is language. The rendered text is scanned for
:data:`~src.eval.visual_stress.FORBIDDEN_REPORT_TERMS` before it is written,
so a report that called a software render a photograph, or a rendered rate
real-photograph accuracy, is refused rather than published.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import visual_stress as vs
from src.eval.acceptance import PlanRefused


def _pct(entry) -> str:
    if not isinstance(entry, dict) or entry.get("value") is None:
        return "--"
    return f"{entry['value'] * 100:.1f}%"


def _frac(entry) -> str:
    if not isinstance(entry, dict):
        return "--"
    return f"{entry.get('numerator')}/{entry.get('denominator')}"


def read_summary(out_dir) -> dict:
    path = Path(out_dir) / vs.SUMMARY_NAME
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; a report renders a summary "
                          "and does not derive one")
    return json.loads(path.read_text())


def read_results(out_dir) -> dict:
    path = Path(out_dir) / vs.RESULTS_NAME
    if not path.is_file():
        raise PlanRefused(f"{path} is not here")
    return json.loads(path.read_text())


def render(summary: dict, results: dict, manifest: dict) -> str:
    lines: list[str] = []
    w = lines.append
    overall = summary["overall"]

    w("# 合成視覺壓力測試 — V1 確定性渲染基準")
    w("")
    w("這份報告只涵蓋 **V1**。每一張影像都是由 `src.vision.synthetic` "
      "以算術繪製的**軟體渲染圖**，不是照片，也不得被當成照片呈現。")
    w("")

    w("## 讀任何數字之前")
    w("")
    w("1. **這些不是照片。** V1 是俯視的方塊繪圖：長寬比、stud 間距與顏色是對的，"
      "因為那是辨識器實際測量的東西；光學、真實材質、真實光線與真實桌面都不存在。")
    w("2. **ground truth 是「建構」出來的，不是標註出來的。** 場景說「橘色 2x4 在第 1 列第 1 行」，"
      "渲染器就畫那個。沒有標註誤差要扣除，這正是 V1 是量化層而 V3 不是的原因。")
    w("3. **這裡的比率描述的是「對渲染影像條件的韌性」**，"
      "不是真實照片準確率，也不是實際使用情境下的表現。")
    w("4. **V2 與 V3 沒有執行。** 原因在下方「未執行的兩層」一節具名說明，"
      "本報告沒有任何數字涵蓋它們。")
    w("5. **沒有任何實體組裝或物理驗證。** 這條管線讀影像、提出庫存，如此而已。")
    w("")

    w("## 產物與來源")
    w("")
    w("| 欄位 | 值 |")
    w("|---|---|")
    w(f"| `manifest_digest` | `{summary['manifest_digest']}` |")
    w(f"| `summary_digest` | `{summary['summary_digest']}` |")
    w(f"| `source_manifest_digest` | `{summary['source_manifest_digest']}` |")
    w(f"| 辨識方法 | `{summary['method']}` |")
    w(f"| 渲染器 | `{manifest['provenance']['renderer']}` |")
    w(f"| render seed | `{manifest['provenance']['render_seed']}` |")
    w(f"| AI 生成模型 | `{manifest['provenance']['ai_model']}`（無，V1 不使用 AI 生成）|")
    w(f"| manifest 影像數 | {summary['n_images_in_manifest']} |")
    w(f"| accepted | {summary['n_accepted']} |")
    w(f"| rejected | {summary['n_rejected']} |")
    w(f"| 實際計分 | {summary['n_scored']} |")
    w("")

    w("## 整體結果")
    w("")
    w("| 指標 | 值 | 分數 |")
    w("|---|---|---|")
    w(f"| 完整 inventory exact match | {_pct(overall['inventory_exact_match'])} "
      f"| {_frac(overall['inventory_exact_match'])} |")
    w(f"| UI 能載入並產出可編輯項目 | {_pct(overall['ui_loaded'])} "
      f"| {_frac(overall['ui_loaded'])} |")
    w(f"| oracle 修正路徑成立 | "
      f"{_pct(overall['oracle_correction_path_valid'])} "
      f"| {_frac(overall['oracle_correction_path_valid'])} |")
    w(f"| 有數量錯誤的影像 | {_pct(overall['images_with_a_count_error'])} "
      f"| {_frac(overall['images_with_a_count_error'])} |")
    w(f"| 真實積木總數 | {overall['true_bricks']} | |")
    w(f"| 偵測到的物件總數 | {overall['detected']} | |")
    w(f"| part 類型錯誤 | {overall['part_errors']} | |")
    w(f"| 顏色錯誤 | {overall['colour_errors']} | |")
    w(f"| false positive | {overall['false_positives']} | |")
    w(f"| false negative | {overall['false_negatives']} | |")
    w(f"| 分類器 top-1 正確（在有配對到的積木上）"
      f"| {_pct(overall['top1_correct_among_matched'])} "
      f"| {_frac(overall['top1_correct_among_matched'])} |")
    w(f"| 棄權（回報 unknown）次數 | {overall['abstentions']} | |")
    w(f"| 平均信心 | {overall['mean_confidence']} | |")
    w("")
    w("`false positive` 是偵測到而沒有對應真實積木的框，`false negative` 是"
      "沒有任何框對上的真實積木；兩者都以 IoU ≥ "
      f"{vs.MATCH_IOU} 配對。")
    w("")
    w("**棄權與答錯是兩種不同的失敗，這裡分開計。** 棄權（`unknown`）會把該裁切"
      "送到人面前，修正流程隨後解決它；有信心的錯誤標籤則會把一顆桌上沒有的積木"
      "放進庫存。合成單一錯誤率會把兩者混在一起，看不出這條路徑實際是哪一種。")
    w("")
    w("**`top-1 正確` 與 `inventory exact match` 的落差是算術，不是矛盾。** "
      "一張影像要全部 3–10 顆積木同時正確才算 exact match，所以逐顆約 "
      f"{_pct(overall['top1_correct_among_matched'])} 的正確率在整張影像的層級上"
      "幾乎必然歸零。這正是這條管線把修正介面當成必要環節而非備案的理由。")
    w("")

    w("## 各影像條件")
    w("")
    w("矩陣是「基準值 + 每次只動一軸」，不是交叉組合，所以差異只有一個成因。")
    w("")
    for axis, values in summary["by_axis"].items():
        w(f"### {axis}")
        w("")
        w("| 值 | 影像 | inventory exact | part 錯誤 | 顏色錯誤 | FP | FN | oracle 路徑 |")
        w("|---|---|---|---|---|---|---|---|")
        for value, block in values.items():
            mark = "（基準）" if value == manifest["baseline"][axis] else ""
            w(f"| `{value}`{mark} | {block['images']} "
              f"| {_pct(block['inventory_exact_match'])} "
              f"| {block['part_errors']} | {block['colour_errors']} "
              f"| {block['false_positives']} | {block['false_negatives']} "
              f"| {_pct(block['oracle_correction_path_valid'])} |")
        w("")

    w("## 各場景")
    w("")
    w("| 場景 | 影像 | 真實積木 | inventory exact | part 錯誤 | 顏色錯誤 |")
    w("|---|---|---|---|---|---|")
    for scene_id, block in summary["by_scene"].items():
        w(f"| `{scene_id}` | {block['images']} | {block['true_bricks']} "
          f"| {_pct(block['inventory_exact_match'])} | {block['part_errors']} "
          f"| {block['colour_errors']} |")
    w("")

    if summary["part_confusions"]:
        w("## 主要 part 混淆")
        w("")
        w("| 真實 → 預測 | 次數 |")
        w("|---|---|")
        for key, count in list(summary["part_confusions"].items())[:12]:
            w(f"| `{key}` | {count} |")
        w("")
    if summary["colour_confusions"]:
        w("## 主要顏色混淆")
        w("")
        w("| 真實 → 預測 | 次數 |")
        w("|---|---|")
        for key, count in list(summary["colour_confusions"].items())[:12]:
            w(f"| `{key}` | {count} |")
        w("")

    w("## 主要失敗模式")
    w("")
    w("按上面的逐軸表，依 false negative 由多到少：")
    w("")
    ranked = []
    for axis, values in summary["by_axis"].items():
        base_value = manifest["baseline"][axis]
        base = values.get(base_value) or {}
        for value, block in values.items():
            if value == base_value:
                continue
            ranked.append((
                block["false_negatives"] - base.get("false_negatives", 0),
                axis, value, block, base))
    ranked.sort(key=lambda r: -r[0])
    w("| 條件 | FN（相對基準） | 偵測數 | part 錯誤 | 顏色錯誤 |")
    w("|---|---|---|---|---|")
    for delta, axis, value, block, base in ranked[:6]:
        w(f"| `{axis}={value}` | {block['false_negatives']} "
          f"（{delta:+d}）| {block['detected']} | {block['part_errors']} "
          f"| {block['colour_errors']} |")
    w("")
    w("**基準條件本身就有 false negative。** 那不是任何一個條件造成的：小面積的積木"
      "（尤其 1x1）在整張大畫面裡會被偵測器的面積與形狀過濾掉，這是 "
      "`src.vision.detect.propose` 既有的行為，在這裡第一次被量到。它對使用者的"
      "意義很直接——一次拍太多、單顆佔畫面比例太小時，小積木會直接消失，"
      "而不是被標成低信心。")
    w("")

    w("## 管線是否安全拒絕")
    w("")
    w("四個不是影像的輸入，看管線會不會具名拒絕，而不是從垃圾裡生出一份庫存。")
    w("")
    w("| 探針 | 具名拒絕 | 由誰 |")
    w("|---|---|---|")
    for name, entry in summary["safe_rejection_shared_input_layer"].items():
        if not isinstance(entry, dict):
            continue
        w(f"| `{name}` | {'是' if entry['refused'] else '**否**'} "
          f"| `{entry.get('by')}` |")
    w("")
    w(f"全部具名拒絕：**"
      f"{'是' if summary['safe_rejection_shared_input_layer'].get('all_refused_by_name') else '否'}**")
    w("")

    w("## 被拒絕的影像")
    w("")
    if summary["n_rejected"] == 0:
        w("沒有影像被拒絕。V1 的拒絕條件是機械性的——影像digest 對不上、"
          "場景digest 對不上、或影像無法讀取——因為 ground truth 是建構的而非標註的。")
        w("")
        w("**V2 的拒絕條件會不同**：AI 變化圖若改動了積木幾何、數量、顏色或零件，"
          "ground truth 就不再可靠，該圖必須列入 rejected 且不計入正式準確率。"
          "那條規則已寫在 manifest 的 `rejection_criteria` 裡，但因為 V2 沒有執行，"
          "它沒有被套用過。")
    else:
        w("| 影像 | 原因 |")
        w("|---|---|")
        for entry in summary["rejected"]:
            w(f"| `{entry['image_id']}` | {entry['reason']} |")
    w("")

    w("## 未執行的兩層")
    w("")
    tiers = summary["tiers"]
    w(f"* **V2（經驗證的 AI 變化圖）**：{tiers['v2_validated_ai_variations']}")
    w(f"* **V3（純 AI 困難案例）**：{tiers['v3_pure_ai_hard_cases']}")
    w("")
    w(f"**原因**：{tiers['reason']}")
    w("")
    w(f"**要解除阻塞需要**：{tiers['what_would_unblock_it']}")
    w("")
    w(f"**因此不宣稱**：{tiers['not_claimed']}")
    w("")

    w("## 這次執行不能被讀成什麼")
    w("")
    w("- 不是真實照片準確率。這裡沒有任何照片。")
    w("- 不是實際使用情境已驗證。")
    w("- 不是實體組裝已驗證——沒有任何積木被組起來。")
    w("- 不是物理穩定性已驗證——這條管線不做任何力學分析。")
    w("- V1 的渲染器不是訓練資料的渲染器，所以 learned 分類器在這裡遇到的是"
      "**domain shift**；它的錯誤率反映的是這一點，而不是它在自身測試集上的表現。")
    # One trailing newline, not two. Every ``w("")`` above is a blank line
    # *between* blocks; keeping the last one made every published report end
    # in a blank line, which ``git diff --check`` reports. gen01 and gen02
    # keep the bytes they were written with -- they are frozen and are not
    # edited to make a linter happy -- and gen03 onwards is clean.
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(out_dir, *, out=None) -> dict:
    out_dir = Path(out_dir)
    summary = read_summary(out_dir)
    results = read_results(out_dir)
    manifest = json.loads((out_dir / vs.MANIFEST_NAME).read_text())

    if summary["manifest_digest"] != manifest["manifest_digest"]:
        raise PlanRefused("the summary is for a different manifest")
    if results.get("manifest_digest") != manifest["manifest_digest"]:
        raise PlanRefused("the per-image results are for a different "
                          "manifest")
    problems = vs.manifest_problems(manifest)
    if problems:
        raise PlanRefused("this manifest does not hold:\n  - "
                          + "\n  - ".join(problems))
    # Whole-record, not a chosen subset: the fields a subset left out --
    # the rejected list, the safe-rejection probe, the tier declaration --
    # are the ones a rewrite would target.
    recomputed = vs.summarise(results, manifest)
    if vs.summary_identity(recomputed) != vs.summary_identity(summary):
        raise PlanRefused(
            "the stored summary is not what the per-image results imply; "
            "refusing to report a number nobody can re-derive")
    if summary.get("summary_digest") != recomputed.get("summary_digest"):
        raise PlanRefused(
            "the stored summary's own digest does not cover it")

    body = render(summary, results, manifest)
    hits = vs.forbidden_terms_in(body)
    if hits:
        raise PlanRefused(
            f"the rendered report uses {hits}, which a software-render tier "
            "may not claim")

    successes, failures = vs.case_indices(results)
    report_path = Path(out) if out else out_dir / vs.REPORT_NAME
    # write-once, like every other artefact here. These three used to be
    # written with ``write_text``, so a second run replaced a published
    # report and two published indices without anything saying so.
    written = {"report": vs.write_once_text_checked(
        report_path, body, what="report")}

    for name, payload in (
            (vs.SUCCESS_INDEX_NAME, {
                "kind": "brickagain.visual_stress_success_cases",
                "tier": "V1",
                "manifest_digest": manifest["manifest_digest"],
                "summary_digest": summary["summary_digest"],
                "criterion": "the adopted inventory equals the scene exactly",
                "n": len(successes), "cases": successes}),
            (vs.FAILURE_INDEX_NAME, {
                "kind": "brickagain.visual_stress_failure_cases",
                "tier": "V1",
                "manifest_digest": manifest["manifest_digest"],
                "summary_digest": summary["summary_digest"],
                "criterion": "the adopted inventory differs from the scene",
                "n": len(failures), "cases": failures}),
    ):
        written[name] = vs.write_once_json_checked(
            report_path.parent / name, payload, what="case index")
    return written


def verify(out_dir, *, root=None, replay: bool = True) -> dict:
    """Re-derive every artefact in a run directory and compare, all of it.

    Seven layers, and they fail separately:

    1. the manifest holds against the tree, including the runtime it pins;
    2. the accepted/rejected partition is recomputed from the stored images;
    3. the result set is internally sound -- own digest, one row per
       accepted image, no row for anything else -- and every row's own
       arithmetic holds without a model: counts equal the lists they count,
       the exact-match flag agrees with the two inventories reported, and
       the true-brick count is the scene's;
    4. **the pinned model, re-run, returns the stored rows** -- the one
       check that is not internal to the directory (see
       :func:`~src.eval.visual_stress.replay_problems`);
    5. the summary is recomputed from the rows and names that exact result
       set by digest;
    6. the report is re-rendered and both indices rebuilt;
    7. the safe-rejection probe is re-run.

    Layer 4 is why the rest can be trusted. Without it, a result set written
    by hand and made self-consistent passes every other layer, because every
    other layer derives from the rows.
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / vs.MANIFEST_NAME).read_text())
    results = read_results(out_dir)
    summary = read_summary(out_dir)
    problems = list(vs.manifest_problems(manifest, root=root))

    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != vs.digest_of(body):
        problems.append("manifest_digest does not cover the manifest")
    problems.extend(f"runtime: {p}" for p in
                    vs.runtime_problems(manifest, root=root))

    _accepted, rejected = vs.partition(out_dir, manifest)
    accepted_now = {a["image_id"] for a in _accepted}
    stored_accepted = {a["image_id"] for a in results.get("accepted") or []}
    if accepted_now != stored_accepted:
        problems.append(
            f"the images on disk now accept {len(accepted_now)} and the "
            f"result set accepted {len(stored_accepted)}")
    if len(rejected) != summary.get("n_rejected"):
        problems.append(
            f"{len(rejected)} images are rejected now and the summary says "
            f"{summary.get('n_rejected')}")
    if [r["image_id"] for r in rejected] != \
            [r["image_id"] for r in results.get("rejected") or []]:
        problems.append("the rejected list is not what these images imply")

    problems.extend(f"results: {p}" for p in
                    vs.results_problems(results, manifest))
    problems.extend(f"row: {p}" for p in
                    vs.row_metric_problems(results, manifest))

    if replay and not problems:
        problems.extend(f"replay: {p}" for p in
                        vs.replay_problems(out_dir, manifest, results,
                                           root=root))

    recomputed = vs.summarise(results, manifest)
    if vs.summary_identity(recomputed) != vs.summary_identity(summary):
        differing = sorted(
            k for k in set(vs.summary_identity(recomputed))
            | set(vs.summary_identity(summary))
            if vs.summary_identity(recomputed).get(k)
            != vs.summary_identity(summary).get(k))
        problems.append(
            f"the summary is not what the results imply; {len(differing)} "
            f"fields differ, first {differing[:4]}")
    if recomputed.get("summary_digest") != summary.get("summary_digest"):
        problems.append("summary_digest does not cover the summary")
    if summary.get("results_digest") != results.get("results_digest"):
        problems.append(
            "the summary names a different per-image result set than the "
            "one in this directory")

    probe = vs.safe_rejection_probe()
    stored_probe = results.get("safe_rejection_shared_input_layer") or {}
    if probe.get("all_refused_by_name") != stored_probe.get(
            "all_refused_by_name"):
        problems.append(
            "the shared input layer refuses differently now than the stored "
            "probe records")
    for name, entry in stored_probe.items():
        if not isinstance(entry, dict):
            continue
        if probe.get(name, {}).get("refused") != entry.get("refused"):
            problems.append(f"the {name} probe is refused differently now")

    stored_report = out_dir / vs.REPORT_NAME
    if not stored_report.is_file():
        problems.append(f"{vs.REPORT_NAME} is not here")
    elif stored_report.read_text(encoding="utf-8") != render(
            summary, results, manifest):
        problems.append("the stored report is not what this summary renders")

    successes, failures = vs.case_indices(results)
    for name, rebuilt in ((vs.SUCCESS_INDEX_NAME, successes),
                          (vs.FAILURE_INDEX_NAME, failures)):
        path = out_dir / name
        if not path.is_file():
            problems.append(f"{name} is not here")
            continue
        stored = json.loads(path.read_text())
        if stored.get("n") != len(rebuilt) or stored.get("cases") != rebuilt:
            problems.append(f"{name} is not what the results imply")
        if stored.get("manifest_digest") != manifest["manifest_digest"]:
            problems.append(f"{name} is for a different manifest")
        if stored.get("summary_digest") != summary.get("summary_digest"):
            problems.append(f"{name} is for a different summary")
    return {"verified": not problems, "problems": problems,
            "replayed": bool(replay)}


__all__ = ["read_summary", "read_results", "render", "write_report",
           "verify"]
