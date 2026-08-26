"""Explanations built out of the numbers, never around them.

Every sentence this module produces names a field of the evidence that a reader
can go and check.  There is no language model in the path and no template that
can assert something the calculation did not:

* the completion figure comes from ``inventory_completion``;
* the two orderings are two fields.  ``semantic_rank`` is the embedding
  search's position and ``rerank_rank`` is the position after the inventory
  and static checks, and neither is ever printed under the other's name;
* the shortfall list comes from ``missing_parts``, part by part with counts;
* "can be built" is printed only when ``fully_buildable`` is true, which means
  both the stock covered every part *and* the structure touches the ground and
  is one component under adjacent-layer connectivity;
* the source is the anonymous ``catalog_id``.  The dataset's own ``object_id``
  is never written into an explanation.

The last point is the one that makes the module worth having as its own file:
the phrase "可以組" is generated in exactly one place, guarded by exactly one
condition.  Spread across a template it would eventually be printed next to a
work that was four bricks short.

Connectivity here is adjacent-layer footprint overlap.  It is not support and
not stability, and the explanation says so rather than leaving a reader to
assume otherwise.
"""

from __future__ import annotations

from src.colour.palette import BY_ID
from src.retrieval.search import Candidate, SearchResult

CONNECTIVITY_NOTE = (
    "「連通」指相鄰層 footprint 有交集，「接地」指有磚位於 z = 0。"
    "兩者都是靜態幾何，不是物理支撐，也不是穩定性分析。")

RETRIEVAL_NOTE = (
    "候選順序先由多語 embedding 的語意相似度決定（`semantic_rank`），"
    "再由精確庫存計算與靜態結構條件重排（`rerank_rank`）。"
    "上面每一段列出的是重排後的順序，但每一件標示的語意排名一律取自"
    "語意檢索本身，不是它在這份清單裡的位置。"
    "語意分數不是成效指標，也沒有經過任何檢索評估。")


def _parts(counts: dict[str, int]) -> str:
    return "、".join(f"{part}×{count}" for part, count in sorted(counts.items()))


def explain_candidate(candidate: Candidate, *,
                      rerank_rank: int | None = None) -> dict:
    """The grounded explanation for one candidate, as fields plus a sentence.

    Both are returned.  The fields are what a machine reader should use; the
    sentence is assembled from those same fields so a person and a program
    cannot come away with different conclusions.

    ``rerank_rank`` is the candidate's position *after* the inventory and
    static-structure sort, and it is written under its own name.  It is not
    allowed to stand in for the semantic rank: the semantic rank always comes
    from the hit, because a work that the embedding search put eighth and the
    re-rank lifted to first is eighth on meaning and first on usability, and a
    sentence that prints "semantic rank 1" for it is false.
    """
    evidence = {
        "catalog_id": candidate.item.catalog_id,
        "semantic_rank": candidate.hit.semantic_rank,
        "rerank_rank": rerank_rank,
        "semantic_score": round(candidate.hit.score, 6),
        "n_bricks": candidate.item.n_bricks,
        "required_inventory": dict(candidate.item.required),
        "required_total": candidate.required_total,
        "missing_parts": dict(candidate.missing),
        "missing_total": candidate.missing_total,
        "inventory_completion": round(candidate.completion, 6),
        "inventory_sufficient": not candidate.missing,
        "touches_ground": candidate.touches_ground,
        "stud_only_connected": candidate.connected,
        "static_structure_ready": candidate.static_ready,
        "fully_buildable": candidate.buildable,
    }

    ranking = (f"語意排名第 {evidence['semantic_rank']}，相似度 "
               f"{evidence['semantic_score']:.4f}。")
    if rerank_rank is not None:
        ranking += (f"庫存與靜態條件重排後排第 {rerank_rank}"
                    "（重排位置不是語意排名，兩者互不推導）。")
    lines = [
        f"來源：catalog_id {candidate.item.catalog_id}"
        f"（train split 的匿名識別碼，不是資料集識別碼）。",
        ranking,
        f"這件作品需要 {candidate.item.n_bricks} 塊："
        f"{_parts(candidate.item.required)}。",
    ]
    if candidate.missing:
        lines.append(
            f"庫存不足：缺 {candidate.missing_total} 塊"
            f"（{_parts(candidate.missing)}）；"
            f"完成比例 {candidate.completion * 100:.0f}%。")
    else:
        lines.append(
            f"庫存足夠：所需 {candidate.required_total} 塊全部有貨，"
            "完成比例 100%。")
    lines.append(
        "靜態條件："
        f"接地 {'通過' if candidate.touches_ground else '未通過'}、"
        f"連通 {'通過' if candidate.connected else '未通過'}。")

    if candidate.buildable:
        lines.append("結論：**可以組**——庫存足夠，且接地與連通都通過。")
    elif candidate.missing and not candidate.static_ready:
        lines.append("結論：**不能組**——庫存不足，且靜態條件未通過。")
    elif candidate.missing:
        lines.append("結論：**不能組**——庫存不足。這是最相似的候選之一，"
                     "但相似不等於可組。")
    else:
        lines.append("結論：**不能組**——庫存雖然足夠，但靜態結構條件未通過，"
                     "因此不會被說成可以組。")
    return {"evidence": evidence, "sentences": lines,
            "verdict": "buildable" if candidate.buildable else "not_buildable"}


def explain_result(result: SearchResult, inventory: dict[str, int]) -> dict:
    """The whole search as grounded evidence, with the caveats attached."""
    conditions = result.conditions
    header = [
        f"需求原文：{conditions.text}",
        "抽取到的條件：" + "；".join(conditions.describe_zh()),
    ]
    if conditions.has_unresolved:
        header.append(
            "以下條件被看到但沒有套用，已具名回報，不做靜默猜測："
            + "；".join(f"{item.field}「{item.text}」：{item.reason}"
                        for item in conditions.unresolved))
    header.append(
        f"手動庫存：{_parts(inventory)}（共 {sum(inventory.values())} 塊）。")
    header.append(f"索引：train split {result.index_documents} 件作品；"
                  f"同物件排除 {result.excluded_same_object} 件。")

    candidates = [explain_candidate(candidate, rerank_rank=position)
                  for position, candidate in enumerate(result.ranked, 1)]
    if result.selected is not None:
        chosen = ("選中 catalog_id "
                  f"{result.selected.item.catalog_id}：它是重排後第一件"
                  "同時庫存足夠、接地且連通的作品。")
    elif not result.retrieved:
        chosen = ("沒有任何語意候選通過條件過濾，因此沒有推薦。"
                  "不會退而拿一件不符合條件的作品當結果。")
    else:
        chosen = ("取回的候選都不符合庫存或靜態結構條件，因此沒有推薦。"
                  "最相似的那一件仍列在上面，但它不會被說成可以組。")

    colour_note = None
    if conditions.preferred_colours:
        names = "、".join(BY_ID[c].label_zh
                          for c in conditions.preferred_colours)
        colour_note = (
            f"偏好顏色（{names}）不影響檢索或可組判定：結構軌是無色的，"
            "顏色在結構決定之後才由配色器在顏色庫存內指派。")

    return {
        "kind": "brickagain.grounded_recommendation",
        "status": result.status,
        "header": header,
        "selection": chosen,
        "candidates": candidates,
        "colour_note": colour_note,
        "notes": [RETRIEVAL_NOTE, CONNECTIVITY_NOTE],
        "generated_from": (
            "structured evidence only: every sentence above corresponds to a "
            "field of the candidate evidence. No language model produced any "
            "of this text"),
    }


def format_explanation(explanation: dict) -> str:
    """The grounded explanation as plain text, for the command line."""
    out = ["=" * 72, "有根據的推薦說明", "=" * 72]
    out += explanation["header"]
    out.append("")
    for index, candidate in enumerate(explanation["candidates"], 1):
        out.append(f"--- 候選 {index} "
                   f"({candidate['evidence']['catalog_id']}) ---")
        out += ["  " + line for line in candidate["sentences"]]
    out += ["", explanation["selection"]]
    if explanation.get("colour_note"):
        out.append(explanation["colour_note"])
    out += [""] + explanation["notes"]
    return "\n".join(out)
