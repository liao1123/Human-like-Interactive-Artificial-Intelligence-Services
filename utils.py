import json
from typing import Dict, List, Any, Optional, Set, Tuple


def default_experiment_dir(test_model: str, dialogue_turn: int) -> str:
    safe_model_name = "".join(
        ch if ch.isalnum() or ch in {".", "-", "_"} else "_"
        for ch in test_model
    ).strip("_")
    if not safe_model_name:
        safe_model_name = "unknown_model"
    return f"outputs/{safe_model_name}/dialogue_turn={dialogue_turn}"

def load_jsonl(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存为JSONL格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 把一整行的jsonl转化为json方便查看
def jsonl_2_json(jsonl_path):
     data = load_jsonl(jsonl_path)
     with open(jsonl_path.replace(".jsonl", "_2_json.json"), "w", encoding="utf-8") as f:
         json.dump(data, f, ensure_ascii=False, indent=2)


def _sort_key(x: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = x.get("meta") or {}
    c = str(meta.get("category_id", "") or "")
    d = str(meta.get("dimension_id", "") or "")
    card_id = str(meta.get("card_id", "") or "")
    return (c, d, card_id)

def sort_jsonl_by_metadata(out_path: str):
    items = load_jsonl(out_path)
    items.sort(key=_sort_key)

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[sorted] wrote {len(items)} lines to {out_path}")
    jsonl_2_json(out_path)

    return items


def summary_result(items, summary_save_path):
    summary_list = []
    index = {}
    for item in items:
        meta = item.get("meta") or {}
        item_test_model = meta.get("test_model")
        item_judge_model = item.get("judge_model")
        item_generate_model = meta.get("generate_model")
        cid = str(meta.get("category_id", "") or "")
        did = str(meta.get("dimension_id", "") or "")
        passed = bool((item.get("judge_result") or {}).get("pass", False))
        score = float((item.get("judge_result") or {}).get("score", 0.0))
        
        key = (item_test_model, item_judge_model, item_generate_model)
        if key not in index:
            d = {
                "test_model": item_test_model,
                "judge_model": item_judge_model,
                "generate_model": item_generate_model,
                "by_dimension": {},
            }
            index[key] = d
            summary_list.append(d)

        bucket = index[key]["by_dimension"]
        bucket.setdefault(cid, {}) # bucket字典中有cid就返回原本的cid，没有就创建一个cid并赋值为{}，然后返回这个cid对应的值（即{}）
        bucket[cid].setdefault(did, {"count": 0, "pass_count": 0, "score": 0.0, "score=0-1": 0.0}) # bucket[cid]字典中有did就返回原本的did，没有就创建一个did并赋值为{"count": 0, "pass_count": 0, "score": 0.0}，然后返回这个did对应的值（即{"count": 0, "pass_count": 0, "score": 0.0}）

        bucket[cid][did]["count"] += 1
        if passed:
            bucket[cid][did]["pass_count"] += 1
        bucket[cid][did]["score"] += score
        if score <= 1.0:
            bucket[cid][did]["score=0-1"] += 1
    
    # 计算 pass_rate
    for d in summary_list:
        for cid, dim_map in d["by_dimension"].items():
            for did, stat in dim_map.items():
                c = stat["count"]
                p = stat["pass_count"]
                stat["pass_rate"] = round(p / c, 4) if c else 0.0
    
    # 计算 avg score, 这里的平均分是按照总分为5分来计算的，即score/5，方便和通过率进行对比分析
    for d in summary_list:
        for cid, dim_map in d["by_dimension"].items():
            for did, stat in dim_map.items():
                c = stat["count"]
                s = stat["score"]
                stat["avg_score"] = round(s / (5*c), 4) if c else 0.0
    # 计算 score=0-1的比例，计算打分0-1占总数的比例
    for d in summary_list:
        for cid, dim_map in d["by_dimension"].items():
            for did, stat in dim_map.items():
                c = stat["count"]
                s_0_1 = stat["score=0-1"]
                stat["score=0-1_rate"] = round(s_0_1 / c, 4) if c else 0.0
    
    with open(summary_save_path, "w", encoding="utf-8") as f:
        for d in summary_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print(f"[summary] wrote summary to {summary_save_path}")
    jsonl_2_json(summary_save_path)


def merge_score_and_trace(score_save_path, dialogue_trace_path, merged_save_path):
    """
    按 meta.card_id 合并 score.jsonl 与 dialogue_trace.jsonl

    - score.jsonl 每行示例：
      {
        "meta": {..., "card_id": "..."},
        "judge_model": "...",
        "judge_result": {...}
      }

    - dialogue_trace.jsonl 每行示例：
      {
        "meta": {..., "card_id": "..."},
        "plan": "...",
        "dialogue_trace": [...]
      }

    合并后输出（以 score 为主）：
      - 保留 score 的所有字段
      - 追加 trace 侧的 "plan"、"dialogue_trace"
      - 如果 score.meta 缺字段，会用 trace.meta 补齐
    """
    score_data = load_jsonl(score_save_path)
    trace_data = load_jsonl(dialogue_trace_path)

    # 提取dialogue提取 card id
    trace_by_card_id: Dict[str, Dict[str, Any]] = {}
    for item in trace_data:
        meta = item.get("meta") or {}
        card_id = str(meta.get("card_id", "") or "")
        if not card_id:
            continue
        trace_by_card_id[card_id] = item

    merged_data: List[Dict[str, Any]] = []
    for score in score_data:
        s_meta = score.get("meta") or {}
        card_id = str(s_meta.get("card_id", "") or "")
        dialogue = trace_by_card_id.get(card_id)

        # 设计dict结构
        merged: Dict[str, Any] = {}
        merged["meta"] = score["meta"]
        merged["meta"]["judge_model"] = score["judge_model"]
        merged["dialogue_trace"] = dialogue.get("dialogue_trace")
        merged["judge_result"] = score["judge_result"]

        # 删除 judge_result 内部不需要的字段（按 key 删除）
        merged["judge_result"].pop("category_id", None)
        merged["judge_result"].pop("dimension_id", None)

        merged_data.append(merged)

    # 写入输出
    with open(merged_save_path, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[merged] wrote {len(merged_data)} lines to {merged_save_path}"
    )
    jsonl_2_json(merged_save_path)

# 原本计算最终得分是按照=5得分的通过率，现在按照score / 500来进行计算
def calculate_score(origin_datas):
    item_score_dict ={
        "C1-D1": 0,
        "C1-D2": 0,
        "C1-D3": 0,
        "C1-D4": 0,
        "C2-D1": 0,
        "C2-D2": 0,
        "C2-D3": 0,
        "C3-D1": 0,
        "C3-D2": 0,
        "C3-D3": 0,
        "C3-D4": 0,
    }
    for data in origin_datas:
        category_id = data.get("meta", {}).get("category_id")
        dimension_id = data.get("meta", {}).get("dimension_id")
        item_score_dict["{category_id}-{dimension_id}".format(category_id=category_id, dimension_id=dimension_id)] += data.get("judge_result", {}).get("score", 0)
    
    return [ score / 500 for score in item_score_dict.values() ]

if __name__ == "__main__":
    score_save_path = "result_2026_2_5/deepseek-v3.2/dialogue_turn=5/score.jsonl"
    dialogue_trace_path = "result_2026_2_5/deepseek-v3.2/dialogue_turn=5/dialogue_trace.jsonl"
    merged_save_path = "result_2026_2_5/deepseek-v3.2/dialogue_turn=5/merge.jsonl"
    merge_score_and_trace(score_save_path, dialogue_trace_path, merged_save_path)
