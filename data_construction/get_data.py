import json

# Path to hidden-state clustering results
HIDDEN_STATE_PATH = "data/hidden_state_results.json"

# Path to the corresponding QA dataset (documents will be aligned by index)
DATASET_PATH = "data/train_top5.json"

data = [json.loads(line) for line in open(HIDDEN_STATE_PATH, "r", encoding="utf-8")]
data1 = [json.loads(line) for line in open(DATASET_PATH, "r", encoding="utf-8")][
    : len(data)
]

print(f"Total items: {len(data)}")
print(f"Total items in dataset: {len(data1)}")


def is_substring_match(candidate: str, answer_set: set) -> bool:
    candidate = candidate.lower().strip()
    for ans in answer_set:
        ans = ans.lower().strip()
        if candidate in ans or ans in candidate:
            return True
    return False


for i, item in enumerate(data):
    answer_set = {}

    for cluster in item["cluster_info"]:
        pred_answer = cluster["cluster_answer"]

        if pred_answer not in answer_set:
            answer_set[pred_answer] = {
                "order": cluster["all_permutations_order"],
                "final_status": is_substring_match(pred_answer, item["answer"]),
            }
        else:
            answer_set[pred_answer]["order"].extend(cluster["all_permutations_order"])

    with open(
        "outputs/random_order_status_train_hidden_state.json",
        "a+",
        encoding="utf-8",
    ) as f:
        res = {
            "question": item["question"],
            "answer": item["answer"],
            "documents": data1[i]["documents"],
            "ans_in_docs": item["ans_in_docs"],
            "final_status": any(v["final_status"] for v in answer_set.values()),
            "answer_set": answer_set,
        }
        json.dump(res, f)
        f.write("\n")
