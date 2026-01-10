import json
import random
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import *


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


INPUT_PATH = "outputs/random_order_final_status_train_hidden_state.json"
OUTPUT_PATH = "outputs/all_dpo_train_shuffled.json"
MODEL_PATH = "models/Llama-3-8B-Instruct"


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def id_to_docs(docs, order):
    """Reorder documents according to a given permutation."""
    return [docs[i] for i in order]


def build_dpo_sample(question, documents, chosen, rejected):
    system_prompt = (
        "You are a helpful, respectful, and honest assistant. "
        "Answer the question with a couple of words using the provided documents. "
        "For example: Question: What is the capital of France? Output: Paris."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\nDocuments: {documents}",
        },
    ]

    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    return {
        "prompt": prompt_str,
        "chosen": chosen,
        "rejected": rejected,
    }


all_correct = []
all_wrong_with_answer = []
all_wrong_without_answer = []
partially_correct = []

category_counter = Counter()


for idx, item in enumerate(data):
    print(f"Processing item {idx + 1}/{len(data)}")

    correct_orders = []
    wrong_orders = []

    for ans, info in item["answer_set"].items():
        if info["final_status"]:
            correct_orders.extend(info["order"])
        else:
            wrong_orders.extend(info["order"])

    correct_ratio = len(correct_orders) / 120
    wrong_ratio = len(wrong_orders) / 120

    print(f"Correct ratio: {correct_ratio:.2%}, Wrong ratio: {wrong_ratio:.2%}")

    # Case 1: All permutations are correct
    if abs(correct_ratio - 1.0) < 1e-6:
        category = "all_correct"
        all_correct.append(item)

    # Case 2: All permutations are wrong
    elif abs(wrong_ratio - 1.0) < 1e-6:
        documents_text = "\n".join(item["documents"]).lower()

        # Check whether a gold answer exists in documents
        gold_answer = next(
            (ans for ans in item["answer"] if ans.lower() in documents_text),
            None,
        )

        if gold_answer is not None:
            category = "all_wrong_with_gold"

            answer_set = item["answer_set"]
            wrong_answers = [
                ans for ans in answer_set if not answer_set[ans]["final_status"]
            ]

            dominant_wrong_answer = max(
                wrong_answers, key=lambda k: len(answer_set[k]["order"])
            )
            selected_order = answer_set[dominant_wrong_answer]["order"][0]

            docs_for_prompt = "\n".join(
                [
                    f"document {i}: {doc}"
                    for i, doc in enumerate(
                        id_to_docs(item["documents"], selected_order), 1
                    )
                ]
            )

            dpo_item = build_dpo_sample(
                item["question"],
                docs_for_prompt,
                gold_answer,
                "I don't know.",
            )

            all_wrong_with_answer.append(dpo_item)

        else:
            category = "all_wrong_without_gold"

            answer_set = item["answer_set"]
            dominant_answer = max(answer_set, key=lambda k: len(answer_set[k]["order"]))

            selected_order = answer_set[dominant_answer]["order"][0]

            docs_for_prompt = "\n".join(
                [
                    f"document {i}: {doc}"
                    for i, doc in enumerate(
                        id_to_docs(item["documents"], selected_order), 1
                    )
                ]
            )

            all_wrong_without_answer.append(
                build_dpo_sample(
                    item["question"],
                    docs_for_prompt,
                    "I don't know.",
                    dominant_answer,
                )
            )

    # Case 3: Mixed correct and incorrect permutations
    else:
        category = "partially_correct"
        answer_set = item["answer_set"]

        wrong_answer = max(
            {k: v for k, v in answer_set.items() if not v["final_status"]},
            key=lambda k: len(answer_set[k]["order"]),
        )
        wrong_order = answer_set[wrong_answer]["order"][0]

        docs_wrong = "\n".join(
            [
                f"document {i}: {doc}"
                for i, doc in enumerate(id_to_docs(item["documents"], wrong_order), 1)
            ]
        )

        correct_answer = max(
            {k: v for k, v in answer_set.items() if v["final_status"]},
            key=lambda k: len(answer_set[k]["order"]),
        )
        correct_order = answer_set[correct_answer]["order"][0]

        docs_correct = "\n".join(
            [
                f"document {i}: {doc}"
                for i, doc in enumerate(id_to_docs(item["documents"], correct_order), 1)
            ]
        )

        partially_correct.append(
            build_dpo_sample(item["question"], docs_wrong, correct_answer, wrong_answer)
        )

        partially_correct.append(
            build_dpo_sample(
                item["question"], docs_correct, correct_answer, wrong_answer
            )
        )

    category_counter[category] += 1


# Statistics
total = sum(category_counter.values())
print("\nCategory distribution:")
for cat, count in category_counter.items():
    print(f"{cat}: {count} ({count / total:.2%})")

print("\nDataset sizes:")
print(f"All correct: {len(all_correct)}")
print(f"All wrong (with gold): {len(all_wrong_with_answer)}")
print(f"All wrong (without gold): {len(all_wrong_without_answer)}")
print(f"Partially correct: {len(partially_correct)}")


# Merge and save DPO data
all_dpo = all_wrong_with_answer + all_wrong_without_answer + partially_correct

random.shuffle(all_dpo)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in all_dpo:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nTotal DPO samples: {len(all_dpo)}")
print(f"Saved to: {OUTPUT_PATH}")
