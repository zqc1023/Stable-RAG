import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import permutations
import argparse
import random


def set_seed(seed=42):

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA with LLaMA model and permuted docs"
    )
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--output", type=str, required=True, help="output_file")
    parser.add_argument("--model", type=str, help="model_path")

    return parser.parse_args()


def load_llama(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def load_qa_documents(data_path):
    q, a, docs = [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                q.append(d["question"])
                a.append(d["answer"])
                docs.append(d["documents"])
            except Exception as e:
                print(f"Error: {e}")
    return q, a, docs


def answer_in_docs(answer, docs):
    if isinstance(answer, list):
        return any(any(ans in doc for doc in docs) for ans in answer)
    return any(answer in doc for doc in docs)


def get_answer(documents, question, model, tokenizer):
    final_docs = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(documents, 1)]
    )
    system_prompt = "You are a helpful, respectful, and honest assistant. Answer the question with couple of words using the provided documents. For example: Question: What is the capital of France? Output: Paris."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nDocuments: {final_docs}"},
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=3096,
    )

    input_ids = tokenized.to(model.device)

    if isinstance(input_ids, dict):
        attention_mask = input_ids["attention_mask"]
        input_ids = input_ids["input_ids"]
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
        )
    new_tokens = output_ids[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(answer)
    return answer


def is_substring_match(candidate: str, answer_set: set) -> bool:
    candidate = candidate.lower().strip()
    for ans in answer_set:
        ans = ans.lower().strip()
        if candidate in ans or ans in candidate:
            return True
    return False


def main():
    args = parse_args()
    question, answer, documents = load_qa_documents(args.input)
    model, tokenizer = load_llama(args.model)
    doc_orders = list(permutations(range(5)))
    for i in tqdm(range(len(question))):
        answer_set = {}
        ans_in_docs = answer_in_docs(answer[i], documents[i])
        for order in doc_orders:
            docs = [documents[i][j] for j in order]
            ans = get_answer(
                documents=docs, question=question[i], model=model, tokenizer=tokenizer
            )
            if ans not in answer_set:
                answer_set[ans] = {"order": [order], "final_status": ""}
            else:
                answer_set[ans]["order"].append(order)
            if is_substring_match(ans, answer[i]) == True:
                answer_set[ans]["final_status"] = True
            else:
                answer_set[ans]["final_status"] = False

        with open(
            args.output,
            "a+",
            encoding="utf-8",
        ) as f:
            res = {
                "question": question[i],
                "answer": answer[i],
                "documents": documents[i],
                "ans_in_docs": ans_in_docs,
                "answer_set": answer_set,
            }

            json.dump(res, f)
            f.write("\n")


if __name__ == "__main__":
    main()
