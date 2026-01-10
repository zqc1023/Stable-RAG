import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from itertools import permutations
import argparse
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hidden-state clustering with PCA and cluster-center answers"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/train_top5.json",
        help="Path to input QA dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results.json",
        help="Path to output file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/llama-3-8b-instruct",
        help="Path or name of the LLM",
    )
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Similarity scale for adjacency"
    )
    return parser.parse_args()


def load_llama(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def load_qa_documents(data_path):
    questions, answers, documents = [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                questions.append(d["question"])
                answers.append(d["answer"])
                documents.append(d["documents"])
            except Exception as e:
                print(f"Error parsing line: {e}")
    return questions, answers, documents


def answer_in_docs(answer, docs):
    if isinstance(answer, list):
        return any(any(ans in doc for doc in docs) for ans in answer)
    return any(answer in doc for doc in docs)


def get_last_token_hidden_state(documents, question, model, tokenizer):
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
    if isinstance(tokenized, dict):
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
    else:
        input_ids = tokenized.to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
    last_token_h = last_hidden[:, -1, :].squeeze(0).cpu().numpy()
    return last_token_h


def compute_similarity_matrix(H, sigma=0.5):
    H_norm = H / np.linalg.norm(H, axis=1, keepdims=True)
    dist = 1 - np.dot(H_norm, H_norm.T)
    A = np.exp(-dist / sigma)
    return A


def estimate_k_via_eigengap(A, max_k=15):
    D = np.diag(np.sum(A, axis=1))
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(A, axis=1) + 1e-10))
    L = np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv
    eigvals, eigvecs = eigh(L)
    gaps = np.diff(eigvals[: max_k + 1])
    K = max(2, np.argmax(gaps) + 1)
    return K


def cluster_hidden_states(H_reduced, sigma=0.5):
    A = compute_similarity_matrix(H_reduced, sigma)
    K = estimate_k_via_eigengap(A, max_k=min(20, H_reduced.shape[0] - 1))
    kmeans = KMeans(n_clusters=K, random_state=42).fit(H_reduced)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers, K


def select_representative_indices(H_reduced, labels, centers):
    rep_indices = []
    for k in np.unique(labels):
        cluster_idx = np.where(labels == k)[0]
        cluster_vecs = H_reduced[cluster_idx]
        center = centers[k]
        distances = cdist(cluster_vecs, center.reshape(1, -1))
        min_idx = cluster_idx[np.argmin(distances)]
        rep_indices.append(min_idx)
    return rep_indices  #


def generate_answer_for_rep_docs(question, rep_docs, model, tokenizer):
    final_docs = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(rep_docs, 1)]
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
        padding=True,
        truncation=True,
        max_length=2048,
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
    return answer


def main():
    args = parse_args()
    questions, answers, documents = load_qa_documents(args.input)
    model, tokenizer = load_llama(args.model)
    doc_orders = list(permutations(range(5)))

    for i in tqdm(range(len(questions))):

        question = questions[i]
        answer = answers[i]
        docs_set = documents[i]

        hidden_states_list = []
        ans_in_docs = answer_in_docs(answer, docs_set)

        for order in doc_orders:
            docs = [docs_set[j] for j in order]
            h = get_last_token_hidden_state(docs, question, model, tokenizer)
            hidden_states_list.append(h)

        H = np.stack(hidden_states_list, axis=0)  # [120, hidden_dim]

        # PCA
        pca = PCA(n_components=min(args.pca_dim, H.shape[1]))
        H_reduced = pca.fit_transform(H)

        # Clusing
        labels, centers, K = cluster_hidden_states(H_reduced, sigma=args.sigma)
        print(f"Question {i}: num_clusters = {K}")

        rep_indices = select_representative_indices(H_reduced, labels, centers)

        cluster_info = []
        for k in range(K):
            cluster_idx = np.where(labels == k)[0]
            cluster_orders = [doc_orders[idx] for idx in cluster_idx]
            rep_idx = rep_indices[k]
            rep_docs = [docs_set[j] for j in doc_orders[rep_idx]]

            # Representative Decoding
            cluster_answer = generate_answer_for_rep_docs(
                question, rep_docs, model, tokenizer
            )

            cluster_info.append(
                {
                    "cluster_id": int(k),
                    "cluster_answer": cluster_answer,
                    "all_permutations_order": cluster_orders,
                    "rep_permutation_idx": int(rep_idx),
                }
            )

        res = {
            "question": question,
            "answer": answer,
            "ans_in_docs": ans_in_docs,
            "num_clusters": int(K),
            "cluster_info": cluster_info,
        }

        with open(args.output, "a+", encoding="utf-8") as f:
            json.dump(res, f)
            f.write("\n")


if __name__ == "__main__":
    main()
