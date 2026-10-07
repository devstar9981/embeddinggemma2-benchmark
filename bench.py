"""EmbeddingGemma 2 vs EmbeddingGemma on code search, at 768, 512, 256 and 128 dimensions.

    python bench.py quality --model eg1          # retrieval quality (embeddings are cached in cache/)
    python bench.py quality --model eg2
    python bench.py speed --model eg2            # RAM, query latency and throughput on this machine
    python bench.py speed --model eg2-full
    python bench.py speed --model eg1
    python bench.py report                       # writes results/RESULTS.md

Both models get the same inputs, using the prefixes from the EmbeddingGemma 2 model card:
queries "task: code retrieval | query: {query}", documents "title: none | text: {code}".
Each corpus is encoded once at 768 dimensions. Shorter sizes keep the leading dimensions and
re-normalize, which is what sentence-transformers' truncate_dim does.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

MODELS = {
    # Byte-identical copy of google/embeddinggemma-300m (same SHA-256 for every weight file),
    # which needs a Hugging Face login to download.
    "eg1": {
        "id": "unsloth/embeddinggemma-300m",
        "revision": "bfa3c846ac738e62aa61806ef9112d34acb1dc5a",
        "dir": "embeddinggemma-300m",
        "label": "EmbeddingGemma (300M)",
        "kwargs": {},
    },
    "eg2": {
        "id": "google/embeddinggemma-2",
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "dir": "embeddinggemma-2",
        "label": "EmbeddingGemma 2, text only (270M)",
        "kwargs": {"config_kwargs": {"vision_config": None, "audio_config": None}},
    },
    "eg2-full": {
        "id": "google/embeddinggemma-2",
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "dir": "embeddinggemma-2",
        "label": "EmbeddingGemma 2, all encoders (740M)",
        "kwargs": {},
    },
}

QUERY_PREFIX = "task: code retrieval | query: "
DOC_PREFIX = "title: none | text: "
DIMS = [768, 512, 256, 128]
CSN_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
TASKS = [f"csn-{lang}" for lang in CSN_LANGS] + ["cosqa"]
MAX_SEQ_LENGTH = 1024


# ---------- data ----------


def load_task(task):
    """Returns (query ids, query texts, doc ids, doc texts, qrels {query id: {doc id: grade}})."""
    from datasets import load_dataset

    if task.startswith("csn-"):
        lang = task[4:]
        name = "mteb/CodeSearchNetRetrieval"
        corpus = load_dataset(name, f"{lang}-corpus", split="test")
        queries = load_dataset(name, f"{lang}-queries", split="test")
        qrels = load_dataset(name, f"{lang}-qrels", split="test")
    elif task == "cosqa":
        name = "mteb/cosqa"
        corpus = load_dataset(name, "corpus", split="test")
        queries = load_dataset(name, "queries", split="test")
        qrels = load_dataset(name, "qrels", split="test")
    else:
        raise ValueError(task)

    # CodeSearchNet names the id column "id", CosQA "_id"; scores come as strings.
    def key(row):
        return str(row["_id"] if "_id" in row else row["id"])

    rel = {}
    for row in qrels:
        if float(row["score"]) > 0:
            rel.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row["score"])
    # Only queries that have a relevant document are scored.
    queries = [q for q in queries if key(q) in rel]

    def doc_text(d):
        title = (d.get("title") or "").strip()
        return f"{title}\n{d['text']}" if title else d["text"]

    return (
        [key(q) for q in queries],
        [q["text"] for q in queries],
        [key(d) for d in corpus],
        [doc_text(d) for d in corpus],
        rel,
    )


# ---------- model ----------


def load_model(key, device="cpu"):
    import torch
    from sentence_transformers import SentenceTransformer

    spec = MODELS[key]
    # A local copy in models/<dir> (e.g. from download.sh) is used when present; otherwise the pinned
    # revision is downloaded from Hugging Face.
    local = ROOT / "models" / spec["dir"]
    source = {"model_name_or_path": str(local)} if (local / "model.safetensors").exists() else {
        "model_name_or_path": spec["id"],
        "revision": spec["revision"],
    }
    # float32: the model card warns against float16, and bfloat16 is slow on most CPUs.
    model = SentenceTransformer(
        **source, device=device, model_kwargs={"torch_dtype": torch.float32}, **spec["kwargs"]
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def encode(model, texts, prefix, batch_size):
    return model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def truncate(emb, dim):
    out = emb[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ---------- metrics ----------


def evaluate(q, d, qids, dids, qrels, k=10):
    scores = q @ d.T
    top = np.argsort(-scores, axis=1)[:, :k]
    ndcg, mrr, r1, r10 = [], [], [], []
    for i, qid in enumerate(qids):
        rel = qrels[qid]
        ranked = [dids[j] for j in top[i]]
        gains = [rel.get(doc, 0) for doc in ranked]
        dcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(ideal))
        ndcg.append(dcg / idcg)
        first = next((rank for rank, g in enumerate(gains) if g > 0), None)
        mrr.append(0 if first is None else 1 / (first + 1))
        r1.append(sum(1 for doc in ranked[:1] if doc in rel) / len(rel))
        r10.append(sum(1 for doc in ranked if doc in rel) / len(rel))
    return {
        "ndcg@10": 100 * float(np.mean(ndcg)),
        "mrr@10": 100 * float(np.mean(mrr)),
        "recall@1": 100 * float(np.mean(r1)),
        "recall@10": 100 * float(np.mean(r10)),
        "queries": len(qids),
        "docs": len(dids),
    }


# ---------- commands ----------


def environment():
    import sentence_transformers
    import torch
    import transformers

    cpu = platform.processor()
    try:
        cpu = (
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            or cpu
        )
    except Exception:
        pass
    return {
        "cpu": cpu,
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "max_seq_length": MAX_SEQ_LENGTH,
    }


def cmd_quality(args):
    model = load_model(args.model, args.device)
    out = {"model": MODELS[args.model], "environment": environment(), "tasks": {}}
    for task in args.tasks:
        qids, qtexts, dids, dtexts, qrels = load_task(task)
        folder = CACHE / args.model
        folder.mkdir(parents=True, exist_ok=True)
        qfile, dfile = folder / f"{task}-queries.npy", folder / f"{task}-corpus.npy"
        if not dfile.exists():
            started = time.perf_counter()
            np.save(dfile, encode(model, dtexts, DOC_PREFIX, args.batch_size))
            print(f"{task}: encoded {len(dtexts)} docs in {time.perf_counter() - started:.0f}s")
        if not qfile.exists():
            np.save(qfile, encode(model, qtexts, QUERY_PREFIX, args.batch_size))
        q, d = np.load(qfile), np.load(dfile)
        out["tasks"][task] = {str(dim): evaluate(truncate(q, dim), truncate(d, dim), qids, dids, qrels) for dim in DIMS}
        print(task, {dim: round(m["ndcg@10"], 2) for dim, m in out["tasks"][task].items()})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"quality-{args.model}.json").write_text(json.dumps(out, indent=2))


def cmd_speed(args):
    import psutil
    import torch

    proc = psutil.Process()
    rss = lambda: proc.memory_info().rss / 2**20  # noqa: E731
    base = rss()
    started = time.perf_counter()
    model = load_model(args.model, "cpu")
    load_s = time.perf_counter() - started
    after_load = rss()

    _, qtexts, _, dtexts, _ = load_task("cosqa")
    queries = qtexts[: args.queries]
    docs = dtexts[: args.docs]
    with torch.inference_mode():
        for text in queries[:10]:  # warm-up
            model.encode(QUERY_PREFIX + text)
        latencies = []
        for text in queries:
            t = time.perf_counter()
            model.encode(QUERY_PREFIX + text, normalize_embeddings=True)
            latencies.append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        model.encode([DOC_PREFIX + doc for doc in docs], batch_size=32, normalize_embeddings=True)
        docs_per_s = len(docs) / (time.perf_counter() - t)

    params = sum(p.numel() for p in model.parameters())
    mem = proc.memory_info()
    out = {
        "model": MODELS[args.model],
        "environment": environment(),
        "parameters_millions": round(params / 1e6, 1),
        "load_seconds": round(load_s, 1),
        "rss_model_mb": round(after_load - base),
        "rss_after_encoding_mb": round(rss()),
        "peak_working_set_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20),
        "query_latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
            "queries": len(latencies),
        },
        "docs_per_second": round(docs_per_s, 1),
        "docs": len(docs),
    }
    print(json.dumps(out, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"speed-{args.model}.json").write_text(json.dumps(out, indent=2))


def cmd_report(args):
    def load(name):
        path = RESULTS / name
        return json.loads(path.read_text()) if path.exists() else None

    lines = ["# Results", ""]
    q1, q2 = load("quality-eg1.json"), load("quality-eg2.json")
    if q1 and q2:
        env = q2["environment"]
        lines += [
            f"Embeddings in float32 with transformers {env['transformers']} and sentence-transformers "
            f"{env['sentence_transformers']}, max sequence length {env['max_seq_length']}. "
            "CPU and GPU runs give the same vectors (see README).",
            "",
            "## NDCG@10 by vector size",
            "",
            "| Task | Queries / docs | Model | 768 | 512 | 256 | 128 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for task in q2["tasks"]:
            for key, q in (("eg1", q1), ("eg2", q2)):
                if task not in q["tasks"]:
                    continue
                m = q["tasks"][task]
                size = f"{m['768']['queries']:,} / {m['768']['docs']:,}"
                lines.append(
                    f"| {task} | {size} | {MODELS[key]['label']} | "
                    + " | ".join(f"{m[str(d)]['ndcg@10']:.1f}" for d in DIMS)
                    + " |"
                )
        lines += ["", "## Recall@10 by vector size", "", "| Task | Model | 768 | 512 | 256 | 128 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for task in q2["tasks"]:
            for key, q in (("eg1", q1), ("eg2", q2)):
                if task in q["tasks"]:
                    m = q["tasks"][task]
                    lines.append(f"| {task} | {MODELS[key]['label']} | " + " | ".join(f"{m[str(d)]['recall@10']:.1f}" for d in DIMS) + " |")

    speeds = [(k, load(f"speed-{k}.json")) for k in MODELS]
    speeds = [(k, s) for k, s in speeds if s]
    if speeds:
        lines += [
            "",
            "## Speed and memory (CPU, float32)",
            "",
            f"{speeds[0][1]['environment']['cpu']}, {speeds[0][1]['environment']['threads']} threads, "
            f"torch {speeds[0][1]['environment']['torch']}. RAM is the process memory added by loading the model.",
            "",
            "| Model | Parameters loaded | RAM for the model | Query latency, median (p95) | Docs per second (batch 32) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for key, s in speeds:
            lat = s["query_latency_ms"]
            lines.append(
                f"| {MODELS[key]['label']} | {s['parameters_millions']}M | {s['rss_model_mb']:,} MB | "
                f"{lat['median']} ms ({lat['p95']} ms) | {s['docs_per_second']} |"
            )
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--model", choices=["eg1", "eg2"], required=True)
    q.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    q.add_argument("--device", default="cpu")
    q.add_argument("--batch-size", type=int, default=16)
    q.set_defaults(run=cmd_quality)
    s = sub.add_parser("speed")
    s.add_argument("--model", choices=list(MODELS), required=True)
    s.add_argument("--queries", type=int, default=200)
    s.add_argument("--docs", type=int, default=512)
    s.set_defaults(run=cmd_speed)
    r = sub.add_parser("report")
    r.set_defaults(run=cmd_report)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.run(args)


if __name__ == "__main__":
    main()
