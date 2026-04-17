#!/usr/bin/env python3
"""
MEGADOC × FinePhrase — Phase 1 Pipeline (Local vLLM Batch Mode, Multi-GPU)
Generates FAQ/Math/Table/Tutorial rephrasings from DCLM and assembles stitched megadocs.
Pushes completed shards to a HuggingFace dataset repo.

Uses data parallelism across multiple GPUs for maximum throughput.

Usage:
    python pipeline.py --output-repo essobi/dclm-crossover-megadoc-p1
"""

import argparse
import json
import os
import tempfile
from itertools import islice
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SOURCE_DATASET = "essobi/dclm-crossover-source"

MAX_INPUT_CHARS = 20_000    # ~5k tokens @ ~4 chars/tok (leaves room for output in 8k ctx)
MAX_MODEL_LEN   = 8192      # total context window
MAX_OUTPUT_TOK  = 2048      # max generation tokens
BATCH_SIZE      = 256       # docs per GPU per batch (256 * 8 GPUs = 2048 total)
SHARD_SIZE      = 1_000     # docs per pushed parquet shard
NUM_GPUS        = 1         # number of GPUs to use


# ─────────────────────────────────────────────
# Prompts (verbatim from FinePhrase paper)
# ─────────────────────────────────────────────

PROMPTS = {
    "faq": (
        "Here is a document:\n\n"
        "<document>\n{document}\n</document>\n\n"
        "Rewrite the document above as a comprehensive FAQ (Frequently Asked Questions).\n"
        "Extract or infer the key questions a reader would have, then provide clear,\n"
        "direct answers. Order questions logically. Each answer should be self-contained.\n"
        "Do not include the document tags in your output. Output only the FAQ text."
    ),
    "math": (
        "Here is a document:\n\n"
        "<document>\n{document}\n</document>\n\n"
        "Using the document above, create a mathematical word problem based on the\n"
        "numerical data or relationships in the text. Then provide a step-by-step solution\n"
        "showing each calculation. The problem should require multi-step reasoning and\n"
        "basic arithmetic. Do not include the document tags in your output.\n"
        "Output only the problem and solution."
    ),
    "table": (
        "Here is a document:\n\n"
        "<document>\n{document}\n</document>\n\n"
        "Rewrite the document above as a markdown table that organizes the key information,\n"
        "then generate one insightful question-answer pair based on the table data.\n"
        "Use proper markdown table syntax with headers. Do not include the document tags\n"
        "in your output. Output only the table followed by the question-answer pair."
    ),
    "tutorial": (
        "Here is a document:\n\n"
        "<document>\n{document}\n</document>\n\n"
        "Rewrite the document above as a clear, step-by-step tutorial or instructional\n"
        "guide. Use numbered steps where appropriate. Preserve all essential information\n"
        "while making the style didactic and easy to follow. Do not include the document\n"
        "tags in your output. Output only the tutorial."
    ),
}


def strip_leaked_tags(text: str) -> str:
    """Remove any <document>/<faq>/etc tags that leaked into output."""
    import re
    return re.sub(r"</?(?:document|faq|tutorial|table|problem|answer|question)>", "", text).strip()


def has_numerical_content(text: str) -> bool:
    """True if doc has enough numbers to be worth generating a math problem."""
    import re
    # Require at least 3 multi-digit numbers
    numbers = re.findall(r"\b\d{2,}\b", text)
    return len(numbers) >= 3


def is_valid_markdown_table(text: str) -> bool:
    """True if output contains a valid markdown table (pipe rows + separator)."""
    import re
    # Markdown table requires a separator row like |---|---|
    return bool(re.search(r"\|[\s:-]*\|[\s:-]*(?:\|)?", text)) and text.count("|") >= 4

REPHRASING_ORDER = ("faq", "math", "table", "tutorial")


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def truncate(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Truncate at a newline boundary to avoid mid-sentence cuts."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[:cut] if cut > 0 else text[:max_chars]


def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def get_text(row: dict) -> str:
    return row.get("text") or row.get("content") or row.get("raw_content") or ""


# ─────────────────────────────────────────────
# GPU Worker (runs in dedicated process per GPU)
# ─────────────────────────────────────────────

def gpu_worker_loop(gpu_id: int, model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker loop: load model once, then process batches from queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=2048,
        max_num_batched_tokens=16384,
        enable_prefix_caching=True,
        speculative_config={
            # Change "suffix" to one of the supported strings below
            "method": "ngram", 
            "num_speculative_tokens": 32,
        },
        trust_remote_code=True,
        #dtype="half",
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOK)

    # Signal ready
    result_queue.put(("ready", gpu_id))

    while True:
        task = task_queue.get()
        if task is None:  # Shutdown signal
            break

        task_id, batch_data = task

        prompt_map = []
        conversations = []
        valid_docs = []

        for batch_pos, (idx, row) in enumerate(batch_data):
            text = get_text(row)
            if not text.strip():
                continue
            valid_docs.append((batch_pos, idx, row, text))

        for batch_pos, idx, row, text in valid_docs:
            truncated = truncate(text)
            has_numbers = has_numerical_content(truncated)
            for rephrase_type in REPHRASING_ORDER:
                # Skip math for non-numerical docs
                if rephrase_type == "math" and not has_numbers:
                    continue
                prompt = PROMPTS[rephrase_type].format(document=truncated)
                conversations.append([{"role": "user", "content": prompt}])
                prompt_map.append((batch_pos, rephrase_type))

        if not conversations:
            result_queue.put((task_id, []))
            continue

        outputs = llm.chat(conversations, sampling_params)

        doc_outputs = {}
        for (batch_pos, rephrase_type), output in zip(prompt_map, outputs):
            cleaned = strip_leaked_tags(output.outputs[0].text)
            # Discard table output if not valid markdown
            if rephrase_type == "table" and not is_valid_markdown_table(cleaned):
                cleaned = ""
            if batch_pos not in doc_outputs:
                doc_outputs[batch_pos] = {}
            doc_outputs[batch_pos][rephrase_type] = cleaned

        results = []
        for batch_pos, idx, row, text in valid_docs:
            rephrasings = doc_outputs.get(batch_pos, {})
            doc_id = str(row.get("id", idx))

            parts = [rephrasings.get(k, "") for k in REPHRASING_ORDER if rephrasings.get(k)]
            parts.append(text)
            megadoc = "\n\n".join(parts)

            results.append({
                "id":         doc_id,
                "megadoc":    megadoc,
                "text":       text,
                "url":        row.get("url") or "",
                "word_count": int(row.get("word_count") or 0),
                "faq":        rephrasings.get("faq") or "",
                "math":       rephrasings.get("math") or "",
                "table":      rephrasings.get("table") or "",
                "tutorial":   rephrasings.get("tutorial") or "",
            })

        result_queue.put((task_id, results))


class GPUWorkerPool:
    """Pool of dedicated GPU workers for data-parallel inference."""

    def __init__(self, num_gpus: int, model_name: str):
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.workers = []
        self.task_queues = []
        self.result_queue = None
        self._task_counter = 0

    def start(self):
        """Start worker processes and load models."""
        print(f"Starting {self.num_gpus} GPU workers...")
        ctx = mp.get_context("spawn")
        self.result_queue = ctx.Queue()

        for gpu_id in range(self.num_gpus):
            task_queue = ctx.Queue()
            self.task_queues.append(task_queue)

            p = ctx.Process(
                target=gpu_worker_loop,
                args=(gpu_id, self.model_name, task_queue, self.result_queue),
                daemon=False,
            )
            p.start()
            self.workers.append(p)

        # Wait for all workers to signal ready
        ready_count = 0
        while ready_count < self.num_gpus:
            msg, gpu_id = self.result_queue.get()
            if msg == "ready":
                print(f"  GPU {gpu_id} ready")
                ready_count += 1

        print(f"All {self.num_gpus} GPU workers ready\n")

    def process_batches(self, batches: list[list[tuple[int, dict]]]) -> list[list[dict]]:
        """Process multiple batches in parallel across GPUs."""
        # Submit tasks
        task_ids = []
        for gpu_id, batch in enumerate(batches):
            if gpu_id >= self.num_gpus:
                break
            task_id = self._task_counter
            self._task_counter += 1
            task_ids.append(task_id)
            self.task_queues[gpu_id].put((task_id, batch))

        # Collect results
        results_map = {}
        while len(results_map) < len(task_ids):
            task_id, results = self.result_queue.get()
            results_map[task_id] = results

        return [results_map[tid] for tid in task_ids]

    def shutdown(self):
        """Stop all workers."""
        for q in self.task_queues:
            q.put(None)  # Shutdown signal
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ─────────────────────────────────────────────
# HuggingFace push
# ─────────────────────────────────────────────

SCHEMA = pa.schema([
    pa.field("id",            pa.string()),
    pa.field("megadoc",       pa.string()),
    pa.field("text",          pa.string()),
    pa.field("url",           pa.string()),
    pa.field("word_count",    pa.int32()),
    pa.field("faq",           pa.string()),
    pa.field("math",          pa.string()),
    pa.field("table",         pa.string()),
    pa.field("tutorial",      pa.string()),
])


def push_shard(
    records: list[dict],
    repo_id: str,
    shard_idx: int,
    api: HfApi,
    tmpdir: str,
    worker_id: str = "",
) -> None:
    prefix = f"{worker_id}-" if worker_id else ""
    name = f"shard-{prefix}{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    local = os.path.join(tmpdir, name)
    pq.write_table(table, local)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=f"data/train/{name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    os.unlink(local)
    print(f"\n  → {name} pushed ({len(records)} docs)")


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: str):
        self.path = Path(path)
        self.done: set[str] = set()
        if self.path.exists():
            self.done = set(json.loads(self.path.read_text()))
        print(f"Checkpoint: {len(self.done):,} already done.")

    def is_done(self, doc_id: str) -> bool:
        return doc_id in self.done

    def add(self, doc_id: str):
        self.done.add(doc_id)

    def save(self):
        self.path.write_text(json.dumps(list(self.done)))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run(args):
    num_gpus  = args.num_gpus
    worker_id = args.worker_id

    print(f"Streaming {SOURCE_DATASET} (shard {args.shard_index}/{args.num_shards})...")
    if worker_id:
        print(f"Worker: '{worker_id}'")

    ds = load_dataset(SOURCE_DATASET, split="train", streaming=True)
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)
    try:
        total = ds.info.splits["train"].num_examples
        if args.num_shards > 1:
            total = total // args.num_shards
        print(f"  ~{total:,} documents in slice")
    except Exception:
        total = None
        print("  (slice size unknown)")

    limit_arg = args.limit if args.limit > 0 else float("inf")
    limit     = limit_arg

    # Initialize GPU worker pool
    pool = GPUWorkerPool(num_gpus, args.model)
    pool.start()

    checkpoint = Checkpoint(args.checkpoint)
    api = HfApi() if not args.dry_run else None
    if not args.dry_run:
        api.create_repo(args.output_repo, repo_type="dataset", exist_ok=True)

    buffer: list[dict] = []
    pending_uploads: list = []  # futures from async upload executor
    shard_idx = 0
    docs_processed = 0

    # Batch size per iteration = BATCH_SIZE * num_gpus
    multi_batch_size = BATCH_SIZE * num_gpus

    def safe_iter(ds):
        """Iterate over dataset, skipping corrupted rows."""
        it = iter(ds)
        idx = 0
        while True:
            try:
                row = next(it)
                yield idx, row
                idx += 1
            except StopIteration:
                break
            except Exception as e:
                print(f"\n[WARN] Skipping corrupted row {idx}: {e}")
                idx += 1

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_pool = ThreadPoolExecutor(max_workers=2)
            effective_total = int(limit) if limit != float("inf") else None
            with tqdm(total=effective_total, desc="docs", unit="doc", smoothing=0.01) as pbar:
                for mega_batch in chunked(safe_iter(ds), multi_batch_size):
                    if docs_processed >= limit:
                        break

                    pending = [
                        (idx, row) for idx, row in mega_batch
                        if not checkpoint.is_done(str(row.get("id", idx)))
                    ]
                    # Respect limit
                    remaining = limit - docs_processed
                    if len(pending) > remaining:
                        pending = pending[:int(remaining)]

                    pbar.update(len(mega_batch) - len(pending))  # account for skipped

                    if not pending:
                        continue

                    # Split into per-GPU batches
                    gpu_batches = [pending[i::num_gpus] for i in range(num_gpus)]
                    gpu_batches = [b for b in gpu_batches if b]  # remove empty

                    # Process in parallel across GPUs
                    all_results = pool.process_batches(gpu_batches)

                    for results in all_results:
                        for result in results:
                            buffer.append(result)
                            checkpoint.add(result["id"])
                            pbar.update(1)
                            docs_processed += 1

                    # Async-push complete shards (don't block inference)
                    while len(buffer) >= SHARD_SIZE and not args.dry_run:
                        shard_records = buffer[:SHARD_SIZE]
                        buffer = buffer[SHARD_SIZE:]
                        fut = upload_pool.submit(
                            push_shard, shard_records, args.output_repo,
                            shard_idx, api, tmpdir, worker_id,
                        )
                        pending_uploads.append(fut)
                        shard_idx += 1
                        checkpoint.save()

                # Wait for in-flight uploads before final shard
                for fut in pending_uploads:
                    fut.result()
                pending_uploads.clear()

                # Final partial shard
                if buffer and not args.dry_run:
                    push_shard(buffer, args.output_repo, shard_idx, api, tmpdir, worker_id)
                    checkpoint.save()

            upload_pool.shutdown(wait=True)

    finally:
        pool.shutdown()

    if args.dry_run:
        print(f"\n[DRY RUN] Processed {docs_processed} docs, {len(buffer)} in buffer (not pushed)")
        if buffer:
            print(f"Sample output keys: {list(buffer[0].keys())}")
            print(f"Sample megadoc length: {len(buffer[0]['megadoc'])} chars")
    else:
        print(f"\nDone. {len(checkpoint.done):,} docs → {args.output_repo}")


def main():
    parser = argparse.ArgumentParser(description="MEGADOC Phase 1 pipeline (local vLLM batch, multi-GPU)")
    parser.add_argument(
        "--output-repo", required=True,
        help="HF dataset repo to push to (e.g. essobi/dclm-crossover-megadoc-p1)",
    )
    parser.add_argument(
        "--checkpoint", default=".checkpoint.json",
        help="JSON file tracking completed doc IDs (default: .checkpoint.json)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help=f"Number of GPUs to use (default: {NUM_GPUS})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to N documents within the slice (0 = no limit, for testing)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Which shard this worker processes, 0-based (default: 0)",
    )
    parser.add_argument(
        "--worker-id", type=str, default="",
        help="Unique ID for this worker, used to namespace output shard filenames (e.g. 'w0', 'w1')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process docs but don't push to HuggingFace",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
