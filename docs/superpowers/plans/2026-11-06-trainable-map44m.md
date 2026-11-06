# Trainable MAP-44M Implementation Plan

> For agentic workers: use superpowers:executing-plans task-by-task.

**Goal:** A runnable, auditable paper-only scratch LM with resumable pretraining,
answer-masked SFT, supervised corrections/replay, generation and a bounded Modal runner.
**Architecture:** Follow `docs/DECISION.md`; use PyTorch SDPA without custom CUDA.
Data and tokenizer are immutable artifacts. Training and update share one trainer.
**Tech Stack:** Python >=3.11, PyTorch 2.10, NumPy; optional Modal/Hugging Face/PyArrow.
**Spec:** docs/DECISION.md

## Global Constraints
- 12 layers, width 512, 8 query / 2 KV heads, FFN 1536, vocab 16384;
  tied embedding/output, no biases, 44,577,280 unique parameters.
- Scientific source data are not bundled. Synthetic fixtures are software tests only.
- No charged cloud execution, no secrets, no weights or raw papers in GitHub.
- Never claim scientific correctness, GPU speed, or provider spend limits from CPU tests.
- Preserve default branch; publish a feature branch and PR, not an automatic merge.

## Review Focus
- Future information leakage and cached non-square attention masks.
- Response-only labels, next-token alignment and token-weighted accumulation.
- Resume must restore optimizer/RNG/sampler/counters and reject changed data/tokenizer.
- Unknown/unverified provenance, duplicate papers across splits, invalid replay inputs.
- Budget guardrails, accidental output overwrite, public artifact secret/data scanning.

## Tasks
1. [x] Model/config: test counts, causality, GQA reference, cache equivalence and gradients;
   implement `ModelConfig` and `PaperLM`. Run pytest; commit.
2. [x] Data/tokenizer: test UTF-8 round trip, BPE merges, split isolation, labels;
   implement deterministic byte-BPE and bounded JSONL-to-memmap preparation. Run tests; commit.
3. [x] Trainer: test real loss decrease, exact CPU resume, masked accumulation, replay
   accounting and invalid artifact refusal; implement checkpoint/train/eval/generate. Commit.
4. [x] Entry points: test CLI end-to-end and argument safety; add optional HF sample
   importer, bounded Modal job, CPU CI, Japanese runbook and research limitations. Commit.
5. [ ] Final self-review: inspect each module against Review Focus, add regression tests
   for findings, run the entire suite and a full-size CPU forward/backward, record results,
   scan public contents, publish branch/PR and verify remote files.

## Interface map
- ModelConfig -> PaperLM: validated dimensions, total parameter count, max context.
- ByteBPE -> corpus manifest: immutable SHA-256, vocabulary capacity and control IDs.
- PreparedCorpus -> trainer: memmapped documents, answer mask and byte-length accounting.
- StatefulStream -> checkpoint: generator state, permutation/cursor, observed source tokens.
- Trainer -> CLI/Modal: explicit fresh/resume/init-from modes and per-run bounds.

## Rulings
- Network clone and package installation unavailable in the container. Reconstruct a
  fresh isolated local branch from the fetched public README; use GitHub connector to
  publish. Use a tested stdlib byte-BPE rather than an untestable required tokenizer wheel.
- No subagent tool is available: separate final author self-review, explicitly disclosed.
