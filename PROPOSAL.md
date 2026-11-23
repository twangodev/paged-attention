# Paged Attention

[GitHub repository](https://github.com/twangodev/paged-attention)

## Quick intro

- **Name** — Arjun Sahlot
- **Email** — <asahlot@wisc.edu>
- **Home department** — CS
- **Status** — Undergraduate

### Teammates

Everyone needs to submit the same proposal.

- James Ding <james.ding@wisc.edu>
- Geet Gambhir <ggambhir2@wisc.edu>

## Problem statement

Our goal is to reimplement PagedAttention, an algorithm introduced in the vLLM paper that improves GPU memory usage during LLM inference.

Normally, KV caches can waste a lot of GPU memory because of preallocated space and memory fragmentation. PagedAttention solves this by splitting the KV cache into smaller blocks that don't need to be stored next to each other in memory.

We want to implement this block-based memory management system along with a CUDA attention kernel that can work with these blocks. We'll focus on the core algorithm rather than building a complete LLM inference engine.

## Motivation and rationale

We wanted a project that can draw heavily on CUDA while also being technically challenging and applicable to AI infrastructure and compute.

## ECE 759 aspects the proposed work draws on

- CUDA programming and kernel execution
- Thread-level parallelism and GPU scheduling
- Memory hierarchy and caching
- GPU memory access and optimization
- Parallel reductions and synchronization

## Deliverables

We expect to present on 11/30, 12/4, or 12/7. The assigned date is to be determined.

- A KV-cache memory manager supporting fixed-size physical blocks, logical-to-physical mappings, and dynamic allocation.
- A working paged attention kernel that performs attention over noncontiguous KV-cache blocks on an NVIDIA GPU.
- An experimental evaluation of memory efficiency, kernel performance, and simulated serving capacity compared with a contiguous KV cache.

## Demonstration

We'll run benchmarks with different sequence lengths, batch sizes, and block sizes to compare GPU memory usage and execution time. We want to show how much memory PagedAttention can save and what the performance tradeoffs are compared to a regular KV cache.

We'll also demonstrate how blocks can be allocated and freed as sequences grow or finish.

## Other remarks
