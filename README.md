# paged-attention

ECE 759 final project. rebuilding PagedAttention (the vllm memory trick) from scratch in c++/cuda

## current idea

- llm writes 1 token at a time. attention looks back at every past token's k, v -> kept around in the "kv cache"
- naive way: reserve max length (like 2048 tokens) per request upfront, one contiguous chunk. most of it sits empty -> fewer users fit on the gpu
- paged way: chop memory into fixed blocks (~16 tokens each), hand them out only as a request grows. per-request block table = list of which blocks it owns. basically os virtual memory

## what we're NOT doing

- not implementing a full decoder in c++. attention is the only step that looks at past tokens -> cut it out, test it alone w/ random q/k/v

## build + test

```bash
cmake -B build && cmake --build build
./build/test_attention_cpu && ./build/test_attention_gpu
```

## layout

```
src/            one header + source per piece 
tests/          test_<piece>, gpu pieces checked against attention_cpu
bench/          kernel benchmarks + plots
visualizer/     browser visualizer (stretch)
```

## checklist

- [x] 1. cpu attention
- [ ] 2. gpu attention, naive kv (partially done)
- [ ] 3. block manager + randomized tests
- [ ] 4. paged attention kernel, matches cpu (proposal core done here)
- [ ] stretch: faster kernel (close to parity with vllm?)
- [ ] 6. benchmarks: paged vs contiguous, block sizes 8–128, ms + gb/s
- [ ] stretch: side-by-side memory map visualizer (browser)
- [ ] stretch: real small model (qwen2.5-0.5b / tinyllama / qwen3-0.6b?) in pytorch, calls our kernel via torch extension

## demo

- lvl 1: paged attention performance improvement charts, benchmark results, etc.
- lvl 2: live side-by-side memory map, reserve-max vs paged. users served, % wasted, waiting line
- lvl 3: same viz, but real chats typing out from a real model (need to integrate our attention code with pytorch)
