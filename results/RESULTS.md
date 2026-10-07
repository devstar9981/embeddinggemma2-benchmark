# Results

Embeddings in float32 with transformers 5.19.0 and sentence-transformers 6.1.0, max sequence length 1024. CPU and GPU runs give the same vectors (see README).

## NDCG@10 by vector size

| Task | Queries / docs | Model | 768 | 512 | 256 | 128 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| csn-python | 1,000 / 1,000 | EmbeddingGemma (300M) | 96.1 | 95.9 | 95.7 | 93.9 |
| csn-python | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 95.5 | 95.5 | 95.2 | 93.4 |
| csn-javascript | 1,000 / 1,000 | EmbeddingGemma (300M) | 83.4 | 82.4 | 80.4 | 77.4 |
| csn-javascript | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 82.5 | 82.3 | 81.6 | 78.1 |
| csn-go | 1,000 / 1,000 | EmbeddingGemma (300M) | 93.9 | 93.7 | 93.1 | 89.7 |
| csn-go | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 96.7 | 96.7 | 96.5 | 95.1 |
| csn-ruby | 1,000 / 1,000 | EmbeddingGemma (300M) | 87.4 | 86.3 | 85.4 | 82.1 |
| csn-ruby | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 88.7 | 88.4 | 87.9 | 84.6 |
| csn-java | 1,000 / 1,000 | EmbeddingGemma (300M) | 91.4 | 91.2 | 88.9 | 87.0 |
| csn-java | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 91.6 | 91.7 | 90.8 | 89.3 |
| csn-php | 1,000 / 1,000 | EmbeddingGemma (300M) | 88.6 | 88.6 | 87.3 | 85.7 |
| csn-php | 1,000 / 1,000 | EmbeddingGemma 2, text only (270M) | 89.6 | 89.5 | 89.1 | 87.2 |
| cosqa | 500 / 20,604 | EmbeddingGemma (300M) | 44.7 | 42.5 | 40.8 | 38.7 |
| cosqa | 500 / 20,604 | EmbeddingGemma 2, text only (270M) | 49.4 | 49.1 | 46.2 | 42.0 |

## Recall@10 by vector size

| Task | Model | 768 | 512 | 256 | 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| csn-python | EmbeddingGemma (300M) | 99.3 | 99.1 | 99.3 | 99.0 |
| csn-python | EmbeddingGemma 2, text only (270M) | 99.3 | 99.3 | 99.0 | 98.2 |
| csn-javascript | EmbeddingGemma (300M) | 89.6 | 88.7 | 87.7 | 85.7 |
| csn-javascript | EmbeddingGemma 2, text only (270M) | 88.9 | 88.9 | 88.4 | 85.6 |
| csn-go | EmbeddingGemma (300M) | 99.0 | 99.0 | 99.0 | 97.5 |
| csn-go | EmbeddingGemma 2, text only (270M) | 99.6 | 99.6 | 99.6 | 99.4 |
| csn-ruby | EmbeddingGemma (300M) | 94.0 | 93.1 | 92.9 | 91.4 |
| csn-ruby | EmbeddingGemma 2, text only (270M) | 94.1 | 94.1 | 94.4 | 92.5 |
| csn-java | EmbeddingGemma (300M) | 96.8 | 96.8 | 96.3 | 95.8 |
| csn-java | EmbeddingGemma 2, text only (270M) | 96.6 | 96.7 | 96.5 | 96.2 |
| csn-php | EmbeddingGemma (300M) | 95.1 | 95.0 | 94.4 | 93.8 |
| csn-php | EmbeddingGemma 2, text only (270M) | 95.9 | 95.9 | 95.8 | 94.8 |
| cosqa | EmbeddingGemma (300M) | 75.0 | 71.6 | 70.0 | 67.4 |
| cosqa | EmbeddingGemma 2, text only (270M) | 81.4 | 82.0 | 77.2 | 70.2 |

## Speed and memory (CPU, float32)

Intel(R) Core(TM) i7-9700F CPU @ 3.00GHz, 8 threads, torch 2.14.1+cpu. RAM is the process memory added by loading the model.

| Model | Parameters loaded | RAM for the model | Query latency, median (p95) | Docs per second (batch 32) |
| --- | ---: | ---: | ---: | ---: |
| EmbeddingGemma (300M) | 307.6M | 1,668 MB | 67.3 ms (93.5 ms) | 7.9 |
| EmbeddingGemma 2, text only (270M) | 271.0M | 1,528 MB | 80.8 ms (100.1 ms) | 5.2 |
| EmbeddingGemma 2, all encoders (740M) | 744.4M | 3,346 MB | 85.2 ms (110.2 ms) | 5.5 |
