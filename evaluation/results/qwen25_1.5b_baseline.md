# Qwen2.5 1.5B Baseline

## Model

- Model: Qwen2.5 1.5B
- Evaluation prompts: 40
- Benchmark runs: 5
- Temperature: 0
- Maximum output tokens: 100
- Hardware: AMD Athlon Silver 3050U, 8 GB RAM

## Performance Results

| Metric | Average | Standard Deviation |
|---|---:|---:|
| TTFT | 0.72 sec | 1.11 sec |
| Total Latency | 9.95 sec | 0.80 sec |
| Tokens/sec | 7.85 | 0.39 |

### System Memory Snapshot

- Average memory before inference: 5.44 GB
- Average memory after inference: 5.38 GB

> Note: These are system-wide memory snapshots from psutil, not direct measurements of model memory usage.

## Quality Evaluation

- Total prompts: 40
- Correct: 31
- Incorrect: 9
- Overall accuracy: 77.50%

### Results by Category

| Category | Score | Accuracy |
|---|---:|---:|
| Knowledge | 4/5 | 80.00% |
| Explanation | 4/5 | 80.00% |
| Summarization | 5/5 | 100.00% |
| Classification | 4/5 | 80.00% |
| Information Extraction | 5/5 | 100.00% |
| Reasoning | 3/5 | 60.00% |
| Instruction Following | 2/5 | 40.00% |
| Structured Output | 4/5 | 80.00% |

## Baseline Conclusion

Qwen2.5 1.5B achieved an overall evaluation accuracy of 77.50%.

The model generated an average of 7.85 tokens/sec with an average total latency of 9.95 seconds across 5 benchmark runs.
