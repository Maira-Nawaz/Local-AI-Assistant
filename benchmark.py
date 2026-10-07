import time
import statistics
import csv
import psutil
import os
import sys

from ollama import chat


# --------------------------------------------------
# MODEL ARGUMENT
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python benchmark.py <model_name>")
    print()
    print("Examples:")
    print("python benchmark.py qwen2.5:1.5b")
    print("python benchmark.py phi3.5")
    print("python benchmark.py llama3.2:1b")
    sys.exit(1)


MODEL = sys.argv[1]

MODEL_SAFE_NAME = (
    MODEL
    .replace(":", "_")
    .replace("/", "_")
)


# --------------------------------------------------
# BENCHMARK SETTINGS
# --------------------------------------------------

PROMPT = "Explain what a data pipeline is in exactly 3 sentences."

RUNS = 5

RESULTS_FILE = (
    f"benchmark_results/{MODEL_SAFE_NAME}_baseline.csv"
)


results = []


# --------------------------------------------------
# RUN BENCHMARK
# --------------------------------------------------

for run in range(1, RUNS + 1):

    # Measure system memory before inference
    memory_before = psutil.virtual_memory().used / (1024 ** 3)

    start_time = time.perf_counter()
    first_token_time = None
    last_chunk = None

    stream = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 100
        },
        stream=True
    )

    for chunk in stream:

        last_chunk = chunk

        if first_token_time is None:
            first_token_time = time.perf_counter()

    end_time = time.perf_counter()

    # Measure system memory after inference
    memory_after = psutil.virtual_memory().used / (1024 ** 3)

    # Ollama should always return at least one chunk
    if first_token_time is None or last_chunk is None:
        print("Benchmark failed: no response received.")
        sys.exit(1)

    ttft = first_token_time - start_time

    total_latency = end_time - start_time

    output_tokens = last_chunk.eval_count

    generation_time = (
        last_chunk.eval_duration / 1_000_000_000
    )

    tokens_per_second = (
        output_tokens / generation_time
    )

    results.append({
        "ttft": ttft,
        "latency": total_latency,
        "tokens": output_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "memory_before": memory_before,
        "memory_after": memory_after
    })

    print(f"Run {run}")
    print(f"  TTFT: {ttft:.2f} sec")
    print(f"  Total latency: {total_latency:.2f} sec")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Tokens/sec: {tokens_per_second:.2f}")
    print(f"  Memory before: {memory_before:.2f} GB")
    print(f"  Memory after: {memory_after:.2f} GB")
    print()


# --------------------------------------------------
# CREATE RESULTS DIRECTORY
# --------------------------------------------------

os.makedirs("benchmark_results", exist_ok=True)


# --------------------------------------------------
# SAVE RAW RESULTS TO CSV
# --------------------------------------------------

with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "run",
        "model",
        "temperature",
        "max_output_tokens",
        "ttft_seconds",
        "total_latency_seconds",
        "output_tokens",
        "generation_time_seconds",
        "tokens_per_second",
        "memory_before_gb",
        "memory_after_gb"
    ])

    for i, result in enumerate(results, start=1):

        writer.writerow([
            i,
            MODEL,
            0,
            100,
            round(result["ttft"], 4),
            round(result["latency"], 4),
            result["tokens"],
            round(result["generation_time"], 4),
            round(result["tokens_per_second"], 4),
            round(result["memory_before"], 4),
            round(result["memory_after"], 4)
        ])


# --------------------------------------------------
# CALCULATE SUMMARY
# --------------------------------------------------

ttft_values = [
    r["ttft"]
    for r in results
]

latency_values = [
    r["latency"]
    for r in results
]

token_speed_values = [
    r["tokens_per_second"]
    for r in results
]

memory_before_values = [
    r["memory_before"]
    for r in results
]

memory_after_values = [
    r["memory_after"]
    for r in results
]


print("========================================")
print("BENCHMARK SUMMARY")
print("========================================")

print(f"Model: {MODEL}")
print(f"Runs: {RUNS}")

print(
    f"Average TTFT: "
    f"{statistics.mean(ttft_values):.2f} sec"
)

print(
    f"Average latency: "
    f"{statistics.mean(latency_values):.2f} sec"
)

print(
    f"Average tokens/sec: "
    f"{statistics.mean(token_speed_values):.2f}"
)

print(
    f"TTFT std deviation: "
    f"{statistics.stdev(ttft_values):.2f}"
)

print(
    f"Latency std deviation: "
    f"{statistics.stdev(latency_values):.2f}"
)

print(
    f"Tokens/sec std deviation: "
    f"{statistics.stdev(token_speed_values):.2f}"
)

print(
    f"Average memory before: "
    f"{statistics.mean(memory_before_values):.2f} GB"
)

print(
    f"Average memory after: "
    f"{statistics.mean(memory_after_values):.2f} GB"
)

print()
print(f"Results saved to: {RESULTS_FILE}")
print("========================================")