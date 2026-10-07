import csv
import statistics
from pathlib import Path


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

BENCHMARK_FILE = Path(
    "benchmark_results/qwen25_1.5b_baseline.csv"
)

SCORED_FILE = Path(
    "evaluation/results/qwen25_1.5b_scored.csv"
)

OUTPUT_FILE = Path(
    "evaluation/results/qwen25_1.5b_baseline.md"
)


# ---------------------------------------------------------
# READ PERFORMANCE RESULTS
# ---------------------------------------------------------

with open(
    BENCHMARK_FILE,
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    benchmark_rows = list(reader)


ttft_values = [
    float(row["ttft_seconds"])
    for row in benchmark_rows
]

latency_values = [
    float(row["total_latency_seconds"])
    for row in benchmark_rows
]

tokens_per_second_values = [
    float(row["tokens_per_second"])
    for row in benchmark_rows
]

memory_before_values = [
    float(row["memory_before_gb"])
    for row in benchmark_rows
]

memory_after_values = [
    float(row["memory_after_gb"])
    for row in benchmark_rows
]


avg_ttft = statistics.mean(ttft_values)
avg_latency = statistics.mean(latency_values)
avg_tokens_per_second = statistics.mean(
    tokens_per_second_values
)

ttft_std = statistics.stdev(ttft_values)
latency_std = statistics.stdev(latency_values)
tokens_std = statistics.stdev(
    tokens_per_second_values
)

avg_memory_before = statistics.mean(
    memory_before_values
)

avg_memory_after = statistics.mean(
    memory_after_values
)


# ---------------------------------------------------------
# READ QUALITY RESULTS
# ---------------------------------------------------------

with open(
    SCORED_FILE,
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    scored_rows = list(reader)


total_prompts = len(scored_rows)

correct = sum(
    int(row["score"])
    for row in scored_rows
)

incorrect = total_prompts - correct

accuracy = (
    correct / total_prompts * 100
)


# ---------------------------------------------------------
# CATEGORY RESULTS
# ---------------------------------------------------------

categories = {}

for row in scored_rows:

    category = row["category"]

    if category not in categories:

        categories[category] = {
            "total": 0,
            "correct": 0
        }

    categories[category]["total"] += 1

    categories[category]["correct"] += int(
        row["score"]
    )


# ---------------------------------------------------------
# CREATE REPORT
# ---------------------------------------------------------

report = []

report.append("# Qwen2.5 1.5B Baseline")
report.append("")

report.append("## Model")
report.append("")
report.append("- Model: Qwen2.5 1.5B")
report.append(f"- Evaluation prompts: {total_prompts}")
report.append(f"- Benchmark runs: {len(benchmark_rows)}")
report.append("- Temperature: 0")
report.append("- Maximum output tokens: 100")
report.append("- Hardware: AMD Athlon Silver 3050U, 8 GB RAM")
report.append("")

report.append("## Performance Results")
report.append("")

report.append(
    "| Metric | Average | Standard Deviation |"
)

report.append(
    "|---|---:|---:|"
)

report.append(
    f"| TTFT | {avg_ttft:.2f} sec | "
    f"{ttft_std:.2f} sec |"
)

report.append(
    f"| Total Latency | {avg_latency:.2f} sec | "
    f"{latency_std:.2f} sec |"
)

report.append(
    f"| Tokens/sec | {avg_tokens_per_second:.2f} | "
    f"{tokens_std:.2f} |"
)

report.append("")

report.append("### System Memory Snapshot")
report.append("")

report.append(
    f"- Average memory before inference: "
    f"{avg_memory_before:.2f} GB"
)

report.append(
    f"- Average memory after inference: "
    f"{avg_memory_after:.2f} GB"
)

report.append("")

report.append(
    "> Note: These are system-wide memory snapshots "
    "from psutil, not direct measurements of model "
    "memory usage."
)

report.append("")

report.append("## Quality Evaluation")
report.append("")

report.append(
    f"- Total prompts: {total_prompts}"
)

report.append(
    f"- Correct: {correct}"
)

report.append(
    f"- Incorrect: {incorrect}"
)

report.append(
    f"- Overall accuracy: {accuracy:.2f}%"
)

report.append("")

report.append("### Results by Category")
report.append("")

report.append("| Category | Score | Accuracy |")
report.append("|---|---:|---:|")

for category, data in categories.items():

    category_accuracy = (
        data["correct"]
        / data["total"]
        * 100
    )

    report.append(
        f"| {category} | "
        f"{data['correct']}/{data['total']} | "
        f"{category_accuracy:.2f}% |"
    )

report.append("")

report.append("## Baseline Conclusion")
report.append("")

report.append(
    f"Qwen2.5 1.5B achieved an overall evaluation "
    f"accuracy of {accuracy:.2f}%."
)

report.append("")

report.append(
    f"The model generated an average of "
    f"{avg_tokens_per_second:.2f} tokens/sec "
    f"with an average total latency of "
    f"{avg_latency:.2f} seconds across "
    f"{len(benchmark_rows)} benchmark runs."
)

report.append("")


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(report))


print("========================================")
print("Baseline report created!")
print("========================================")
print(f"Saved to: {OUTPUT_FILE}")
