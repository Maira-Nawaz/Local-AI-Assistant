import csv
import os
import sys

from ollama import chat


PROMPTS_FILE = "evaluation/prompts.csv"


# ---------------------------------------------------------
# GET MODEL FROM COMMAND LINE
# ---------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python evaluation/evaluate.py <model_name>")
    print()
    print("Examples:")
    print("python evaluation/evaluate.py qwen2.5:1.5b")
    print("python evaluation/evaluate.py phi3.5")
    print("python evaluation/evaluate.py llama3.2:1b")
    sys.exit(1)


MODEL = sys.argv[1]


# ---------------------------------------------------------
# CREATE SAFE FILE NAME
# ---------------------------------------------------------
# Keep Qwen filename consistent with the existing project
# naming convention.

if MODEL == "qwen2.5:1.5b":
    MODEL_SAFE_NAME = "qwen25_1.5b"
else:
    MODEL_SAFE_NAME = (
        MODEL
        .replace(":", "_")
        .replace("/", "_")
    )


RESULTS_FILE = (
    f"evaluation/results/{MODEL_SAFE_NAME}_results.csv"
)


# ---------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------

with open(
    PROMPTS_FILE,
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    results = []

    for row in reader:

        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": row["prompt"]
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 100
            }
        )

        model_response = response.message.content

        results.append({
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "expected_answer": row["expected_answer"],
            "model_response": model_response
        })

        print(
            f"Completed prompt {row['id']}/40"
        )


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

os.makedirs(
    "evaluation/results",
    exist_ok=True
)


with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "id",
            "category",
            "prompt",
            "expected_answer",
            "model_response"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("\n========================================")
print("Evaluation completed!")
print(f"Model: {MODEL}")
print(f"Results saved to: {RESULTS_FILE}")
print("========================================")