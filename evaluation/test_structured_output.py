import csv
import json
import os

from ollama import chat
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL = "qwen2.5:1.5b-instruct-q3_K_M"

PROMPTS_FILE = "evaluation/structured_prompts.csv"
RESULTS_FILE = "evaluation/structured_results.csv"


# ---------------------------------------------------------
# PYDANTIC SCHEMA
# ---------------------------------------------------------

class Person(BaseModel):
    name: str
    age: int
    country: str


# ---------------------------------------------------------
# VALIDATE MODEL RESPONSE
# ---------------------------------------------------------

def validate_response(raw_response):

    try:
        data = json.loads(raw_response)
        person = Person.model_validate(data)
        return True, person

    except (json.JSONDecodeError, ValidationError):
        return False, None


# ---------------------------------------------------------
# RUN ONE PROMPT
# ---------------------------------------------------------

def process_prompt(prompt):

    base_prompt = f"""
Extract the person's information from the text below.

Return ONLY valid JSON with exactly these fields:
- name
- age
- country

Text:
{prompt}
"""

    # -----------------------------------------------------
    # FIRST ATTEMPT
    # -----------------------------------------------------

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": base_prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    raw_response = response.message.content

    valid, person = validate_response(raw_response)

    if valid:
        return "first_attempt", raw_response

    # -----------------------------------------------------
    # RETRY
    # -----------------------------------------------------

    retry_prompt = f"""
The previous response was invalid.

Extract the person's information from this text:

{prompt}

Return ONLY valid JSON in exactly this format:

{{
    "name": "string",
    "age": 0,
    "country": "string"
}}

Do not include any explanation.
"""

    retry_response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": retry_prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    retry_raw_response = retry_response.message.content

    valid, person = validate_response(retry_raw_response)

    if valid:
        return "retry", retry_raw_response

    return "failed", retry_raw_response


# ---------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------

def main():

    results = []

    first_attempt_count = 0
    retry_count = 0
    failed_count = 0

    with open(
        PROMPTS_FILE,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    total = len(rows)

    print("\n========================================")
    print("STRUCTURED OUTPUT RELIABILITY TEST")
    print("========================================")
    print(f"Model: {MODEL}")
    print(f"Total prompts: {total}")
    print()

    for index, row in enumerate(rows, start=1):

        status, response = process_prompt(row["prompt"])

        if status == "first_attempt":
            first_attempt_count += 1

        elif status == "retry":
            retry_count += 1

        else:
            failed_count += 1

        results.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "status": status,
            "model_response": response
        })

        print(
            f"Completed {index}/{total} - {status}"
        )

    # -----------------------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------------------

    successful_count = (
        first_attempt_count + retry_count
    )

    reliability = (
        successful_count / total * 100
        if total > 0
        else 0
    )

    first_attempt_rate = (
        first_attempt_count / total * 100
        if total > 0
        else 0
    )

    retry_recovery_rate = (
        retry_count / total * 100
        if total > 0
        else 0
    )

    failure_rate = (
        failed_count / total * 100
        if total > 0
        else 0
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    os.makedirs(
        "evaluation",
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
                "prompt",
                "status",
                "model_response"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # PRINT FINAL RESULTS
    # -----------------------------------------------------

    print("\n========================================")
    print("RELIABILITY RESULTS")
    print("========================================")

    print(
        f"First attempt success: "
        f"{first_attempt_count}/{total} "
        f"({first_attempt_rate:.2f}%)"
    )

    print(
        f"Recovered after retry: "
        f"{retry_count}/{total} "
        f"({retry_recovery_rate:.2f}%)"
    )

    print(
        f"Failed after retry: "
        f"{failed_count}/{total} "
        f"({failure_rate:.2f}%)"
    )

    print(
        f"\nFinal reliability: "
        f"{successful_count}/{total} "
        f"({reliability:.2f}%)"
    )

    print("\nResults saved to:")
    print(RESULTS_FILE)

    print("========================================")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()