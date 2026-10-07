import csv
import json
import os
import re
import sys


# ============================================================
# MODEL ARGUMENT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python evaluation/score_results.py <model_name>")
    print()
    print("Examples:")
    print("python evaluation/score_results.py qwen2.5:1.5b")
    print("python evaluation/score_results.py phi3.5")
    print("python evaluation/score_results.py llama3.2:1b")
    sys.exit(1)


MODEL = sys.argv[1]

if MODEL == "qwen2.5:1.5b":
    MODEL_SAFE_NAME = "qwen25_1.5b"
else:
    MODEL_SAFE_NAME = (
        MODEL
        .replace(":", "_")
        .replace("/", "_")
    )

RESULTS_FILE = f"evaluation/results/{MODEL_SAFE_NAME}_results.csv"
SCORED_FILE = f"evaluation/results/{MODEL_SAFE_NAME}_scored.csv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize text for semantic-ish comparisons.
    """
    text = text.lower()

    # Normalize common punctuation
    text = re.sub(r"[*_`#]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_all(response, terms):
    """
    Check whether all required concepts appear in the response.
    """
    response = normalize(response)

    return all(
        normalize(term) in response
        for term in terms
    )


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(response):
    """
    Extract JSON from:
    - raw JSON
    - ```json ... ```
    - ``` ... ```
    """

    text = response.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    # Try entire response first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find an object inside surrounding text
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_date(value):
    """
    Treat common representations of the same date as equivalent.
    """

    value = str(value).strip().lower()

    date_map = {
        "october 15, 2026": "2026-10-15",
        "october 20, 2026": "2026-10-20",
        "oct 15, 2026": "2026-10-15",
        "oct 20, 2026": "2026-10-20",
    }

    return date_map.get(value, value)


# ============================================================
# JSON VALUE NORMALIZATION
# ============================================================

def normalize_json_value(key, value):

    if isinstance(value, str):

        value = value.strip().lower()

        # Normalize singular/plural where appropriate
        if key == "product":
            if value == "laptop":
                value = "laptops"

        # Normalize sentiment
        if key == "sentiment":
            value = value.lower()

        # Normalize category
        if key == "category":
            if value == "product review":
                value = "product"

        # Normalize dates
        if key == "date":
            value = normalize_date(value)

        return value

    return value


# ============================================================
# STRUCTURED OUTPUT SCORING
# ============================================================

def score_structured_output(expected, response):

    actual_json = extract_json(response)

    if actual_json is None:
        return False

    # Expected answers are sometimes plain text descriptions
    # instead of actual JSON.
    expected_pairs = {}

    expected_lower = expected.lower()

    patterns = {
        "name": r"name\s*=\s*([^,]+)",
        "age": r"age\s*=\s*([^,]+)",
        "city": r"city\s*=\s*([^,]+)",
        "product": r"product\s*=\s*([^,]+)",
        "quantity": r"quantity\s*=\s*([^,]+)",
        "price": r"price\s*=\s*([^,]+)",
        "category": r"category\s*=\s*([^,]+)",
        "sentiment": r"sentiment\s*=\s*([^,]+)",
        "task": r"task\s*=\s*([^,]+)",
        "priority": r"priority\s*=\s*([^,]+)",
        "completed": r"completed\s*=\s*([^,]+)",
        "date": r"date\s*=\s*([^,]+)",
        "location": r"location\s*=\s*([^,]+)",
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            expected_lower
        )

        if match:
            expected_pairs[key] = match.group(1).strip()

    # If no pairs could be extracted, fail safely.
    if not expected_pairs:
        return False

    # Compare required fields
    for key, expected_value in expected_pairs.items():

        # Field must exist
        if key not in actual_json:
            return False

        actual_value = normalize_json_value(
            key,
            actual_json[key]
        )

        expected_value = normalize_json_value(
            key,
            expected_value
        )

        # Numeric comparison
        if key in ["age", "quantity", "price"]:

            try:
                if float(actual_value) != float(expected_value):
                    return False

            except Exception:
                return False

        else:

            if str(actual_value) != str(expected_value):
                return False

    return True


# ============================================================
# CATEGORY SCORING
# ============================================================

def score_response(
    category,
    prompt,
    expected,
    response
):

    response_normalized = normalize(response)
    expected_normalized = normalize(expected)


    # --------------------------------------------------------
    # KNOWLEDGE
    # --------------------------------------------------------

    if category == "Knowledge":

        # Prompt-specific concepts
        if "database" in prompt.lower():

            return contains_all(
                response,
                [
                    "store",
                    "manage",
                    "data"
                ]
            )

        if "ram" in prompt.lower():

            return (
                "volatile" in response_normalized
                and (
                    "temporary" in response_normalized
                    or "lost" in response_normalized
                )
                and (
                    "permanent storage" in response_normalized
                    or "ssd" in response_normalized
                    or "hard drive" in response_normalized
                )
            )

        return contains_all(
            response,
            expected.split("|")
        )


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if category == "Explanation":

        prompt_lower = prompt.lower()

        if "api" in prompt_lower:

            return (
                "api" in response_normalized
                and (
                    "communicat" in response_normalized
                    or "talk" in response_normalized
                )
                and (
                    "application" in response_normalized
                    or "software" in response_normalized
                    or "system" in response_normalized
                )
            )

        if "data warehouse" in prompt_lower:

            return (
                "data warehouse" in response_normalized
                and (
                    "central" in response_normalized
                    or "repository" in response_normalized
                    or "store" in response_normalized
                )
                and (
                    "analy" in response_normalized
                    or "report" in response_normalized
                )
            )

        if "ai" in prompt_lower and "machine learning" in prompt_lower:

            return (
                "artificial intelligence" in response_normalized
                and "machine learning" in response_normalized
                and (
                    "broader" in response_normalized
                    or "subset" in response_normalized
                )
            )

        if "etl" in prompt_lower:

            return all(
                term in response_normalized
                for term in [
                    "extract",
                    "transform",
                    "load"
                ]
            )

        if "primary key" in prompt_lower:

            return (
                "primary key" in response_normalized
                and "unique" in response_normalized
                and (
                    "record" in response_normalized
                    or "row" in response_normalized
                )
            )

        return contains_all(
            response,
            expected.split("|")
        )


    # --------------------------------------------------------
    # SUMMARIZATION
    # --------------------------------------------------------

    if category == "Summarization":

        prompt_lower = prompt.lower()

        if "data pipeline" in prompt_lower:

            return (
                "collect" in response_normalized
                and (
                    "transform" in response_normalized
                    or "process" in response_normalized
                )
                and (
                    "deliver" in response_normalized
                    or "send" in response_normalized
                )
            )

        if "machine learning" in prompt_lower:

            return (
                "historical data" in response_normalized
                and "pattern" in response_normalized
                and (
                    "predict" in response_normalized
                    or "prediction" in response_normalized
                )
                and (
                    "training data" in response_normalized
                    or "quality" in response_normalized
                )
            )

        if "power bi" in prompt_lower:

            return (
                "power bi" in response_normalized
                and (
                    "connect" in response_normalized
                    or "data" in response_normalized
                )
                and (
                    "transform" in response_normalized
                    or "model" in response_normalized
                )
                and (
                    "dashboard" in response_normalized
                    or "report" in response_normalized
                )
            )

        if "cloud computing" in prompt_lower:

            return (
                "internet" in response_normalized
                and (
                    "scal" in response_normalized
                    or "flexible" in response_normalized
                )
                and (
                    "upfront" in response_normalized
                    or "hardware" in response_normalized
                )
            )

        if "data quality" in prompt_lower:

            return (
                "data quality" in response_normalized
                and (
                    "analysis" in response_normalized
                    or "accuracy" in response_normalized
                )
                and (
                    "decision" in response_normalized
                    or "business" in response_normalized
                )
            )

        return contains_all(
            response,
            expected.split("|")
        )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if category == "Classification":

        expected_class = expected_normalized

        # Look primarily at the beginning because models often
        # explain their classification afterward.
        beginning = response_normalized[:250]

        return expected_class in beginning


    # --------------------------------------------------------
    # INFORMATION EXTRACTION
    # --------------------------------------------------------

    if category == "Information Extraction":

        prompt_lower = prompt.lower()

        # Required values are more important than exact labels.
        # We extract the important values from the expected answer.

        expected_parts = [
            part.strip()
            for part in expected.split(";")
        ]

        for part in expected_parts:

            if ":" not in part:
                continue

            _, value = part.split(":", 1)

            value = normalize(value)

            if value not in response_normalized:
                return False

        return True


    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    if category == "Reasoning":

        # For the arithmetic reasoning task, accept the final
        # numerical answer if it appears in the response.

        if expected.strip().isdigit():

            return re.search(
                rf"\b{re.escape(expected.strip())}\b",
                response
            ) is not None

        return expected_normalized in response_normalized


    # --------------------------------------------------------
    # INSTRUCTION FOLLOWING
    # --------------------------------------------------------

    if category == "Instruction Following":

        expected_lower = expected.lower()
        prompt_lower = prompt.lower()


        # Exact sentence reproduction
        if expected_lower.strip() == (
            "the analyst prepared the report."
        ):

            return (
                "the analyst prepared the report."
                in response_normalized
            )


        # Exact phrase
        if expected_lower.strip() == (
            "to support better business decisions"
        ):

            return (
                "to support better business decisions"
                in response_normalized
            )


        # Exactly 6 words
        if "exactly 6 words" in expected_lower:

            first_line = response.strip().splitlines()[0]

            # Remove surrounding quotes
            first_line = first_line.strip('"').strip("'")

            return len(first_line.split()) == 6


        # Exactly 3 bullet points, <= 8 words each
        if "exactly 3 bullet points" in expected_lower:

            lines = [
                line.strip()
                for line in response.splitlines()
                if line.strip().startswith(("-", "•", "*"))
            ]

            if len(lines) != 3:
                return False

            for line in lines:

                cleaned = re.sub(
                    r"^[-•*]\s*",
                    "",
                    line
                )

                if len(cleaned.split()) > 8:
                    return False

            return True


        # Leave request
        if "leave request" in expected_lower:

            response_lower = response.lower()

            return (
                len(response.split()) <= 50
                and "urgent" not in response_lower
                and "best regards" in response_lower
            )


        return False


    # --------------------------------------------------------
    # STRUCTURED OUTPUT
    # --------------------------------------------------------

    if category == "Structured Output":

        return score_structured_output(
            expected,
            response
        )


    return False


# ============================================================
# LOAD RESULTS
# ============================================================

if not os.path.exists(RESULTS_FILE):

    print(f"Results file not found: {RESULTS_FILE}")

    sys.exit(1)


with open(
    RESULTS_FILE,
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    results = list(reader)


# ============================================================
# SCORE
# ============================================================

scored_results = []

correct = 0


for row in results:

    is_correct = score_response(
        row["category"],
        row["prompt"],
        row["expected_answer"],
        row["model_response"]
    )

    if is_correct:
        correct += 1

    scored_results.append({
        **row,
        "correct": is_correct
    })


# ============================================================
# SAVE
# ============================================================

with open(
    SCORED_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "id",
        "category",
        "prompt",
        "expected_answer",
        "model_response",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(scored_results)


# ============================================================
# CATEGORY SUMMARY
# ============================================================

categories = sorted(
    set(row["category"] for row in scored_results)
)


print()
print("========================================")
print("EVALUATION SCORE")
print("========================================")

print(f"Model: {MODEL}")
print(f"Total: {len(scored_results)}")
print(f"Correct: {correct}")
print(f"Incorrect: {len(scored_results) - correct}")
print(
    f"Accuracy: "
    f"{(correct / len(scored_results)) * 100:.2f}%"
)

print()
print("Category Results:")
print("----------------------------------------")

for category in categories:

    category_rows = [
        row
        for row in scored_results
        if row["category"] == category
    ]

    category_correct = sum(
        row["correct"]
        for row in category_rows
    )

    category_total = len(category_rows)

    category_accuracy = (
        category_correct / category_total
    ) * 100

    print(
        f"{category}: "
        f"{category_correct}/{category_total} "
        f"({category_accuracy:.2f}%)"
    )


print()
print(f"Scored results saved to:")
print(SCORED_FILE)

print("========================================")