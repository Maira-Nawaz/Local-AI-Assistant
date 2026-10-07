from ollama import chat
from pydantic import BaseModel, ValidationError
import json


# ---------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------

class Person(BaseModel):
    name: str
    age: int
    country: str


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL = "qwen2.5:1.5b"


# ---------------------------------------------------------
# GENERATE STRUCTURED OUTPUT
# ---------------------------------------------------------

def extract_person(user_input: str):

    prompt = f"""
Extract the person's information from the text below.

Return ONLY valid JSON.

Required fields:
- name
- age
- country

Text:
{user_input}
"""

    # First attempt
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    raw_response = response.message.content

    print("\nFirst response:")
    print(raw_response)

    # -----------------------------------------------------
    # VALIDATE FIRST RESPONSE
    # -----------------------------------------------------

    try:
        data = json.loads(raw_response)
        person = Person.model_validate(data)

        print("\n✓ First attempt valid!")
        return person

    except (json.JSONDecodeError, ValidationError) as error:

        print("\n✗ First attempt failed validation.")
        print("Retrying...")

        # -------------------------------------------------
        # RETRY
        # -------------------------------------------------

        retry_prompt = f"""
Your previous response was invalid.

Extract the person's information from this text:

{user_input}

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

        retry_raw = retry_response.message.content

        print("\nRetry response:")
        print(retry_raw)

        # -------------------------------------------------
        # VALIDATE RETRY
        # -------------------------------------------------

        try:
            retry_data = json.loads(retry_raw)
            person = Person.model_validate(retry_data)

            print("\n✓ Retry successful!")
            return person

        except (json.JSONDecodeError, ValidationError):

            print("\n✗ Retry failed.")
            print("Could not produce valid structured output.")

            return None


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    user_input = (
        "My name is Ali. I am 25 years old "
        "and I live in Pakistan."
    )

    result = extract_person(user_input)

    if result:
        print("\n================================")
        print("VALIDATED DATA")
        print("================================")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")
        print(f"Country: {result.country}")