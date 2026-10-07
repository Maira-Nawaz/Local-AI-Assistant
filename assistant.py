from ollama import chat

print("Local AI Assistant")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    print(f"Assistant: {response.message.content}\n")