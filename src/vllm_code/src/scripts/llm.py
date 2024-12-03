from openai import OpenAI

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key="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"
)

speech_text = "Place the red marker inside the blue box"

messages = [
    {
        "role": "user",
        "content": f"\"{speech_text}\". Give me only the objects and its properties mentioned in the previous sentence in correct order of appearance, separated by commas."
    }
]

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct", 
    messages=messages, 
    max_tokens=500
)

# Print the input message and the completion message content
input_message = messages[0]["content"]
output_content = completion.choices[0].message.content

print(f"Input: {input_message}")
print(f"Output: {output_content}")
