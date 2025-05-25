import os
import json
import google.generativeai as genai

# Configuring the API
client = genai.configure(api_key=os.getenv("AIzaSyAF9ytmiAj1qtJaA7zmeKvZXrg5FiDrOo4"))

# Generate prompt text for use in large language models
def build_prompt(species: str, venomous: bool) -> str:
    # Spell prompt according to snake species and toxicity
    header = "You are an expert in snake bite first aid\n"
    header += f"-Snake species：{species}\n"
    header += f"- Is it toxic：{venomous}\n\n"
    header += "Please output in JSON format：{\"species\":\"...\",\"venomous\":true|false,\"first_aid\":[\"Step1\",\"Step2\",...]}"
    return header

# Calling Gemini Models to Generate Content
def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    print("[DEBUG] prompt >>>", prompt)
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    text = response.text.strip()

    # Remove the leading ```json
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    print("[DEBUG] cleaned response <<<", text)
    return text


