import os
import json
import google.generativeai as genai

# Configuring the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# Generate prompt text for use in large language models
def build_prompt(species: str, venomous: bool) -> str:
    header = "You are an expert in snakebite first aid.\n"
    header += f"- Snake species: {species}\n"
    header += f"- Is it toxic: {venomous}\n\n"
    header += (
        "Please return a JSON object with the following format:\n"
        "{\n"
        "  \"species\": \"...\",\n"
        "  \"venomous\": true|false,\n"
        "  \"first_aid\": [\"Step 1\", \"Step 2\", ...],\n"
        "  \"additional_notes\": \"...\"\n"
        "}"
    )
    return header

# Calling Gemini Models to Generate Content
def call_gemini(prompt: str) -> dict:
    print("[DEBUG] prompt >>>", prompt)

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        print("[DEBUG] cleaned response <<<", text)

        return json.loads(text)
    except Exception as e:
        print("[ERROR] Gemini failed:", e)
        return None



