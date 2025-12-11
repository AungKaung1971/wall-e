# voice/conversation.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are WALL-E, a friendly robot assistant.
Keep responses short, helpful, and conversational.
If the user asks about objects or surroundings,
assume the answer will come from the robot’s vision system.
"""


def ask_llm(user_text: str) -> str:
    """
    Sends the user text to GPT-5-mini and returns the assistant's reply.
    Uses the SAME pattern as your working tweet generator.
    """
    print("\n[LLM] Sending to gpt-5-mini:", user_text)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )

        # SAME ACCESS PATTERN AS YOUR WORKING CODE
        reply = response.choices[0].message.content

        print("[LLM] Reply:", reply)
        return reply or "(empty response)"

    except Exception as e:
        print("[LLM] Error:", e)
        return "OpenAI request failed."


# python voice/conversation.py
