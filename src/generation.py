import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


# Initialize Groq client once
client = Groq(api_key=api_key)

def rewrite_query(
    query,
    conversation_prompt
):
    """
    Reformulate a conversational query into a
    standalone search query.

    Args:
        query (str):
            Current user question.

        conversation_prompt (str):
            Prompt containing recent conversation.

    Returns:
        str:
            Standalone search query.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": conversation_prompt
            }
        ],
        temperature=0,
        max_tokens=100
    )

    rewritten_query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return rewritten_query


def generate_answer(prompt):
    """
    Generate an answer using the Groq LLM.

    Args:
        prompt (str): Complete RAG prompt.

    Returns:
        str: Generated answer.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()