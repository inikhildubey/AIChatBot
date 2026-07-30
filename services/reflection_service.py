import json
import ollama

from pydantic import ValidationError

from models.reflection import ReflectionResponse
from models.task import Task

MAX_FORMAT_RETRIES = 2


def reflect(
    task: Task,
    result: dict
) -> ReflectionResponse:

    prompt = f"""
    You are the Reflection component of an AI Agent.

    Your ONLY responsibility is to evaluate whether the previous tool execution
    successfully satisfied the user's request.

    You MUST NOT:
    - Answer the user's question yourself.
    - Call tools.
    - Return markdown.
    - Return anything except valid JSON.

    task:
    {task}

    Tool Result:
    {result}

    Evaluate whether the tool result satisfies the user's request.

    Possible Statuses:

    1. COMPLETE

    Use when the tool result completely satisfies the user's request.

    Return:
    {{
        "status": "COMPLETE",
        "response": null,
        "reason": null,
        "next_task": null
    }}


    2. RETRY

    Use when:
    - The selected tool was appropriate.
    - The result is incorrect, incomplete, or insufficient.
    - Another attempt with an improved query may succeed.

    For RETRY:
    - Briefly explain the problem in "reason".
    - Create "next_task" using the same appropriate tool.
    - Rewrite the query so the next attempt has a better chance of succeeding.
    - Do NOT manage or increment retry_count.
      The application manages retry count.

    Return:
    {{
        "status": "RETRY",
        "response": null,
        "reason": "The result did not sufficiently answer the request.",
        "next_task": {{
            "tool": "search_banking_docs",
            "query": "Explain RTGS and its key characteristics."
        }}
    }}


    3. WAITING

    Use when the agent cannot continue until the user provides additional
    information or performs an action.

    For WAITING:
    - Put the message that should be shown to the user in "response".

    Return:
    {{
        "status": "WAITING",
        "response": "Please provide the required information.",
        "reason": "Additional information is required before execution can continue.",
        "next_task": null
    }}


    4. ACTION_NEEDED

    Use when the previous execution was successful, but the overall task
    requires another tool or another execution step.

    For ACTION_NEEDED:
    - Briefly explain what remains to be done in "reason".
    - Create "next_task" describing the next required action.

    Return:
    {{
        "status": "ACTION_NEEDED",
        "response": null,
        "reason": "Another execution step is required.",
        "next_task": {{
            "tool": "calculator",
            "query": "Perform the remaining calculation."
        }}
    }}


    Important:

    - COMPLETE means stop.
    - WAITING means stop and ask the user for the required information.
    - RETRY means the previous attempt was insufficient and should be attempted again.
    - ACTION_NEEDED means the previous attempt may have succeeded,
      but another step is required.
    - Return ONLY valid JSON.
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    for attempt in range(MAX_FORMAT_RETRIES + 1):

        response = ollama.chat(
            model="llama3",
            messages=messages,
            options={
                "temperature": 0
            }
        )

        content = response["message"]["content"]

        try:
            reflection_json = json.loads(content)

            return ReflectionResponse(
                **reflection_json
            )

        except (json.JSONDecodeError, ValidationError) as error:

            if attempt >= MAX_FORMAT_RETRIES:
                raise ValueError(
                    f"Reflection failed after "
                    f"{MAX_FORMAT_RETRIES} format retries.\n"
                    f"Last response:\n{content}"
                ) from error

            # Preserve the invalid response so the LLM
            # can see exactly what it needs to correct.
            messages.append({
                "role": "assistant",
                "content": content
            })

            messages.append({
                "role": "user",
                "content": f"""
                Your previous response violated the required JSON contract.

                Validation Error:
                {error}

                Correct your previous response.

                Return ONLY valid JSON.

                Do NOT include:
                - explanations
                - markdown
                - code fences
                - comments
                - text before the JSON
                - text after the JSON

                Return ONLY the JSON object.
                """
            })