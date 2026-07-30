# agents/agent_loop.py

import asyncio
import json
import ollama

from agents.tool_registry import TOOLS
from memory.memory_service import get_planning_context
from memory.redis_memory import save_turn
from models.planner import PlannerResponse
from models.reflection import ReflectionStatus
from models.task import Task
from services.reflection_service import reflect

MAX_RETRIES = 3


def decide_action(query: str, conversation_history: list) -> PlannerResponse:
    tool_text = ""
    for tool in TOOLS:
        tool_text += f"""
        Tool: {tool}
        
        Description:
        {TOOLS[tool]["description"]}
    """
    prompt = f"""
    You are the Planner of an AI Agent.

    Your ONLY responsibility is to create an execution plan.

    You MUST NOT:
    - Answer the user's question.
    - Solve mathematical expressions.
    - Search documents.
    - Retrieve system information.
    - Explain your reasoning.
    - Return markdown.

    Available Tools:

    {tool_text}

    Planning Responsibilities:

    1. Read the current user request carefully.
    2. Use the previous conversation to understand the user's intent.
    3. If the current request contains references to previous messages, resolve those references.
    4. Rewrite incomplete or conversational requests into complete standalone queries.
    5. Break the request into one or more independent tasks.
    6. Select the most appropriate tool for each task.
    7. Preserve the user's original intent.
    8. Return ONLY valid JSON.

    Reference Resolution Rules:

    The current request may contain references such as:

    - it
    - this
    - that
    - these
    - those
    - they
    - them
    - previous answer
    - full form
    - values
    - compare
    - difference
    - explain it
    - explain this
    - summarize it
    - tell me more

    When such references exist:

    - Use the previous conversation to determine what the user is referring to.
    - Rewrite the task query into a complete standalone query.
    - Never leave ambiguous words like "it", "this", or "that" inside the generated task query if they can be resolved.

    Examples

    Example 1

    Previous Conversation

    User:
    What is RTGS?

    Assistant:
    RTGS stands for Real Time Gross Settlement.

    Current User Request

    What is the full form?

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "search_banking_docs",
                "query": "What is the full form of RTGS?"
            }}
        ]
    }}

    --------------------------------------------------

    Example 2

    Previous Conversation

    User:
    What is RTGS?

    Assistant:
    RTGS stands for Real Time Gross Settlement.

    Current User Request

    this vs NEFT

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "search_banking_docs",
                "query": "Compare RTGS and NEFT."
            }}
        ]
    }}

    --------------------------------------------------

    Example 3

    Previous Conversation

    User:
    26 + 97

    Assistant:
    123

    Current User Request

    What are the values?

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "calculator",
                "query": "What are the values in the expression 26 + 97?"
            }}
        ]
    }}

    --------------------------------------------------

    Example 4

    Current User Request

    What is CRR?

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "search_banking_docs",
                "query": "What is CRR?"
            }}
        ]
    }}

    --------------------------------------------------

    Example 5

    Current User Request

    Multiply twenty five by thirty

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "calculator",
                "query": "Multiply twenty five by thirty"
            }}
        ]
    }}

    --------------------------------------------------

    Example 6

    Current User Request

    What time is it?

    Expected Output

    {{
        "tasks": [
            {{
                "tool": "system_tool",
                "query": "What time is it?"
            }}
        ]
    }}

    Output Format

    Always return ONLY a valid JSON object.

    Example

    {{
        "tasks": [
            {{
                "tool": "search_banking_docs",
                "query": "What is CRR?"
            }}
        ]
    }}

    Previous Conversation:

    {conversation_history}

    Current User Request:

    {query}
    """
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    content = response["message"]["content"]

    try:
        decision_json = json.loads(content)
        decision = PlannerResponse(**decision_json)
        print("=" * 60)
        print("Planner Query    :", query)
        print("Planner Decision :", decision)
        print("=" * 60)
        return decision

    except json.JSONDecodeError:
        print("=" * 60)
        print("Planner returned invalid JSON")
        print(content)
        print("=" * 60)
        raise ValueError("Planner returned invalid JSON.")


# async def run_agent(query):
#     turn = {"user": query}
#
#     planning_context = get_planning_context(query)
#
#     decision = decide_action(
#         query=query,
#         conversation_history=planning_context
#     )
#
#     if not decision.tasks:
#         return {
#             "error": "Planner returned no tasks."
#         }
#
#     retry_count = 0
#     current_tasks = decision.tasks
#     MAX_RETRIES = 3
#
#     while True:
#
#         jobs = []
#
#         for task in current_tasks:
#
#             tool_info = TOOLS.get(task.tool)
#
#             if tool_info is None:
#                 return {
#                     "error": f"Unknown tool: {task.tool}"
#                 }
#
#             handler = tool_info["handler"]
#
#             if handler is None:
#                 return {
#                     "error": f"Handler not available for tool: {task.tool}"
#                 }
#
#             jobs.append(
#                 handler(task.model_dump())
#             )
#
#         results = await asyncio.gather(*jobs)
#
#         reflections = []
#         for task, result in zip(current_tasks, results):
#             reflection = reflect(
#                 task=task,
#                 result=result
#             )
#             reflections.append(reflection)
#
#         print("=" * 60)
#         print("Reflection:", reflections)
#         print("=" * 60)
#         full_result = combine_results(reflections)
#
#         if reflection.status == ReflectionStatus.COMPLETE:
#             turn["assistant"] = full_result["answer"]
#             save_turn(turn)
#             return full_result
#
#         elif reflection.status == ReflectionStatus.WAITING:
#             turn["assistant"] = reflection.response
#             save_turn(turn)
#             return {
#                 "answer": reflection.response
#             }
#
#         elif reflection.status == ReflectionStatus.RETRY:
#             if retry_count >= MAX_RETRIES:
#                 return {"error": "Maximum retry attempts reached."}
#             if reflection.next_task is None:
#                 return {"error": "Reflection requested retry but did not provide next_task."}
#             retry_count += 1
#             reflection.next_task.retry_count = retry_count
#             current_tasks = [reflection.next_task]
#
#         elif reflection.status == ReflectionStatus.ACTION_NEEDED:
#             if reflection.next_task is None:
#                 return {"error": "Reflection requested another action but did not provide next_task."}
#             current_tasks = [reflection.next_task]
#
#         else:
#             return {"error": f"Unknown reflection status: {reflection.status}"}


async def execute_tool(task: Task):
    tool_info = TOOLS.get(task.tool)

    if tool_info is None:
        return {
            "error": f"Unknown tool: {task.tool}"
        }

    handler = tool_info.get("handler")

    if handler is None:
        return {
            "error": f"Handler not available for tool: {task.tool}"
        }

    return await handler(task.model_dump())


async def process_task(task: Task):
    while True:

        result = await execute_tool(task)

        reflection = reflect(
            task=task,
            result=result
        )

        if reflection.status == ReflectionStatus.COMPLETE:
            return result

        elif reflection.status == ReflectionStatus.WAITING:
            return {
                "answer": reflection.response
            }

        elif reflection.status == ReflectionStatus.RETRY:

            if task.retry_count >= MAX_RETRIES:
                return {
                    "error": "Maximum retry attempts reached."
                }

            if reflection.next_task is None:
                return {
                    "error": "Reflection requested retry but did not provide next_task."
                }

            reflection.next_task.retry_count = task.retry_count + 1
            task = reflection.next_task
            continue

        elif reflection.status == ReflectionStatus.ACTION_NEEDED:

            if reflection.next_task is None:
                return {
                    "error": "Reflection requested another action but did not provide next_task."
                }

            task = reflection.next_task
            continue

        return {
            "error": f"Unknown reflection status: {reflection.status}"
        }


async def run_agent(query):
    turn = {"user": query}

    planning_context = get_planning_context(query)

    decision = decide_action(
        query=query,
        conversation_history=planning_context
    )

    if not decision.tasks:
        return {
            "error": "Planner returned no tasks."
        }

    results = await asyncio.gather(
        *[
            process_task(task)
            for task in decision.tasks
        ]
    )

    full_result = combine_results(results)

    turn["assistant"] = full_result["answer"]
    save_turn(turn)

    return full_result


def combine_results(results):
    if len(results) == 1:
        return results[0]

    answers = []

    for result in results:
        answers.append(result["answer"])
    return {
        "answer": " And ".join(str(x) for x in answers)
    }
