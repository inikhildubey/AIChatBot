import asyncio
import json
import ollama

from agents.tool_registry import TOOLS
from memory.memory_service import *
from memory.redis_memory import save_turn
from tools.calculator_tool import calculate
from tools.rag_tool import ask_question


# def decide_action(query):
#     tool_text = ""
#
#     for tool in TOOLS:
#         tool_text += f"""
#         Tool: {tool['name']}
#         Description: {tool['description']}
#         """
#     print("Tool text:-", tool_text)
#     prompt = f"""
#                     You are a JSON routing engine.
#                     Your ONLY responsibility is to classify the user's request.
#
#                     You are NOT an assistant.
#                     You are NOT allowed to answer the user's question.
#
#                     You must decide whether to:
#
#                     1. Return a direct greeting.
#                     2. Use one tool.
#                     3. Use multiple tools.
#                     4. Check if tools are not available to solve the problem and if question is related to
#                     greetings then only you will be allowed to response by yourself.
#
#                     Return exactly ONE valid JSON object.
#
#                     Rules:
#                     - Never explain your reasoning.
#                     - Never add text before or after the JSON.
#                     - Never use markdown.
#                     - Your response must start with '{' and end with '}'.
#
#
#                     Output format along with Tools Examples:
#                     === Banking Tool ===
#                     User: What is CRR?
#
#                     {{
#                       "action":"tool",
#                       "tool":"search_banking_docs",
#                       "query":"What is CRR?"
#                     }}
#
#                     === Calculator Tool ===
#
#                     Purpose:
#                     Convert mathematical questions into valid Python mathematical expressions.
#
#                     Rules:
#
#                     - Return ONLY a valid Python mathematical expression.
#                     - Use:
#                       + - * / % ** // ()
#                     - Convert words into numbers.
#                     - Preserve operator precedence.
#                     - Do not explain the calculation.
#
#                     User: What is 2 + 2?
#                     User: What is the sum of 2 and 2?
#                     User: Add 2 and 2.
#                     User: Addition 2 and 2.
#
#                     {{
#                       "action":"tool",
#                       "tool":"calculator",
#                       "expression":"2+2"
#                     }}
#
#
#                     User: Calculate 25 * 30
#                     User: Multiply twenty five by thirty.
#                     User: What is twenty five times thirty?
#                     User: Product of 25 and 30.
#                     User: 25 multiplied by 30.
#                     User: Find multiplication of twenty five and thirty.
#
#
#                     {{
#                       "action":"tool",
#                       "tool":"calculator",
#                       "expression":"25*30"
#                     }}
#
#                     User: What is (3+((25+5)*3)) ?
#
#                     {{
#                         "action":"tool",
#                         "tool":"calculator",
#                         "expression":"(3+((25+5)*3))"
#                     }}
#
#                     User: What is the remainder when 24 is divided by 5?
#
#                     {{
#                         "action":"tool",
#                         "tool":"calculator",
#                         "expression":"24%5"
#                     }}
#
#                     User: What is Square of 25.
#
#                     {{
#                         "action":"tool",
#                         "tool":"calculator",
#                         "expression":""25**2""
#                     }}
#
#                     User: Subtract twenty from one hundred and divide by four.
#
#                     {{
#                         "action":"tool",
#                         "tool":"calculator",
#                         "expression":"(100-20)/4"
#                     }}
#
#                     === System Tool ===
#                     User:
#                     What time is it?
#
#                     {{
#                         "action":"tool",
#                         "tool":"system_tool",
#                         "query":"What time is it?"
#                     }}
#
#                     User:
#                     What day is today?
#
#                     {{
#                             "action": "tool",
#                             "tool": "system_tool",
#                             "query": "What day is today?"
#                     }}
#
#                     === Greeting ===
#
#                     Purpose:
#                     Identify greetings.
#
#                     Rules:
#                     - Greetings MUST NOT use any tool.
#                     - Return action="direct_answer".
#                     - Do not classify greetings as calculator, banking, or system_tool.
#
#                     Examples:
#
#                     Hi
#                     Hello
#                     Hey
#                     Hello!
#                     Hello?
#                     Hi?
#                     Good Morning
#                     Good Evening
#                     How are you?
#
#                     {{
#                         "action":"direct_answer",
#                         "answer":"Hello! How can I help you?"
#                     }}
#
#
#                     User Question:
#                     {query}
#                     """
#     print("Query:-",query)
#     response = ollama.chat(
#         model="llama3",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         options={
#             "temperature": 0
#         }
#     )
#     content = response["message"]["content"]
#     try:
#         decision = json.loads(content)
#         return decision
#     except Exception as e:
#         print("JSON Parse Error:", e)
#         print(content)
#         return {
#             "action": "direct_answer",
#             "answer": content
#         }

def decide_action(query: str, conversation_history: list) -> dict:
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
        decision = json.loads(content)

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


async def run_agent(query):
    turn = {"user": query}
    planning_context = get_planning_context(query)

    decision = decide_action(
        query=query,
        conversation_history=planning_context
    )
    print(decision)
    # if decision["action"] == "direct_answer":
    #     return {
    #         "answer": decision["answer"]
    #     }
    # if decision["action"] == "tool":
    #     tool_handler = None
    #     for tool in TOOLS:
    #         if tool['name'] == decision["tool"]:
    #             tool_handler = tool['handler']
    #
    #     if tool_handler is None:
    #         return {
    #             "error": f"Unknown tool: {decision['tool']}"
    #         }
    #     result = await tool_handler(decision)
    #     return result
    if not decision['tasks']:
        return {
            "error": f"Tool not available or Unknown tool: {decision['tool']}"
        }
    jobs = []
    for task in decision['tasks']:
        if not task['tool']:
            return "Unknown tool"
        # handler = None
        # import pdb
        # pdb.set_trace()
        # for tool in TOOLS:
        #     if tool == task["tool"]:
        #         handler = TOOLS[tool]['handler']
        #         jobs.append(handler(task))
        #         break
        tool = task["tool"]
        tool_info = TOOLS.get(tool)
        if tool_info is None:
            return {
                "error": f"Unknown tool: {tool}"
            }
        handler = tool_info['handler']
        jobs.append(handler(task))

        if handler is None:
            return {
                "error": f"Unknown tool: {task['tool']}"
            }

    results = await asyncio.gather(*jobs)
    full_result = combine_results(results)
    turn['assistant'] = full_result["answer"]
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
