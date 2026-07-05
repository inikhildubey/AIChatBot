import json
import ollama

from agents.tool_registry import TOOLS
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

def decide_action(query: str) -> dict:
    tool_text = ""
    for tool in TOOLS:
        tool_text += f"""
        Tool: {tool["name"]}
        
        Description:
        {tool["description"]}
"""

    prompt = f"""
        You are the Planner of an AI Agent.
        Your ONLY responsibility is to decide which tool should execute the user's request.
        
        You MUST NOT:
        - Answer the user's question.
        - Rewrite the user's query.
        - Generate mathematical expressions.
        - Summarize documents.
        - Execute calculations.
        - Explain your reasoning.
        
        Available Tools:
        
        {tool_text}
        
        Instructions:
        1. Read the user's request.
        2. Choose the SINGLE best tool.
        3. Return ONLY valid JSON.
        4. Never return explanations.
        5. Never return markdown.
        
        If the request is only a greeting, return:
        
        {{
            "action":"direct_answer",
            "answer":"Hello! How can I help you?"
        }}
        
        Otherwise return:
        
        {{
            "action":"tool",
            "tool":"<tool_name>",
            "query":"{query}"
        }}
        
        User Request:
        
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
    decision = decide_action(query)

    print(decision)

    if decision["action"] == "direct_answer":
        return {
            "answer": decision["answer"]
        }

    if decision["action"] == "tool":
        tool_handler = None
        for tool in TOOLS:
            if tool['name'] == decision["tool"]:
                tool_handler = tool['handler']

        if tool_handler is None:
            return {
                "error": f"Unknown tool: {decision['tool']}"
            }
        result = await tool_handler(decision)
        return result

