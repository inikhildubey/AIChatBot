import json
import ollama

from agents.tool_registry import TOOLS
from tools.calculator_tool import calculate
from tools.rag_tool import ask_question


def decide_action(query):
    tool_text = ""

    for tool in TOOLS:
        tool_text += f"""
        Tool: {tool['name']}
        Description: {tool['description']}
        """

    prompt = f"""
                You are an AI Agent.
                Available Tools:
                {tool_text}
                
                Rules:
                1. Use a tool whenever a relevant tool exists.
                2. Do not answer banking questions directly.
                3. Greetings can be answered directly.
                4. Return ONLY valid JSON.
                
                Examples:
                User: What is CRR?
                
                {{
                  "action":"tool",
                  "tool":"search_banking_docs",
                  "query":"What is CRR?"
                }}
                
                User: Hi
                
                {{
                  "action":"direct_answer",
                  "answer":"Hello! How can I help you?"
                }}
                
                User: What is 2 + 2?
                
                {{
                  "action":"tool",
                  "tool":"calculator",
                  "expression":"2+2"
                }}
                
                User: Calculate 25 * 30
                
                {{
                  "action":"tool",
                  "tool":"calculator",
                  "expression":"25*30"
                }}
                Rules:

                - search_banking_docs must use field "query"
                - calculator must use field "expression"
                - Preserve mathematical operators (+, -, *, /) in calculator expressions.
                User Question:
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

    # return json.loads(
    #     response["message"]["content"]
    # )
    content = response["message"]["content"]
    try:
        decision = json.loads(content)
        return decision
    except Exception as e:
        print("JSON Parse Error:", e)
        print(content)
        return {
            "action": "direct_answer",
            "answer": content
        }


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

