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

                    General Rules:
                    1. Use a tool whenever a relevant tool exists.
                    2. Do not answer banking questions directly.
                    3. Greetings can be answered directly.
                    4. Return ONLY valid JSON.
                    5. search_banking_docs must use field "query"
                    6. calculator must use field "expression"
                    7. Preserve mathematical operators (+, -, *, /) in calculator expressions.


                    Output format along with Tools Examples:
                    === Banking Tool ===
                    User: What is CRR?

                    {{
                      "action":"tool",
                      "tool":"search_banking_docs",
                      "query":"What is CRR?"
                    }}

                    === General questions ===
                    User: Hi

                    {{
                      "action":"direct_answer",
                      "answer":"Hello! How can I help you?"
                    }}

                    === Calculator Tool ===

                    Purpose:
                    Convert mathematical questions into valid Python mathematical expressions.

                    Rules:

                    - Return ONLY a valid Python mathematical expression.
                    - Use:
                      + - * / % ** // ()
                    - Convert words into numbers.
                    - Preserve operator precedence.
                    - Do not explain the calculation.

                    User: What is 2 + 2?
                    User: What is the sum of 2 and 2?
                    User: Add 2 and 2.
                    User: Addition 2 and 2.

                    {{
                      "action":"tool",
                      "tool":"calculator",
                      "expression":"2+2"
                    }}


                    User: Calculate 25 * 30
                    User: Multiply twenty five by thirty.
                    User: What is twenty five times thirty?
                    User: Product of 25 and 30.
                    User: 25 multiplied by 30.
                    User: Find multiplication of twenty five and thirty.


                    {{
                      "action":"tool",
                      "tool":"calculator",
                      "expression":"25*30"
                    }}

                    User: What is (3+((25+5)*3)) ?

                    {{
                        "action":"tool",
                        "tool":"calculator",
                        "query":"(3+((25+5)*3))"
                    }}

                    User: What is the remainder when 24 is divided by 5?

                    {{
                        "action":"tool",
                        "tool":"calculator",
                        "query":"24%5"
                    }}

                    User: What is Square of 25.

                    {{
                        "action":"tool",
                        "tool":"calculator",
                        "query":""25**2""
                    }}
                    
                    User: Subtract twenty from one hundred and divide by four.
                    
                    {{
                        "action":"tool",
                        "tool":"calculator",
                        "expression":"(100-20)/4"
                    }}

                    === System Tool ===
                    User:
                    What time is it?

                    {{
                        "action":"tool",
                        "tool":"system_tool",
                        "query":"What time is it?"
                    }}

                    User:
                    What day is today?

                    {{
                            "action": "tool",
                            "tool": "system_tool",
                            "query": "What day is today?"
                    }}


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

