# tools/calculator_tool.py

import ollama


def build_expression(query: str) -> str:
    prompt = f"""
                You convert natural language mathematical questions into valid Python arithmetic expressions.
                
                Rules:
                - Return ONLY the expression.
                - Do not explain anything.
                - Do not return JSON.
                - Preserve operator precedence.
                - Supported operators:
                  +  -  *  /  %  **  //  ()
                
                Examples:                
                User:
                2 + 2
                
                Output:
                2+2
                
                User:
                Multiply twenty five by thirty
                
                Output:
                25*30
                
                User:
                Subtract twenty from one hundred and divide by four
                
                Output:
                (100-20)/4
                
                User:
                What is the remainder when 24 is divided by 5?
                
                Output:
                24%5
                
                User:
                Square of 25
                
                Output:
                25**2
                
                Question:
                
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

    return response["message"]["content"].strip()


async def calculate(query):
    # query = decision["query"]
    expression = build_expression(query)
    try:
        answer = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    return {
        "status": "success",
        "answer": answer
    }