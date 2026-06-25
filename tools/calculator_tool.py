# tools/calculator_tool.py

async def calculate(decision):
    expression = decision['expression']
    return eval(expression)
