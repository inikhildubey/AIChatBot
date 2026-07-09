# agents/tool_registry.py
from tools.calculator_tool import calculate
from tools.rag_tool import ask_question
from tools.system_tool import get_system_info


# TOOLS = [
#     {
#         "name": "search_banking_docs",
#         "description": """
#         Search information from banking documents.
#         Use this tool whenever the user asks about:
#         - CRR (Cash Reserve Ratio)
#         - SLR (Statutory Liquidity Ratio)
#         - RTGS
#         - NEFT
#         - Repo Rate
#         - RBI
#         - Banking regulations
#         - Monetary policy
#
#         Never use this tool for:
#         - mathematics
#         - date or time
#         - operating system information
#         """,
#         "handler": ask_question
#     },
#     {
#         "name": "calculator",
#         "description": """
#         Perform arithmetic calculations.
#         Use this tool whenever the user asks to:
#         - add
#         - subtract
#         - multiply
#         - divide
#         - modulo (%)
#         - exponent (**)
#         - evaluate arithmetic expressions
#
#         Examples:
#         25 + 30
#         100 / 4
#         24 % 5
#         2 ** 10
#
#         Never use this tool for:
#         - current time
#         - current date
#         - day
#         - month
#         - year
#         - banking questions
#         - system questions
#         """,
#         "handler": calculate
#     },
#     {
#         "name": "system_tool",
#         "description": """
#         Retrieve information from the local computer.
#         Use this tool whenever the user asks about:
#         - current time
#         - current date
#         - today's day
#         - month
#         - year
#         - python version
#         - operating system
#
#         This includes questions like:
#
#         - What time is it?
#         - What's the time?
#         - Time now
#         - Current time
#         - Tell me the time
#         - What day is today?
#
#         Never use this tool for:
#
#         - arithmetic
#         - banking questions
#         """,
#         "handler": get_system_info
#     }
# ]

TOOLS = {
    "calculator": {
        "description": """
        Perform arithmetic calculations.
        Use this tool whenever the user asks to:
        - add
        - subtract
        - multiply
        - divide
        - modulo (%)
        - exponent (**)
        - evaluate arithmetic expressions
        
        Examples:
        25 + 30
        100 / 4
        24 % 5
        2 ** 10
        
        Never use this tool for:
        - current time
        - current date
        - day
        - month
        - year
        - banking questions
        - system questions
        """,
        "handler": calculate
    },
    "search_banking_docs": {
        "description": """
        Search information from banking documents.
        Use this tool whenever the user asks about:
        - CRR (Cash Reserve Ratio)
        - SLR (Statutory Liquidity Ratio)
        - RTGS
        - NEFT
        - Repo Rate
        - RBI
        - Banking regulations
        - Monetary policy
        
        Never use this tool for:
        - mathematics
        - date or time
        - operating system information
        """,
        "handler": ask_question
    },
    "system_tool": {
        "description": """
        Retrieve information from the local computer.
        Use this tool whenever the user asks about:
        - current time
        - current date
        - today's day
        - month
        - year
        - python version
        - operating system
        
        This includes questions like:
        
        - What time is it?
        - What's the time?
        - Time now
        - Current time
        - Tell me the time
        - What day is today?
        
        Never use this tool for:
        
        - arithmetic
        - banking questions
        """,
        "handler": get_system_info
    }
}
