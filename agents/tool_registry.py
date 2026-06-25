# agents/tool_registry.py
from tools.calculator_tool import calculate
from tools.rag_tool import ask_question
from tools.system_tool import get_system_info

# todo optimize the TOOLS dict like this
# TOOLS1 = {"search_banking_docs": {
#     "description": """
#         Search banking concepts including:
#         CRR (Cash Reserve Ratio),
#         SLR (Statutory Liquidity Ratio),
#         RTGS,
#         NEFT,
#         Repo Rate,
#         RBI regulations,
#         Monetary Policy
#         """,
#     "handler": ask_question
# }}

TOOLS = [
    {
        "name": "search_banking_docs",
        "description": """
        Search banking concepts including:
        CRR (Cash Reserve Ratio),
        SLR (Statutory Liquidity Ratio),
        RTGS,
        NEFT,
        Repo Rate,
        RBI regulations,
        Monetary Policy
        """,
        "handler": ask_question
    },
    {
        "name": "calculator",
        "description": """
        Perform mathematical calculations.
        Examples:
        25 * 30
        100 / 4
        15 + 20
        """,
        "handler": calculate
    },
    {
        "name": "system_tool",
        "description": """
        Get current date, time,
        current day,
        month,
        year,
        python version,
        operating system.
        """,
        "handler": get_system_info
    }
]

