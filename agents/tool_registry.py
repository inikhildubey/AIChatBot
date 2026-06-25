# agents/tool_registry.py
from tools.calculator_tool import calculate
from tools.rag_tool import ask_question

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
    }
]
