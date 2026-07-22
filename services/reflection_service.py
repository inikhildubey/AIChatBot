from models.reflection import ReflectionResponse


def reflect(
    user_query: str,
    execution_plan: dict,
    tool_result: str
) -> ReflectionResponse:
    """
    Uses an LLM to determine whether
    the user's request has been completed.
    """
    prompt =("""
            You are the Reflection component of an AI Agent.
        
        Your job is NOT to answer the user.
        
        Your job is to evaluate whether the previous execution satisfied the user's request.
        
        Possible statuses:
        
        completed
        next_action
        retry
        waiting_for_user
        
        Rules:
        
        1. If the user request has been fulfilled,
           return completed.
        
        2. If another tool is required,
           return next_action.
        
        3. If the previous action should be attempted again
           using a better query or strategy,
           return retry.
        
        4. If the agent cannot continue until the
           user provides more information,
           return waiting_for_user.
        
        Return ONLY valid JSON.
    """)
    pass