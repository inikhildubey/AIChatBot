from memory.redis_memory import get_history, save_turn


def save(turn):
    save_turn(turn)


def _format_conversation(history):
    conversation = ""
    for turn in history:
        conversation += (
            f"User: {turn['user']}\n"
            f"Assistant: {turn['assistant']}\n\n"
        )
    return conversation


def get_planning_context(query):
    history = get_history()
    return _format_conversation(history)


def get_execution_context(query):
    history = get_history()
    return _format_conversation(history)