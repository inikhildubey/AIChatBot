from datetime import datetime
import platform


async def get_system_info(decision):

    query = decision["query"].lower()

    if "time" in query:
        return {
            "answer": datetime.now().strftime("%H:%M:%S")
        }

    if "date" in query:
        return {
            "answer": datetime.now().strftime("%d-%m-%Y")
        }

    if "day" in query:
        return {
            "answer": datetime.now().strftime("%A")
        }

    if "month" in query:
        return {
            "answer": datetime.now().strftime("%B")
        }

    if "year" in query:
        return {
            "answer": datetime.now().strftime("%Y")
        }

    if "python" in query:
        return {
            "answer": platform.python_version()
        }

    if "operating system" in query or "os" in query:
        return {
            "answer": platform.system()
        }

    return {
        "answer": "System information not available."
    }