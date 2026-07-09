from datetime import datetime
import platform


async def get_system_info(decision):
    query = decision["query"].lower()

    if "time" in query:
        return {
            "answer": f'Time is{datetime.now().strftime("%H:%M:%S")}'
        }

    if "date" in query:
        return {
            "answer": f'Date is {datetime.now().strftime("%d-%m-%Y")}'
        }

    if "day" in query:
        return {
            "answer": f'Day is {datetime.now().strftime("%A")}'
        }

    if "month" in query:
        return {
            "answer": f'Month is {datetime.now().strftime("%B")}'
        }

    if "year" in query:
        return {
            "answer": f'{datetime.now().strftime("%Y")}'
        }

    if "python" in query:
        return {
            "answer": f'{platform.python_version()}'
        }

    if "operating system" in query or "os" in query:
        return {
            "answer": f'{platform.system()}'
        }

    return {
        "answer": "System information not available."
    }