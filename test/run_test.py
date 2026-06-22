import json
from datetime import datetime

import requests

questions = {
    "basics": [
        "what is crr",
        "what is slr",
        "what is bank rate",
        "what is reverse repo rate",
        "what is rtgs",
        "what is neft",
        "what is debit card",
        "what is credit card",
        "what is statutory liquidity ratio",
        "what is cash reserve ratio"
    ],

    "acronym": [
        "what is crr",
        "what is slr",
        "what is rtgs",
        "what is neft",
        "what is dns",
        "what is rbi"
    ],

    "comparison": [
        "difference between rtgs and neft",
        "compare rtgs and neft",
        "rtgs vs neft",
        "difference between debit card and credit card",
        "debit card vs credit card",
        "difference between bank guarantee and letter of credit"
    ],

    "reasoning": [
        "how rbi use crr",
        "why rbi use crr",
        "how does rbi control liquidity",
        "why slr is important",
        "how rtgs works",
        "why banks maintain crr"
    ],

    "factual": [
        "minimum amount for rtgs",
        "rtgs timing",
        "neft timing",
        "current reverse repo rate",
        "locker rent deposit how many years"
    ],

    "noisy": [
        "differance between rtgs and neft",
        "debit card vs credir card",
        "what is statuary liquidity ratio",
        "how rbi use crrr"
    ],

    "out_of_context": [
        "what is bitcoin",
        "how to trade crypto",
        "what is machine learning",
        "who is virat kohli",
        "how to cook pasta"
    ]
}

if __name__ == '__main__':
    BASE_URL = "http://localhost:8000/greet/ask"
    final_list = []
    for question in questions.keys():
        print(f"Finding answers for {question} questions")
        for que in questions.get(question):
            response = requests.post(
                BASE_URL,
                params={"query": que}
            )
            final_list.append({
                question: response.json()
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(final_list, f, indent=4)
    print(f"\nResults saved to: {output_file}")
