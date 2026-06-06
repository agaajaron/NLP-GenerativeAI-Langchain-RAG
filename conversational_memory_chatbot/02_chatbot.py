"""
Conversational Memory Chatbot
Interactive CLI demo: runs a scripted exchange to illustrate cross-turn memory,
then opens an optional REPL for live conversation.
"""

import sys
from setup import chatbot, get_session_history

SESSION = "demo"

DEMO_TURNS = [
    "Hi! My name is Alex and I love hiking.",
    "What outdoor activities might I enjoy based on what I told you?",
    "What's my name again?",
]


def chat(message: str, session_id: str = SESSION) -> str:
    response = chatbot.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def run_demo() -> None:
    print("=== Conversational Memory Demo ===\n")
    for turn in DEMO_TURNS:
        print(f"User : {turn}")
        reply = chat(turn)
        print(f"Bot  : {reply}\n")

    history = get_session_history(SESSION)
    print(f"Messages in history: {len(history.messages)}")


def run_repl(session_id: str = "user") -> None:
    print("\n=== Live Chat (type 'quit' to exit) ===\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break
        reply = chat(user_input, session_id=session_id)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    run_demo()
    if "--interactive" in sys.argv or "-i" in sys.argv:
        run_repl()
