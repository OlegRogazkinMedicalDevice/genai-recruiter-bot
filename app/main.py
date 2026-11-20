"""Orchestration: the Main Agent routes every turn to one of three advisors.

    candidate SMS
          |
      Main Agent  --> continue --> Info Advisor      (Chroma + job description)
                  --> schedule --> Scheduling Advisor (SQLite + function call)
                  --> end      --> Exit Advisor       (fine-tuned)

The conversation history is the memory: it is passed whole to every advisor,
because the OpenAI API is stateless and the next decision depends on all of it.
"""

from typing import List, Optional

from .modules import config, exit_advisor, info_advisor, main_agent, scheduling_advisor

SCHEDULING_HINTS = ("monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday", "tomorrow", "today", "next week",
                    "am", "pm", "o'clock")


class RecruiterBot:
    """One instance per conversation."""

    def __init__(self, conversation_start: str, turns: Optional[List[dict]] = None):
        self.conversation_start = conversation_start
        self.turns: List[dict] = list(turns or [])

    # -- history ----------------------------------------------------------
    def add_candidate(self, text: str) -> None:
        self.turns.append({"speaker": "candidate", "text": text})

    def add_recruiter(self, text: str, action: Optional[str] = None) -> None:
        self.turns.append({"speaker": "recruiter", "text": text, "action": action})

    # -- behaviour --------------------------------------------------------
    def decide(self) -> str:
        return main_agent.decide(self.turns)

    def respond(self, candidate_text: Optional[str] = None) -> dict:
        """Take the candidate's SMS, return {'action', 'text'}."""
        if candidate_text:
            self.add_candidate(candidate_text)

        action = self.decide()

        if action == "schedule":
            phrase = candidate_text or ""
            if not any(hint in phrase.lower() for hint in SCHEDULING_HINTS):
                phrase = ""
            text = scheduling_advisor.propose(self.turns, self.conversation_start,
                                              requested_phrase=phrase)
        elif action == "end":
            text = exit_advisor.close(self.turns)
        else:
            question = candidate_text or "Tell me about the role."
            text = info_advisor.answer(question)

        self.add_recruiter(text, action)
        return {"action": action, "text": text}

    def opening(self) -> dict:
        text = ("Thanks for applying to our Python Developer opening. What kinds of "
                "Python projects have you worked on recently?")
        self.add_recruiter(text, "continue")
        return {"action": "continue", "text": text}


def main() -> None:
    """Run the bot in the terminal: python -m app.main"""
    bot = RecruiterBot(conversation_start="2024-04-03T15:12:00Z")
    print(f"[recruiter] {bot.opening()['text']}")
    while True:
        try:
            candidate = input("[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not candidate or candidate.lower() in {"quit", "exit"}:
            break
        reply = bot.respond(candidate)
        print(f"[recruiter / {reply['action']}] {reply['text']}")
        if reply["action"] == "end":
            break


if __name__ == "__main__":
    main()
