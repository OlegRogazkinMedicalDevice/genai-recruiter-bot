"""Orchestration: the Main Agent consults advisors, then answers the candidate.

    candidate SMS
          |
      Main Agent ---- decides continue / schedule / end
          |
          |  consults the matching advisor
          v
      Info Advisor        info needed      -> Chroma + job description -> draft
                          info not needed  -> back to the Main Agent
      Scheduling Advisor  schedule         -> SQLite + function call   -> draft
                          don't schedule   -> back to the Main Agent
      Exit Advisor        end conversation -> fine-tuned model         -> draft
                          don't end        -> back to the Main Agent
          |
          |  an advisor that declines sends the turn back, and the Main Agent
          |  consults a second one before giving up
          v
      Main Agent ---- sends one message to the candidate

No advisor talks to the candidate. They return drafts and evidence to the Main
Agent, which owns the channel - the shape the project brief specifies.

The conversation history is the memory: it is passed whole to every advisor,
because the OpenAI API is stateless and the next decision depends on all of it.
"""

from typing import List, Optional

from .modules import config, exit_advisor, info_advisor, main_agent, scheduling_advisor

# Which advisor the Main Agent consults first, and whom it asks when that one
# declines. The brief's diagram calls the second hop "Consult Advisor Again".
FIRST_ADVISOR = {"continue": "info", "schedule": "scheduling", "end": "exit"}
SECOND_ADVISOR = {"info": "scheduling", "scheduling": "info", "exit": "scheduling"}

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

    # -- consulting -------------------------------------------------------
    def _consult(self, advisor: str, candidate_text: str) -> dict:
        """Ask one advisor. It decides for itself whether to act."""
        if advisor == "scheduling":
            phrase = candidate_text or ""
            if not any(hint in phrase.lower() for hint in SCHEDULING_HINTS):
                phrase = ""
            return scheduling_advisor.consult(self.turns, self.conversation_start,
                                              requested_phrase=phrase)
        if advisor == "exit":
            return exit_advisor.consult(self.turns)
        question = candidate_text or "Tell me about the role."
        return info_advisor.consult(self.turns, question)

    def respond(self, candidate_text: Optional[str] = None) -> dict:
        """Take the candidate's SMS, return {'action', 'text', 'consultations'}.

        The Main Agent decides, consults, consults again if the first advisor
        declined, and only then speaks. consultations is the trail, kept so the
        interface can show which advisor was asked and what it decided.
        """
        if candidate_text:
            self.add_candidate(candidate_text)

        action = self.decide()
        first = FIRST_ADVISOR[action]

        consultations = [self._consult(first, candidate_text or "")]
        if not consultations[0]["draft"]:
            consultations.append(
                self._consult(SECOND_ADVISOR[first], candidate_text or ""))

        text = main_agent.compose(self.turns, consultations)

        # The Main Agent's label is a routing decision, not an outcome. The
        # conversation is over only if the Exit Advisor agreed that it is:
        # it may well decline, as it does when a candidate who names a day is
        # routed to `end` but still has no time booked.
        ended = any(c["decision"] == "end_conversation" for c in consultations)

        self.add_recruiter(text, action)
        return {"action": action, "text": text,
                "consultations": consultations, "ended": ended}

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
        if reply["ended"]:
            break


if __name__ == "__main__":
    main()
