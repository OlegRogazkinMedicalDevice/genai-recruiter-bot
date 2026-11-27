"""Tests for the orchestration in app/main.py.

These run offline. No API key, no network, no vector store: every advisor is
replaced by a stub that records how it was called. What is under test is the
Main Agent's control flow, which is the part the project brief specifies -
who is consulted, what happens when an advisor declines, and who is allowed
to speak to the candidate.

    python tests/test_main.py        # plain, no dependencies
    pytest tests/test_main.py        # same tests, if pytest is installed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main as orchestrator                                # noqa: E402
from app.main import RecruiterBot                                   # noqa: E402

START = "2024-04-03T15:12:00Z"


# --- stubs ----------------------------------------------------------------

class Recorder:
    """Stands in for one advisor and remembers that it was consulted."""

    def __init__(self, decision, draft):
        self.decision = decision
        self.draft = draft
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return {"advisor": "stub", "decision": self.decision,
                "draft": self.draft, "evidence": []}


def wire(monkey, action, info, scheduling, exit_):
    """Point the orchestrator at stubs instead of the real advisors."""
    monkey["decide"] = lambda turns: action
    orchestrator.main_agent.decide = monkey["decide"]
    orchestrator.info_advisor.consult = info
    orchestrator.scheduling_advisor.consult = scheduling
    orchestrator.exit_advisor.consult = exit_
    orchestrator.main_agent.compose = lambda turns, consultations, llm=None: next(
        (c["draft"] for c in consultations if c["draft"]), "WROTE IT MYSELF")


# --- tests ----------------------------------------------------------------

def test_continue_consults_the_info_advisor():
    info = Recorder("info_needed", "The team works with Django.")
    sched = Recorder("schedule", "never reached")
    exit_ = Recorder("end_conversation", "never reached")
    wire({}, "continue", info, sched, exit_)

    reply = RecruiterBot(START).respond("What stack do you use?")

    assert reply["action"] == "continue"
    assert reply["text"] == "The team works with Django."
    assert (info.calls, sched.calls, exit_.calls) == (1, 0, 0)
    assert len(reply["consultations"]) == 1


def test_schedule_consults_the_scheduling_advisor():
    info = Recorder("info_needed", "never reached")
    sched = Recorder("schedule", "Thursday 09:00 or 10:00?")
    exit_ = Recorder("end_conversation", "never reached")
    wire({}, "schedule", info, sched, exit_)

    reply = RecruiterBot(START).respond("Can we meet next week?")

    assert reply["text"] == "Thursday 09:00 or 10:00?"
    assert (info.calls, sched.calls, exit_.calls) == (0, 1, 0)


def test_end_consults_the_exit_advisor():
    info = Recorder("info_needed", "never reached")
    sched = Recorder("schedule", "never reached")
    exit_ = Recorder("end_conversation", "No worries, thanks for letting me know.")
    wire({}, "end", info, sched, exit_)

    reply = RecruiterBot(START).respond("I took another offer.")

    assert reply["text"] == "No worries, thanks for letting me know."
    assert (info.calls, sched.calls, exit_.calls) == (0, 0, 1)


def test_a_declining_advisor_sends_the_turn_back_and_a_second_is_consulted():
    """The brief's "Consult Advisor Again" branch."""
    info = Recorder("info_not_needed", "")
    sched = Recorder("schedule", "Thursday at 10:00 works, shall I book it?")
    exit_ = Recorder("end_conversation", "never reached")
    wire({}, "continue", info, sched, exit_)

    reply = RecruiterBot(START).respond("Sounds good.")

    assert (info.calls, sched.calls) == (1, 1)
    assert reply["text"] == "Thursday at 10:00 works, shall I book it?"
    assert [c["decision"] for c in reply["consultations"]] == \
        ["info_not_needed", "schedule"]


def test_the_main_agent_writes_the_turn_when_every_advisor_declines():
    info = Recorder("info_not_needed", "")
    sched = Recorder("dont_schedule", "")
    exit_ = Recorder("dont_end", "")
    wire({}, "continue", info, sched, exit_)

    reply = RecruiterBot(START).respond("Hm.")

    assert (info.calls, sched.calls) == (1, 1)
    assert reply["text"] == "WROTE IT MYSELF"


def test_the_reply_is_appended_to_the_history_with_its_action():
    info = Recorder("info_needed", "We are hiring for the data platform team.")
    wire({}, "continue", info, Recorder("schedule", ""), Recorder("dont_end", ""))

    bot = RecruiterBot(START)
    bot.respond("Which team is this for?")

    assert [t["speaker"] for t in bot.turns] == ["candidate", "recruiter"]
    assert bot.turns[-1]["action"] == "continue"
    assert bot.turns[-1]["text"] == "We are hiring for the data platform team."


def test_the_opening_turn_needs_no_advisor():
    bot = RecruiterBot(START)
    opening = bot.opening()

    assert opening["action"] == "continue"
    assert len(bot.turns) == 1


def test_the_conversation_ends_only_when_the_exit_advisor_says_so():
    """A routing label of `end` is not an outcome.

    The Main Agent can route to `end` while the Exit Advisor declines and the
    Scheduling Advisor rescues the turn with real times. Closing the chat on
    the label would hang up on a candidate who was just offered a slot.
    """
    exit_ = Recorder("dont_end", "")
    sched = Recorder("schedule", "Friday at 09:00 or 10:00 — do either work?")
    wire({}, "end", Recorder("info_not_needed", ""), sched, exit_)

    reply = RecruiterBot(START).respond("Can we meet next Friday morning?")

    assert reply["action"] == "end"
    assert reply["ended"] is False
    assert "Friday" in reply["text"]


def test_the_conversation_ends_when_the_exit_advisor_closes_it():
    exit_ = Recorder("end_conversation", "All the best with the new role.")
    wire({}, "end", Recorder("info_not_needed", ""),
         Recorder("dont_schedule", ""), exit_)

    reply = RecruiterBot(START).respond("I took another offer, thanks.")

    assert reply["ended"] is True


# --- the slots an advisor reports must be the slots it used ---------------

def test_the_requested_day_changes_which_slots_are_resolved():
    """The evidence shown to the Main Agent has to follow the request.

    Looking the slots up a second time, without the candidate's wording, put
    Thursday in the consultation panel while the message offered Friday.
    """
    from app.modules import db, scheduling_advisor

    generic, _ = scheduling_advisor.resolve_slots(START)
    friday, _ = scheduling_advisor.resolve_slots(START, "can we do next Friday morning?")

    assert db.describe_slots(generic) != db.describe_slots(friday)
    assert "Friday" in db.describe_slots(friday)
    assert "Thursday" in db.describe_slots(generic)


def test_a_closed_day_is_answered_rather_than_dropped():
    from app.modules import db, scheduling_advisor

    slots, note = scheduling_advisor.resolve_slots(START, "how about Monday?")

    assert slots, "alternatives must still be offered"
    assert "does not interview" in note
    assert "Monday" not in db.describe_slots(slots)


# --- plain runner ---------------------------------------------------------

def _run():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok    {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
