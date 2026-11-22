"""A rule-based reference classifier for the Main Agent's task.

Not part of the running system. It exists to answer one question before any
prompt is written: **how far can you get on this data without a model?**

Scored on the 59 labelled recruiter turns:

    always "continue"   42.37%   the baseline
    these rules         88.14%   no model involved
    perfect             96.61%   the ceiling, two labels contradict each other

That middle number is what makes the eval readable. An LLM scoring 70% is not
a hard task, it is a bad prompt. The rules were also how the decision logic in
MAIN_AGENT_PROMPT_V2 was found: each rule here is one line of that prompt.

The seven turns these rules still miss are exactly the contradictory ones -
the candidate's first experience answer, labelled `schedule` in ten
conversations and `continue` in five.
"""

import re
from typing import List

DECLINE = (r"\b(not interested|no longer|another offer|withdraw|decline|"
           r"remove me|take me off|stop texting|rather not)\b")
CANNOT = (r"\b(can'?t|cannot|unavailable|don'?t work|doesn'?t work|not work|"
          r"busy|other times|another time)\b")
ACCEPT = (r"\b(works|sounds good|sounds great|see you then|perfect|"
          r"yes,? absolutely|that works|is good|confirmed|set an appointment)\b")
READY = r"\b(happy to schedule|confident i can handle|let'?s (meet|schedule))\b"


def predict(turns: List[dict]) -> str:
    """Same signature as main_agent.decide: history in, one label out."""
    candidate_turns = [turn for turn in turns if turn["speaker"] == "candidate"]
    if not candidate_turns:
        return "continue"

    last = candidate_turns[-1]["text"]

    if re.search(DECLINE, last, re.I):                  # rule 1: withdrawal
        return "end"
    if re.search(ACCEPT, last, re.I) and "?" not in last:
        return "end"                                    # rules 2 and 3
    if re.search(CANNOT, last, re.I):                   # rule 4: slot rejected
        return "schedule"
    if re.search(READY, last, re.I):                    # rule 6: ready to meet
        return "schedule"
    if len(candidate_turns) == 1:                       # rule 5: first answer
        return "schedule"
    return "continue"                                   # rule 7
