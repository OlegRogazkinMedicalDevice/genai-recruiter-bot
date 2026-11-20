"""Build the fine-tuning file for the Exit Advisor.

Why this component and not the Main Agent: the label 'end' covers two opposite
endings, and a prompted base model congratulates candidates who just withdrew.

    15 'end' turns in the course data
    11 confirmations  (conversations 2,4,6,7,9,10,11,12,14 and 1,3)
     4 refusals       (conversations 5, 8, 13, 15)

All 15 go into the training file: the refusals are the behaviour being taught,
the confirmations are the counter-examples that stop the model from treating
every ending as a withdrawal.

HONESTY NOTE, worth keeping in the README: 15 examples is a demonstration, not
a trained model. With only 4 refusals there is nothing left to hold out, so
evaluating on those same 4 would be measuring memorisation. The script also
writes a small hand-written validation set of paraphrased refusals and
confirmations - not course data, written for this project - which is what the
before/after comparison should be run on.

    python scripts/build_finetune_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = OUT_DIR / "exit_advisor_train.jsonl"
VALID_FILE = OUT_DIR / "exit_advisor_valid.jsonl"

REFUSAL_CONVERSATIONS = {5, 8, 13, 15}


def to_messages(history, reply):
    """One training example in the OpenAI chat fine-tuning format."""
    messages = [{"role": "system", "content": config.EXIT_ADVISOR_SYSTEM_PROMPT}]
    for turn in history[-8:]:
        role = "assistant" if turn["speaker"] == "recruiter" else "user"
        messages.append({"role": role, "content": turn["text"]})
    messages.append({"role": "assistant", "content": reply})
    return {"messages": messages}


def main() -> None:
    conversations = json.loads(config.CONVERSATIONS_JSON.read_text(encoding="utf-8"))

    rows, refusals = [], 0
    for conversation in conversations:
        turns = conversation["turns"]
        for index, turn in enumerate(turns):
            if turn["label"] == "end":
                rows.append(to_messages(turns[:index], turn["text"]))
                if conversation["conversation_id"] in REFUSAL_CONVERSATIONS:
                    refusals += 1

    TRAIN_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    print(f"{TRAIN_FILE.name}: {len(rows)} examples "
          f"({refusals} refusals, {len(rows) - refusals} confirmations)")

    # --- hand-written held-out set, NOT from the course data ---------------
    held_out = [
        # refusals, paraphrased
        (["Are you still interested in the Python Developer role?",
          "Actually I've accepted another offer, so please withdraw my application."],
         "Thanks for letting me know, and congratulations on the offer. "
         "I'll close your application - do reach out if anything changes."),
        (["Can we book you in for Thursday at 2 PM?",
          "I'd rather not continue with this process. Please take me off your list."],
         "Understood, I've removed you from the list. Thanks for your time, "
         "and best of luck."),
        # confirmations, paraphrased
        (["Thursday at 2 PM is free - does that work?",
          "Yes, Thursday at 2 works for me."],
         "Perfect, Thursday at 2 PM is booked. A calendar invite is on its way."),
        (["The engineering manager is free Friday at 11 AM.",
          "Great, let's do Friday."],
         "Excellent - Friday at 11 AM it is. You'll get the invite shortly."),
    ]
    rows = []
    for texts, reply in held_out:
        history = [{"speaker": "recruiter" if i % 2 == 0 else "candidate",
                    "text": text} for i, text in enumerate(texts)]
        rows.append(to_messages(history, reply))

    VALID_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    print(f"{VALID_FILE.name}: {len(rows)} examples (hand-written, held out)")

    print("\nNext:")
    print("  OpenAI route : upload exit_advisor_train.jsonl at "
          "platform.openai.com -> Fine-tuning, then put the resulting model id "
          "in .env as FINETUNED_EXIT_MODEL")
    print("  Hugging Face : train Qwen/Qwen2.5-0.5B-Instruct locally on the same "
          "file, as in Model Optimization Part B")


if __name__ == "__main__":
    main()
