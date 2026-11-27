"""Fine-tune the Exit Advisor locally, on Hugging Face.

Why locally and not on OpenAI: fine-tuning is closed to new accounts
("The platform is no longer accessible to new users"), so the route is the one
from Model Optimization Part A - a small open model, trained here, and the
result is a folder on disk rather than a model id on someone else's server.

    python scripts/finetune_exit_advisor.py
    python scripts/finetune_exit_advisor.py --epochs 8 --base Qwen/Qwen2.5-0.5B-Instruct

What it trains on: data/exit_advisor_train.jsonl - the 15 'end' turns from the
course conversations, of which 4 are candidate refusals (conversations 5, 8,
13, 15) and 11 are confirmed interviews. The refusals are the behaviour being
taught; the confirmations keep the model from treating every ending as a
withdrawal.

Two decisions worth knowing about:

LoRA, not a full fine-tune. A full fine-tune of a 0.5B model needs roughly
8 GB of RAM for the optimizer states alone. LoRA trains 8.8M parameters, about
1.8% of the 494M in the base model, runs on a laptop CPU, and saves a 34 MB
adapter that can live in the repository.

Loss is computed on the reply only. The prompt tokens are masked with -100, so
the model is scored on what the recruiter should write, not on reproducing the
conversation it was given. With 15 examples this is the difference between
learning the behaviour and memorising the inputs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                      # noqa: E402
from torch.utils.data import Dataset                              # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,    # noqa: E402
                          Trainer, TrainingArguments)

from app.modules import config                                    # noqa: E402

IGNORE = -100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="base model on the Hugging Face Hub")
    parser.add_argument("--train", default=str(config.DATA_DIR / "exit_advisor_train.jsonl"))
    parser.add_argument("--out", default=str(config.ROOT / "models" / "exit-advisor-lora"))
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="full fine-tune instead of LoRA (needs ~8 GB RAM)")
    return parser.parse_args()


def load_examples(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["messages"])
    return rows


def encode(messages, tokenizer, max_length):
    """Tokenize one example; mask everything except the final assistant reply.

    The prompt is the conversation up to the last message, rendered with the
    model's own chat template and a generation prompt on the end. The target is
    that same text plus the reply. Labels are the full ids with the prompt part
    replaced by -100, so loss only lands on the reply.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + messages[-1]["content"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]

    labels = list(full_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = IGNORE

    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


class ExitDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.rows = [encode(m, tokenizer, max_length) for m in examples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch, pad_id):
    """Right-pad a batch; padding is masked out of the loss."""
    width = max(len(row["input_ids"]) for row in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in batch:
        gap = width - len(row["input_ids"])
        out["input_ids"].append(row["input_ids"] + [pad_id] * gap)
        out["attention_mask"].append(row["attention_mask"] + [0] * gap)
        out["labels"].append(row["labels"] + [IGNORE] * gap)
    return {key: torch.tensor(value) for key, value in out.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    examples = load_examples(args.train)
    print(f"training examples: {len(examples)}")
    if len(examples) < 10:
        print("warning: fewer than 10 examples, the run will mostly memorise")

    print(f"base model:        {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)

    if not args.full:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    dataset = ExitDataset(examples, tokenizer, args.max_length)
    lengths = [len(row["input_ids"]) for row in dataset.rows]
    supervised = [sum(1 for label in row["labels"] if label != IGNORE)
                  for row in dataset.rows]
    print(f"token lengths:     min {min(lengths)}, max {max(lengths)}")
    print(f"supervised tokens: {sum(supervised)} of {sum(lengths)} "
          f"({sum(supervised)/sum(lengths):.1%} - the replies only)")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(args.out) / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            use_cpu=not torch.cuda.is_available(),
        ),
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    print("\ntraining - a few minutes on CPU for 15 examples\n")
    trainer.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nsaved: {out_dir}")

    print("\nNext:")
    print(f"  1. put this in .env:  LOCAL_EXIT_MODEL={out_dir}")
    print( "  2. commit the folder — the adapter is about 34 MB")
    print( "  3. re-run section 8 of tests/test_evals.ipynb and compare "
           "against the base model on data/exit_advisor_valid.jsonl")


if __name__ == "__main__":
    main()
