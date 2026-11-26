# Eval summary

Model: `gpt-4.1-mini`, temperature 0.0, best prompt **v2**. 59 eval items.

| Approach | Accuracy |
|---|---|
| Random, 1 of 3 | 33.33% |
| Baseline - always `continue` | 42.37% |
| Turn number alone | 71.19% |
| Main Agent, prompt v1 | 52.54% |
| Main Agent, prompt v2 | 83.05% |
| Main Agent, prompt v3 | 76.27% |
| Main Agent, prompt v4 | 81.36% |
| Rules, no model | 88.14% |
| Ceiling - contradictory labels | 96.61% |

One item is 1.69%, so a difference of two items between two prompts is noise.

## Per label

| Label | TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| continue | 19 | 0 | 6 | 0.760 | 1.000 |
| end | 14 | 1 | 1 | 0.933 | 0.933 |
| invalid | 0 | 3 | 0 | 0.000 | 0.000 |
| schedule | 16 | 6 | 3 | 0.842 | 0.727 |

## The 10 failures

| conv | turn | true | predicted | last candidate message |
|---|---|---|---|---|
| 2 | 3 | continue | schedule | I have three years' experience with Django and Flask. |
| 3 | 3 | continue | schedule | Sure, I have four years of Python experience and two with SQL. |
| 3 | 7 | end | schedule | I would like to set an appointment, does Monday at 3 PM is good. |
| 4 | 3 | continue | schedule | I've been using Python professionally for five years, mostly for data  |
| 8 | 7 | schedule | end | I'm unavailable at that time, as I have other commitments. I'll reach  |
| 10 | 3 | continue | schedule | I've been using Python for a couple of month, I don't have much experi |
| 10 | 5 | schedule | invalid | I've worked a bit with AWS, but I don't have experience with GCP yet. |
| 11 | 3 | continue | schedule | I have three years' experience with Pyhon and AWS. |
| 11 | 5 | continue | invalid | I've worked a bit with AWS, mainly for deploying small apps and managi |
| 11 | 7 | schedule | invalid | Sounds very interesting, I'm confident I can handle it |