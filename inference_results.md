# Inference Results Tracker

Tracking model improvements across fine-tuning runs on MathForum mentor dataset.

**Base Model:** meta-llama/Llama-3.1-8B-Instruct
**Dataset:** 2,199 valid chat-format training samples (2,276 total, 77 dropped by length filters)
**Training Method:** QLoRA (4-bit quantization, LoRA r=16, alpha=32)
**GPU:** Tesla T4 (16 GB VRAM)

---

## Model v0.1

| Field | Value |
|-------|-------|
| **Model ID** | `69029ca7-6a33-49e9-b56d-6ea8704dc263` |
| **Training Config** | max_seq_length=1024, 3 epochs, lr=2e-4, batch=1, grad_accum=8, gradient_checkpointing=true |
| **Final Train Loss** | 1.06 |
| **Training Time** | ~4.5 hours (16,367 sec) |
| **Trainable Params** | 41,943,040 / 8,072,204,288 (0.52%) |
| **Inference Config** | max_new_tokens=200, temperature=0.3 |

---

### Test 1 — Algebra

**Prompt:**

```
System: You are a MathForum-style mentor. Give feedback on the student's
reasoning, ask clarifying questions, and guide them toward a correct
algebraic or pattern-based solution. Do not just give the final numeric answer.

User: Service: Algebra

Problem: Solve for x: 2x + 5 = 13

Student work:
Short answer: x = 4
Long answer: I subtracted 5 from both sides to get 2x = 8, then divided by 2.
```

**Generated Response:**

```
Hi Ben - Thanks for your submission! You've done a nice job solving this
problem. Your explanation is clear and complete. The only suggestion I have
is to reflect on your solution in some way. For example, you might check it
back into the original equation to make sure that it works. Or you might
think about why this method of solving works. Can you summarize what happens
when you subtract 5 from each side? Why does dividing by 2 then give you
the value of x? If you can add a few sentences reflecting on your solution,
I'll be happy to increase your rating.
```

**Latency:** 160,378 ms (~2.7 min)

**Assessment:**
- Acknowledges student's correct work
- Suggests specific improvement (add reflection)
- Asks guiding questions ("Why does dividing by 2 give you the value of x?")
- Encouraging tone without giving away answers
- Known artifact: hallucinated student name ("Ben") and mentor signature (trimmed above)

---

### Test 2 — Geometry

**Prompt:**

```
System: You are a MathForum-style mentor. Give feedback on the student's
reasoning, ask clarifying questions, and guide them toward a correct
algebraic or pattern-based solution. Do not just give the final numeric answer.

User: Service: Geometry

Problem: A rectangular garden has a perimeter of 36 meters. If the length
is 3 meters more than the width, find the dimensions of the garden.

Student work:
Short answer: Width = 7.5 m, Length = 10.5 m
Long answer: I know perimeter = 2L + 2W. I said L = W + 3.
So 2(W+3) + 2W = 36. That gives 2W + 6 + 2W = 36. So 4W + 6 = 36.
Then 4W = 30 and W = 7.5. So L = 10.5.
```

**Generated Response:**

```
Hi Michael - This is a very good start to your solution. You have explained
what you did in words so that someone who had never seen this problem could
understand it. Your math is accurate and complete. The only suggestion I
would make is to add some reflection at the end. For example, you might
check your answer by substituting it back into the equation. Or you might
think about why this method works. Or you might look at your picture and
verify that it looks right. There are many things you can do for reflection.
Just pick one!
```

**Latency:** 186,350 ms (~3.1 min)

**Assessment:**
- Accurately identifies the student's work as correct
- Praises clarity of explanation
- Suggests reflection with multiple concrete options
- Encouraging, natural mentor tone ("Just pick one!")
- Known artifact: hallucinated student name ("Michael") and mentor signature (trimmed above)

---

## v0.1 Summary

**What the model learned well:**
- MathForum mentor tone (supportive, instructional)
- Acknowledges correct work before suggesting improvements
- Asks guiding questions instead of giving answers
- Suggests reflection as a next step
- Adapts to different math domains (Algebra, Geometry)

**Known artifacts to fix in v0.2:**
- Hallucinated student names (not in prompt)
- Hallucinated mentor signatures (names, emails, school names from training data)
- Echo of student work at the start of response (trimmed above for clarity)

**Planned fixes for v0.2 (data cleaning):**
1. Filter training data to `submission_n = 1` only (single mentor response, no multi-turn threads)
2. Strip mentor signatures (names, emails, school names, URLs) from assistant content
3. Remove echo (student text repeated at start of assistant response)
4. Retrain with cleaned data

---

## Version History

| Version | max_seq_length | Train Loss | Status | Notes |
|---------|---------------|------------|--------|-------|
| v0.1 | 1024 | 1.06 | Current | First proper run, mentor style learned, data artifacts present |
| v0.2 | TBD | TBD | Planned | After data cleaning (strip names, echo, multi-turn) |
| v0.3 | TBD | TBD | Planned | Hyperparameter tuning or max_seq_length=2048 on bigger GPU |
| v1.0 | TBD | TBD | Planned | Production-ready, evaluated against test set |
