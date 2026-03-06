1. Base Model
Meta Llama 3.1 8B Instruct

Why this one:
- It's instruction-tuned (already knows how to follow directions)
- 8 billion parameters (smart enough for nuanced feedback)
- Open-source (runs on our own server, no data leaves)

Note:
- 


eg
{
  "messages": [
    {
      "role": "system",
      "content": "You are a MathForum-style mentor. Give feedback on the student's reasoning, ask clarifying questions, and guide them toward a correct algebraic or pattern-based solution. Do not just give the final numeric answer."
    },
    {
      "role": "user",
      "content": "Service: Algebra\n\nProblem:\nJimmy Alejandre had a great summer playing for his baseball team. He even set a team record for hits in a season! One seventh of his hits were doubles. 12.5% of his hits were home runs. He had 82 singles. He didn't have any triples. How many hits did Jimmy have? (Be sure to include the equation you use to solve the problem.) Extra: In baseball, the statistic called \"total bases\" measures how many bases a player generates based on their hits. The official formula is Hits + Doubles + (Triples times 2) + (Home Runs times 3) What is Jimmy's Total Bases statistic?\n\nStudent work:\nShort answer: Jimmy had 112 hits.\nLong answer: 1/7 of his hits were doubles(1/7x) .125 of his hits were home runs(1/8x) 82 of his hits were singles(82) I made and equation from the information I was given. 1/7x+1/8x+82=x NextI gave the fractions an LCD of 56. 8/56+7/56+82=x Then I added the fractions together. 15/56x+82=x Then I cleared the fraction by multiplying all of the terms by it's LCD. 15x+4592=56x Then I subtracted 15x from both sides. 4592=41x And to finish it I had to make there only 1 x so I divided both sides by 41. 4592/41=41x/41 So x=112(112 hits)"
    },
    {
      "role": "assistant",
      "content": "1/7 of his hits were doubles(1/7x) >.125 of his hits were home runs(1/8x)\n82 of his hits were singles(82)\nI made and equation from the information I was given. 1/7x+1/8x+82=x\nNextI gave the fractions an LCD of 56. 8/56+7/56+82=x\nThen I added the fractions together. 15/56x+82=x\nThen I cleared the fraction by multiplying all of the terms by it's\nLCD. 15x+4592=56x\nThen I subtracted 15x from both sides. 4592=41x\nAnd to finish it I had to make there only 1 x so I divided both sides\nby 41. 4592/41=41x/41 So x=112(112 hits)"
    }
  ],
  "metadata": {
    "thread_id": "thread_3",
    "service": "Algebra",
    "submission_n": 1,
    "submitted_on": 1245886808000,
    "responded_on": 1245888117000,
    "rubric": {
      "strategy": 4,
      "interpretation": 4,
      "completeness": 4,
      "clarity": 4,
      "reflection": 4,
      "accuracy": 4
    }
  }
}

2. System Prompt
You are a MathForum-style mentor. Give feedback on the student's
reasoning, ask clarifying questions, and guide them toward a correct
algebraic or pattern-based solution. Do not just give the final
numeric answer.

Notes:
- Same for every single example -- this is intentional
- Acts as a "mode switch" -- tells the model which behavior to activate
- The actual behavior comes from the training examples, not these words alone


3. User Prompt Structure
Service: [Algebra / Geometry / Math Fundamentals]

Problem:
[The actual math problem text]

Student work:
Short answer: [student's answer]
Long answer: [student's explanation/reasoning]

Notes:
- Consistent structure across all examples -- the model learns to expect this layout
- Service label helps the model adjust feedback to the math domain
- Short + Long answer separation lets the model address both the answer and the reasoning


4. Training Data (what the model learns from)
Metric                              Value
----------------------------------  --------------------------------
Total examples                      2,276
Used for training (after filters)   2,199
Dropped (too long / invalid)        77
Avg student submission length       ~2,600 characters
Avg mentor response length          ~1,970 characters
Services covered                    Algebra, Geometry, Math Fundamentals
Data cleaning                       Removed images, base64, empty replies, echo, duplicates

Overall avg rating above 3.5 and 3above all individually

Key point:
Quality of this data is the single biggest factor. If the mentor responses in the training data are good,
the model will be good. If they're noisy or inconsistent, the model will be too.


5. Training Configuration
Setting         Value         Why it matters
--------------  ------------  ---------------------------------------------------------------------------------------
max_seq_length  1024 tokens   How much of each conversation the model "sees." Longer = better quality but more memory.
                              At 1024, roughly 750-800 words fit. Anything beyond gets truncated.
num_epochs      3             Model sees all 2,199 examples three times. Too few = underfit, too many = memorizes
                              instead of generalizing
LoRA rank       16            How many parameters we update. Higher = learns more but risks overfitting.
                              16 is a balanced default.
learning_rate   2e-4          How aggressively the model updates. Too high = unstable, too low = barely learns


6. Loss Masking (assistant-only training)
- The model is only scored on how well it predicts the mentor's response
- It reads the student submission for context but is never penalized for "getting the student part wrong"
- This is critical -- without it, the model would waste capacity trying to memorize student submissions
  instead of learning mentor feedback patterns


What affects the result MOST (ranked)
1) Training data quality -- garbage in, garbage out. The mentor responses in your dataset are the ceiling.
2) max_seq_length -- if set too low, long mentor responses get cut off during training and the model never learns
   to write complete feedback
3) Base model choice -- Llama 3.1 8B is good; 70B would be better but needs bigger hardware
4) Number of training examples -- 2,199 is solid for LoRA fine-tuning. More helps, but quality > quantity.
5) System prompt consistency -- same prompt everywhere keeps the model focused on one style
6) Hyperparameters (epochs, learning rate, LoRA rank) -- matter, but less than the above. Reasonable defaults
   get you 80% of the way.


One thing to watch
The max_seq_length of 1024 tokens means conversations longer than ~800 words get truncated.
Your average mentor response is ~1,970 characters (~400-500 words), so most fit.
But the longer, more detailed mentor responses -- which might be your best examples -- could get clipped.
Moving to 2048 would capture everything but needs a bigger GPU.