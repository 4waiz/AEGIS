# Draft questions for the NLBSE'27 code-classification organizers

To: Moritz Mock (momock@unibz.it), Thomas Borsani (tborsani@unibz.it), Barbara Russo (brusso@unibz.it)
Status: DRAFT, not sent. Sending requires the author's approval.

Subject: NLBSE'27 multitask code classification: clarification questions

Dear organizers,

thank you for preparing the NLBSE'27 multitask code classification competition. We would like to
confirm a few details before submitting:

0. **Submission form.** The "submission form" linked on nlbse2027.github.io/tools/ opens the
   *NLBSE'26* form ("fifth edition", deadline 19 December 2025, ICSE'25 formatting). Is that the
   correct form for NLBSE'27 entries, or will a new form be published?

1. **Inference code.** Will you run the submitted model through the provided notebook unchanged
   (global `padding=True`, batch size 600, `RobertaForSequenceClassification`), or may participants
   supply their own inference cell (e.g. a different tokenizer/model class, input normalization,
   length-sorted batching with dynamic padding) as long as the timing/FLOPs measurement loop is kept?
2. **Hidden-test normalization.** Runtime and GFLOPs are totals over the split. Will the constants
   (1000 s, 3,000,000 GFLOPs) be rescaled for the 61,311-instance hidden split, which is twice the
   size of the public test split?
3. **Repetitions.** The instructions say runtime is measured 10 times; the notebook uses 2 forward
   passes per batch. Which one is used for the ranking?
4. **Baseline score.** With the printed values (F1 0.9102, 198.93 s, 2,666,509.7 GFLOPs) the formula
   gives 0.7286, but the notebook displays 0.71. Could you confirm the constants used for the
   official baseline score?
5. **Distillation.** Is it acceptable to train the single submitted model with soft labels from a
   larger teacher that was itself fine-tuned only on the provided training split?
6. **Thresholds / precision.** May the submitted model use per-label decision thresholds tuned on a
   validation split carved from the training data, and fp16/int8 weights?
7. **Anonymity.** The competition page says papers "need not be double-blind", while the general CFP
   says all submissions must be anonymized. We assume the competition page applies; is that correct?

Best regards,
Awaiz Ahmed
