These are based on the 400 sample run of the semantic entropy abstention system. Our final results will use a 2500 sample run, however I expect them to be similar. 

# Semantic Entropy Abstention Results 

## se_correctness_violin
The graph shows the distrbution of semantic entropy scores for correct and incorrect predictions. 
The distribution of correct answers has a median of of 0 with heavy clustering around 0.  
The distribution of incorrect answers has a median of of ~0.9, with much higher variance. 
This is consistent with the results of the original semantic entropy paper, and our graph of F1 and exact match scores vs. semantic entropy rejection threshold-- higher semantic entropy means the answers is less likely to be correct.

## se_correlation_boxplot_*
### std
The Standard SE is the measure of Semantic entropy of *all* of the 10 sampled answers, including "I don't know". 
This plots the middle 50% of the data for Answered and Abstained Samples. The answered box ranges from 0 to 1.25, while the abstained box ranges from 0.4 to 0.9. This makes sense, becaause if the model abstains it necessarily has >= 5 "I don't know" answers, meaning the semantic entropy has a upper bound <insert lower bound with mathematical equation for 5 IDKs and 5 distinct other answers>, and a lower bound of 0. However, the middle 50% of the distribution does not touch 0 because there are few questions for which all samples are "I don't know". 

### cond
The conditional SE shows a much closer alignement between correct and incorrect answers, as expected. Conditional SE is the measure of semantic entropy of *only* the sampled answers that are not "I don't know". The graphs only differ meaningfully in the 75th percentile, with the Answered box being ~2.25 and Abstained ~1.6. This makes sense because the Abstained is calculated on <= 5 non-IDK answers, while the Answered is calculated on >= 5 non-IDK answers. 

The abstained box has a higher median SE (0.65 vs 0.5), meaning that of the non-IDK answers, the abstained answers have greater variance. This makes sense. 

## se_curves_*
The se curves for f1 and em are nearly identical, with all f1 curves being slightly higher than the em curves across both graphs, and with slightly greater distance between f1 curves than em curves. This is expected by the nature of the scoring systems-- f1 rewards partial credit, while em rewards only exact credit. Therefore, improvements in EM scores will correspond to even greater improvements in F1 scores, as if you can get more exact answers, it follows that you can get more partial credit answers. 

Thus, we will only focus on the f1 curves for our analysis. 

### full_f1_score
The full f1 score scores the model's answers on the entire dataset, including abstained samples (with a score of 0). It shows consistent roughly linear improvment as the rejection threshold increases, with linearity breaking down around 0.5 for conditional SE and 0.8 for standard SE. 

Consensus f1 score is higher than greedy f1 score by a consistent margin of ~0.05, which corresponds to a relative improvement of ~20% at 0% rejection threshold, where Greedy Standard SE has a f1 score of ~0.22 and Consensus Standard SE has a f1 score of ~0.27. At 50% Rejection Threshold, the greedy score is ~0.32, showing a 45% improvement by rejecting answers with high semantic entropy. 

### selective_f1_score
CoT SFT: ~.36

In an operational setting, we would only be interested in the f1 score of the answers that we accept, becuase the purpose of the abstention system is to *reduce hallucinations*, rather than provide *more* correct answers (Higher precision vs recall). 

In that light, we are happy to see that the selective f1 score is significantly higher, showing a Greedy Standard SE f1 score of ~0.39 and a Consensus Standard SE f1 score of ~0.5 at 0% rejection threshold. At 50% rejection, the consensus score is ~0.69, showing a 38% improvement by rejecting answers with high semantic entropy. 

Therefore, the total gain in f1 score via abstention is ~.07 over the simple SFT baseline (21% improvement). By additionally using majority voting for the answer, we see a further improvement of 25% a single answer. By additionally rejecting the top 50% of answers by semantic entropy score (~0.65 in this dataset), we see a further improvement of 38%. 

Summary: f1 ~.69 using all methods, vs ~.32 using none (SFT model), so a 116% improvement by answering only 23.5% of the questions. By abstention alone, we see a gain of 60%, responding to 53% of the questions. By SE score alone (rejecting top 50% of the SFT model's answers by SE score), we see a gain of 60%, responding to 50% of the questions. 

Therefore, we find that our CoT/abstention model is more effective than an SE-based abstention model, reaching similar improvments with a greater percentage of questions answered. By combining both, we see twice the improvement, with half of the question coverage. 
