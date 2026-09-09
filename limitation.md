**Static Defensive Strength Across Season:**
The model uses season-average defensive metrics (e.g., `def_strength_vs_wr = 8.2`) applied uniformly to all weeks, but defenses build cohesion and change over the season. Week 1 predictions use the same defensive rating as Week 9, missing the reality that defenses improve/decline as the season progresses. Solution: Compute rolling defensive stats in Layer 2 per week (use weeks 1-N-1 when predicting week N) and add defensive trend features. Expected gain: 5-10% accuracy improvement.



**Defensive Coordinator Scheme Implementation Over Time:**
The model doesn't account for how a defensive coordinator's scheme matures throughout the season. If the same DC remains, their defensive system typically improves as players learn it, communication tightens, and adjustments are made — but the model treats the DC's effect as static. A DC in Week 1 (implementing new scheme) produces very different results than the same DC in Week 9 (scheme fully installed and proven). Solution: Add a "DC tenure in scheme" feature (weeks since DC started) and interaction terms between DC and weeks-into-season. Expected gain: 3-7% accuracy improvement.




Limitation: No Boom/Bust Probability Distribution

The model outputs a single point prediction (e.g., 16.2 PPR) but doesn't provide uncertainty quantification or tail risk assessment. It can't distinguish between "confident 16.2 (low variance)" and "uncertain 16.2 (high variance)". This means users can't see the probability of boom weeks (>20 points) vs bust weeks (<10 points), which are critical for fantasy roster decisions. Solution: Calculate prediction intervals using training residuals per position/stat (e.g., WR receptions typically ±2.1 predictions), derive floor/ceiling at 10th/90th percentiles, compute probability of user-defined boom/bust thresholds. Expected gain: Not accuracy, but decision-making clarity. Implementation effort: Medium (2-3 hours to compute and serve intervals). Priority for v2: Medium-High (critical for fantasy use case).
