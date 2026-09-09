**Static Defensive Strength Across Season:**
The model uses season-average defensive metrics (e.g., `def_strength_vs_wr = 8.2`) applied uniformly to all weeks, but defenses build cohesion and change over the season. Week 1 predictions use the same defensive rating as Week 9, missing the reality that defenses improve/decline as the season progresses. Solution: Compute rolling defensive stats in Layer 2 per week (use weeks 1-N-1 when predicting week N) and add defensive trend features. Expected gain: 5-10% accuracy improvement.



**Defensive Coordinator Scheme Implementation Over Time:**
The model doesn't account for how a defensive coordinator's scheme matures throughout the season. If the same DC remains, their defensive system typically improves as players learn it, communication tightens, and adjustments are made — but the model treats the DC's effect as static. A DC in Week 1 (implementing new scheme) produces very different results than the same DC in Week 9 (scheme fully installed and proven). Solution: Add a "DC tenure in scheme" feature (weeks since DC started) and interaction terms between DC and weeks-into-season. Expected gain: 3-7% accuracy improvement.