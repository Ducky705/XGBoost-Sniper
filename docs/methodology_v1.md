# Phase 1: The "Pyrite" Prototype (Legacy)

**Status:** Deprecated
**Outcome:** High Volatility / Negative Yield

This document details the development of the initial **V1 "Pyrite"** model. We call it "Pyrite" (Fool's Gold) because, during backtesting, it appeared highly profitable due to high accuracy (65%+). However, in live trading, it failed to overcome the bookmaker's vigorish, revealing that **Accuracy $
eq$ Profitability**.

---

## 1. The Fallacy of Accuracy
Sports betting markets are efficient. The closing lines represent a highly accurate aggregation of public information. A common pitfall in ML sports betting is optimizing for classification accuracy.

*   **Example:** A model that predicts a -400 favorite (80% implied probability) will be "correct" 80% of the time.
*   **The Trap:** If that favorite wins 78% of the time, the model is "accurate" but loses money on every bet.

V1 was designed to maximize **Unit Profit**, but it inadvertently fell into the accuracy trap by over-weighting heavy favorites.

## 2. The Technical Stack
*   **Data:** 58,000+ historical picks (NCAAF, NCAAB, NFL, NBA, NHL).
*   **Features:** Lagged rolling metrics (7D/30D ROI, Volatility), Consensus data, and Implied Probability.
*   **Model:** XGBoost Classifier with Isotonic Calibration.

## 3. The October Crash (The Failure)
We deployed V1 with a simple strategy: *Bet if Model Confidence > Implied Probability.*
The results were disastrous. The model went on a massive "tilt" run in October, losing over 180 units in a single month.

![Initial Failure](../assets/figure_1_initial_failure.png)
*Figure 1: The "Pyrite" equity curve. Note the extreme volatility and the catastrophic drawdown.*

## 4. Diagnostic: The "Fake Lock" Syndrome
To understand the failure, we audited the model's predictions bucketed by confidence. We discovered a critical calibration error we call the **"Fake Lock Syndrome."**

*   **50-60% Confidence:** The model was highly accurate and profitable.
*   **60%+ Confidence:** The model became **overconfident**. It assigned 70% probability to teams that only won 60% of the time.

![Calibration Failure](../assets/figure_2_calibration_failure.png)
*Figure 2: Calibration Plot. The model (Blue Bars) consistently overestimated its edge on high-confidence plays.*

## 5. The Winning Formula (Pyrite DNA)
We reverse-engineered the model's decision-making by comparing accepted bets vs. rejected bets. We visualized this as the model's "DNA."

The Pyrite model had a fatal flaw in its DNA:
1.  **Obsessed with Stability:** It heavily favored handicappers with low volatility (safe grinders).
2.  **Followed the Crowd:** It loved high-consensus plays.
3.  **Ignored Value:** It barely looked at the price (Odds).

![Winning Formula DNA](../assets/figure_4_winning_formula_dna.png)
*Figure 3: The Pyrite DNA Radar. It maximizes Stability and Consensus but ignores Value. This creates a profile that bets on "Safe" favorites that are actually overpriced.*

## 6. Conclusion
V1 Pyrite proved that a model can be "smart" (predicting winners) but "broke" (losing money). It laid the groundwork for V2 by identifying exactly what **NOT** to do:
1.  Do not bet on heavy favorites without a massive edge.
2.  Do not ignore the price.
3.  Do not trust "Consensus" blindly.

[👉 Go to Phase 2: The Diamond Optimization](methodology_v2.md)
