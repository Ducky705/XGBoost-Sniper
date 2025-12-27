# THE QUARRY: Quantitative Sports Trading System

**Current Status:** `Live Monitoring`
# 🚨 [ENTER LIVE CONTROL CENTER](https://Ducky705.github.io/XGBoost-Sniper/selector.html) 🚨

**Last Updated:** 2025-12-28

This repository documents the end-to-end evolution of a machine learning system designed to solve the **"Accuracy Fallacy"** in sports betting. It tracks the journey from a high-variance prototype (**V1 Pyrite**) to a disciplined, regime-based asset manager (**V2 Diamond**) and finally to a high-precision sniper (**V3 Obsidian**).

---

## 📊 Executive Summary

Sports betting markets are efficient. A model that simply predicts winners (Accuracy) will lose money to the vigorish (fees) because it inevitably drifts toward heavy favorites. True alpha requires predicting **Value** (ROI).

We developed two distinct models to test this hypothesis:

| Feature | 🔸 V1 Pyrite (Legacy) | 💎 V2 Diamond (Active) | 🔮 V3 Obsidian (Beta) |
| :--- | :--- | :--- | :--- |
| **Philosophy** | "Bet everything with >50% edge" | "Snipe specific inefficiencies" | "High-Precision Straight Bets" |
| **Volume** | High (~134 bets/day) | **High (~67 bets/day)** | **Surgical (<10 bets/day)** |
| **Risk Profile** | Reckless / Uncapped | **10u Daily Cap / Scaled Kelly** | **Dynamic Allocation** |
| **Key Flaw** | Overconfidence on Favorites | None (so far) | Data Scarcity |
| **Result** | **14.30% ROI** (The "Churn") | **15.03% ROI** (The "Edge") | **Targeting >20% ROI** |

---

## 📡 Live Performance Dashboard

### 1. Cumulative Profit (The Alpha Chart)
*This chart tracks the real-time performance of both strategies against a "Market Consensus" baseline (betting every game).*
*   **Green Line (Diamond):** The optimized strategy. Note the steady, low-volatility growth.
*   **Red Line (Pyrite):** The raw model. Note the high volatility and eventual decay.
*   **Gray Dotted:** The market baseline (losing to the vig).

![Live Curve](docs/assets/live_curve.png)

### 2. Model Evolution & Comparison
*Visualizing the distinct behavior of each generation.*

| 🔸 Pyrite (The Brawler) | 💎 Diamond (The Manager) | 🔮 Obsidian (The Sniper) |
| :---: | :---: | :---: |
| ![V1 Curve](docs/assets/pyrite_live_curve.png) | ![V2 Curve](docs/assets/live_curve.png) | ![V3 Curve](docs/assets/obsidian_curve.png) |
| *High Volatility / High Volume* | *Steady Growth / Capped Risk* | *Max Precision / Selective* |

### 3. The "Toxic Asset" Audit (Sport Health)
*Why did V1 fail? It didn't know when to quit. V2 & V3 implement strict "Regime Filtering" to ban toxic sports.*

| 🔸 V1 Pyrite (Bleeding Edge) | 💎 V2 Diamond (Surgical) |
| :---: | :---: |
| ![V1 Sport](docs/assets/pyrite_sport.png) | ![V2 Sport](docs/assets/diamond_sport.png) |
| *Loses money on NFL/MLB noise.* | *Only trades profitable regimes.* |

### 3. Calibration Check (Bet Sizing)
*Does the model know when it's right? Bigger bets should yield higher ROI.*

| 🔸 V1 Pyrite | 💎 V2 Diamond |
| :---: | :---: |
| ![V1 Size](docs/assets/pyrite_size.png) | ![V2 Size](docs/assets/diamond_size.png) |
| *Flat performance across sizes.* | *Strong correlation: Confidence = Profit.* |

---

## 📚 Methodology & Research

This project is not just code; it is a documented research experiment.

### 1. Phase 1: The "Pyrite" Prototype (Legacy)
*   **The Hypothesis:** Can a calibrated XGBoost model beat the market using raw probability?
*   **The Failure:** Discovering the "Fake Lock" syndrome and the cost of volatility.
*   **The DNA:** Visualizing the flawed logic of the initial model.

### 2. Phase 2: The "Diamond" Optimization (Active)
*   **The Fix:** Using **Grid Search** to find the "Sweet Spot".
*   **The Core 4:** Implementing Regime Filtering to ban toxic sports.
*   **The Math:** Kelly Criterion, Value Floors, and Fade Scores.

### 3. Phase 3: The "Obsidian" Refinement (Beta)
*   **The Goal:** Maximize ROI by sacrificing volume.
*   **The Method:** "Sniper" logic that only fires on straight bets with >60% edge.
*   **The Result:** A vertically scalable strategy with minimal drawdown.

---

## 📂 System Architecture

The project has been reorganized for modularity and professional maintenance:

*   **`src/`**: Core shared modules.
    *   `pipeline.py`: Centralized Supabase data fetching and feature engineering.
    *   `models.py`: Encapsulated simulation logic for V1, V2, and V3 models.
*   **`scripts/`**: Executable orchestration and asset hub.
    *   `monitor.py`: The daily operations script. Orchestrates data flow and updates reports.
    *   `generate_assets.py`: Central hub for all visual assets and dashboard data injection.
*   **`models/`**: Serialized XGBoost models and configuration files.
*   **`docs/`**: Web application files (HTML/CSS) and research methodology.
*   **`research/`**: Original research notebooks and UI mockups.
*   **`tools/`**: Auxiliary utility scripts.

## 🚀 Usage

To run the full daily cycle (data sync, simulation, and asset generation):
```bash
python scripts/monitor.py
```

## 📝 Latest Daily Action
[👉 Click here to view the Daily Log (LATEST_ACTION.md)](./LATEST_ACTION.md)
