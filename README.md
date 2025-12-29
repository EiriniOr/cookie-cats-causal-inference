# Cookie Cats A/B Test Analysis with Causal Inference

Analysis of the Cookie Cats mobile game A/B test using causal inference methods beyond simple hypothesis testing.

## About the Data

Cookie Cats is a mobile puzzle game where players progress through levels. The A/B test examined the effect of moving the first gate (forcing players to wait or pay) from level 30 to level 40 on player retention.

- **Treatment:** gate at level 40
- **Control:** gate at level 30
- **Outcomes:** 1-day and 7-day retention

## Project Structure

```
├── data/                  # Raw and processed data
├── notebooks/             # Exploratory analysis
├── src/
│   ├── eda.py            # Exploratory data analysis
│   ├── classical_ab.py   # Traditional A/B test methods
│   ├── causal_methods.py # Propensity scores, CUPED, doubly robust
│   ├── hte_analysis.py   # Heterogeneous treatment effects
│   ├── sensitivity.py    # Robustness and peeking simulations
│   └── utils.py          # Helper functions
├── app.py                # Streamlit dashboard
└── outputs/              # Figures and results
```

## Methods Implemented

1. **Classical A/B Testing**
   - Difference in means with confidence intervals
   - Chi-squared / Z-test for proportions
   - Power analysis
   - Sample ratio mismatch (SRM) detection

2. **Causal Inference**
   - Regression adjustment
   - Propensity score weighting
   - CUPED variance reduction
   - Doubly robust estimation

3. **Heterogeneous Treatment Effects**
   - Subgroup analysis by game rounds
   - Causal forests (EconML)
   - Meta-learners (T-learner, X-learner)

4. **Sensitivity Analysis**
   - Peeking problem simulation
   - Multiple testing correction
   - Robustness checks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full analysis pipeline
python -m src.run_analysis

# Launch dashboard
streamlit run app.py
```

## References

- [Cookie Cats Dataset on Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
- Rasmus Baath's original analysis
