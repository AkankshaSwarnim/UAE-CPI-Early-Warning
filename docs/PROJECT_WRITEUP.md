# UAE Inflation Early-Warning System — Full Project Writeup

**Author:** Akanksha Swarnim
**MSc Data Science · University of Birmingham Dubai · 2026**

> A structured Why / What / How writeup of the project. Written for a reader who wants the reasoning chain behind the numbers — not just the numbers.

---

## WHY this project exists

**The problem in one sentence.** Official UAE inflation figures tell you what prices did last month — but by the time you read them, the pricing, procurement, and policy decisions that depend on those numbers have already been made on stale information.

**Why this matters.** The UAE has had one of the most volatile inflation journeys in the GCC over the last decade. Prices peaked at **+7.9% (July 2022)** and troughed at **−2.7% (May 2020)**. Twenty-four consecutive months of deflation — the only such stretch in GCC history — happened between 2019 and 2020. A retailer, developer, or sovereign fund navigating these swings using the monthly CPI release is permanently one step behind the market.

**The question I set out to answer.** Can the 13 sector-level components of the CPI basket (housing, food, transportation, and so on) be combined into a signal that reliably forecasts headline inflation 3 months ahead — with enough accuracy to act on?

---

## WHAT the project actually produces

Three things, each useful on its own:

**1. A forecasting model** that predicts headline UAE CPI 3 months ahead using 13 sector-level signals. Out-of-sample accuracy: **90% directional accuracy, RMSE 1.16 percentage points, R² 0.75.** In plain language: nine times out of ten, it gets the direction right (up or down), and when it's wrong on magnitude it's typically off by about one percentage point.

**2. A regime-detection system** that uses unsupervised machine learning to group months into "inflation regimes" based on their sector fingerprints. This surfaces the underlying economic era you're in — the 2018 VAT shock, the 2020 COVID deflation, the 2022 oil-led spike, the current 2024–25 regime — without anyone hand-labelling them.

**3. An honest causality analysis** of which sectors lead headline inflation. The conventional wisdom says "Housing leads." The rigorous test says: Housing leads *only conditional on high-inflation regimes*. This is a more defensible, more operationally useful finding than the unconditional version — and changing the narrative to match the evidence is, itself, part of the deliverable.

---

## HOW it works — the reasoning chain

### Step 1: Reconstruct a clean data foundation (not glamorous but necessary)

**The problem.** The UAE's statistical agency (FCSA) publishes CPI data on two base periods — 2014-base and 2021-base — without an overlap period. If you naively concatenate them, you get a 7% jump at January 2021 that isn't real.

**The fix.** Use the officially published January 2021 month-over-month change (on the new base) to back out what December 2020's index level should be on the 2021 base. The ratio between that value and the published December 2020 level (on the 2014 base) becomes the linking factor.

**Validation.** Applied to the 'All Items' index, the chained series matches FCSA's own published year-over-year figures to within **0.01 percentage points**. This step takes the whole dataset from "usable with caveats" to "publication-ready."

> *Why this matters:* data work like this usually isn't shown — most portfolio projects start with a clean CSV. Showing the reconstruction signals that I handle real-world data issues rather than waiting for someone else to fix them.

### Step 2: Understand what kind of series this is

Before picking a forecasting method, I decompose headline CPI into three components using STL decomposition:

| Component | Share of variance | What it means |
|---|---|---|
| Trend | **55%** | Slow-moving structural forces (property cycle, oil price regime) |
| Seasonal | **3%** | Calendar effects (Ramadan timing, summer travel, etc.) |
| Residual | **26%** | One-off shocks (VAT introduction, COVID, 2022 energy spike) |

**What this tells us.** UAE inflation is driven by *structural economic forces*, not *calendar patterns*. A naive seasonal model — "predict next year's inflation from last year's same month" — will fail, because there's only 3% seasonal signal to exploit.

**Why this matters for the choice of model.** It rules out simple seasonal baselines and motivates a model that uses broader economic signals (the sector panel) rather than just headline's own history.

### Step 3: Test the prevailing causal story — and rewrite it when the data disagrees

**The assumption going in.** Every local analyst note says "Housing leads UAE inflation by 3 months." Housing is 34% of the basket; the rental market cycle is well-known; the story is plausible.

**The test.** I run Granger causality — the standard statistical test for "does sector X's past behaviour help predict headline CPI?" — on each of the 13 sectors.

**The result.** Housing p-value = 0.08 on a properly stationary series. Close, but doesn't clear the 5% significance bar. The sectors that *do* clear the bar are Tobacco (p < 0.001, 3-month lead) and Textiles (p = 0.03, 1-month lead), through excise and import-cost pass-through respectively.

**The follow-up.** I didn't want to just publish "Housing doesn't lead" — that felt like it was missing something. I ran four more tests:

1. Granger on Housing's *contribution* (weight × YoY) — still not significant (p = 0.26)
2. Granger on first-differenced YoY — still not significant (p = 0.26)
3. Cross-correlation across lags — peak at lag 0, which means Housing moves *with* headline, not *before* it
4. **Granger conditional on high-inflation regime** — **significant at p = 0.049, lag 2**

**The honest framing.** Housing is the *magnitude* pivot (34% of the basket, the single largest contributor). It is not an unconditional *lead* indicator. But during high-inflation regimes specifically, rental pass-through kicks in and Housing does Granger-lead headline by about 2 months. This is both more interesting than the original claim and more defensible under scrutiny.

> *Why this step matters.* Rewriting the narrative when the data contradicts the prior is the behaviour consulting analytics teams specifically screen for. Most candidates either cherry-pick a test that "works" or omit the inconvenient finding. I chose to show the reasoning.

### Step 4: Let the data define the regimes (unsupervised ML)

**The idea.** Instead of drawing boxes around 2018 (VAT era) or 2020 (COVID era) by eye, I let an algorithm find them.

**The method.** For each month, I build a 13-dimensional "fingerprint" — each sector's year-over-year change, z-scored across time. Then I run HDBSCAN clustering on these fingerprints.

**Why HDBSCAN (not k-means).** Three reasons:

- It doesn't require a pre-specified number of clusters — letting the data decide removes my bias about how many regimes "should" exist.
- It handles variable-density clusters — real economic regimes don't have the same number of months.
- It marks transitional months as "noise" rather than forcing them into a cluster — which is correct, because 2–3 month transitions between regimes shouldn't be shoehorned into either side.

**The result.** HDBSCAN independently surfaces four macro eras and nine micro-regimes that align precisely with known UAE economic history — 2015–16 oil crash era, 2018 VAT isolate, 2020–21 COVID deflation, 2022–23 oil-led inflation, 2024–25 current regime. Without any supervision.

**UMAP projection** is used only to visualise this in 2D — it's a presentation tool, not an analytical one. The clustering decisions happen in the full 13-dimensional space.

### Step 5: Build a forecast that beats a hard benchmark

**The design.** Predict headline year-over-year inflation at horizons 1, 3, and 6 months ahead. Features are lagged year-over-year changes of all 13 sectors at lags {1, 2, 3, 6, 12}. Three models are compared:

| Model | What it is | Why it's in the comparison |
|---|---|---|
| **Naive seasonal** | Predict YoY(t+h) = YoY(t−12+h) | Honest benchmark — if this wins, the fancy ML isn't needed |
| **Ridge regression** | Linear model with L2 shrinkage | Robust to small samples (144 observations), interpretable coefficients |
| **XGBoost** | Gradient boosting, 300 trees | Can capture non-linear sector interactions; stress-tests whether Ridge's linearity is costing us accuracy |

**Why Ridge is the main model (not XGBoost).** With only 144 monthly observations and ~70 lagged features, the ratio of observations to parameters is too low for boosted trees to generalise well. Linear shrinkage is the right tool for this sample size. XGBoost is included to prove this empirically — and it does: XGBoost matches Ridge on directional accuracy but slightly overfits on RMSE.

**How the backtest works (the critical methodological choice).** Walk-forward expanding window. At every forecast origin, the model is retrained on data up to that date and asked to predict 3 months ahead. This simulates real deployment exactly — the model never sees future data. 71 out-of-sample origins are evaluated.

**The results at the 3-month horizon:**

| Model | RMSE | Directional accuracy | R² |
|---|---|---|---|
| Naive seasonal | 3.15 pp | 49% | **−0.74** |
| **Ridge** | **1.16 pp** | **90%** | **0.75** |
| XGBoost | 1.33 pp | 90% | 0.68 |

**The most informative number is the baseline's negative R².** Negative R² means the naive model performs *worse than predicting the mean*. This tells you something specific about UAE inflation: there is no exploitable annual seasonal pattern. Nothing about January tells you about next January. This is why the sector-based approach pays off — you're not competing with seasonality, you're building a genuinely new signal.

### Step 6: What the model says right now

Using data through December 2025, the Ridge model forecasts June 2026 headline inflation at roughly **0.9%** year-over-year, against the current rate of 2.0%. With a backtest RMSE of ~0.9 pp at the 6-month horizon, a reasonable 95% prediction band is roughly −0.9% to +2.7%.

The takeaway for a decision-maker isn't the point estimate — it's that the model is not calling for a continuation of the upward surprise trend of late 2025.

---

## KEY TAKEAWAYS

For a recruiter or hiring manager, these are the points worth remembering:

1. **A walk-forward-backtested Ridge regression on 13 sector signals forecasts headline UAE inflation 3 months ahead with 90% directional accuracy** — against a naive baseline that has negative R².
2. **The Housing-leads-inflation story is true only conditionally** — during high-inflation regimes (p = 0.049, 2-month lag), not universally. The unconditional claim is commentary, not evidence.
3. **Unsupervised ML (HDBSCAN) independently rediscovered the known economic regimes of UAE inflation** — 2018 VAT, 2020 COVID deflation, 2022 oil spike, current 2024–25 — without any supervision.
4. **UAE inflation has almost no exploitable seasonal pattern** (3% of variance) — it is a structural-shock series. This rules out seasonal baselines and motivates sector-based models.
5. **The right model choice is Ridge, not XGBoost** — at 144 monthly observations, linear shrinkage beats gradient boosting. Tested empirically.

---

## ASSUMPTIONS, CAVEATS, AND LIMITATIONS

Where the work stops honestly:

- **Internal features only.** The model uses only lagged CPI components. Adding Brent crude, USD/AED, mortgage rates, and container freight would very likely tighten the 2022-spike forecast, which the model currently overshoots.
- **Small sample.** 144 monthly observations cover only three or four genuinely distinct macro regimes. Confidence intervals on any estimate are wide. A production version would need to surface this uncertainty to the user.
- **Point forecasts, not probability distributions.** Real procurement decisions need prediction intervals, not single numbers. Quantile regression or conformal prediction is the right next step.
- **The Housing finding depends on how you define "high-inflation."** I split the sample at the median. Different cut-offs would give slightly different p-values. This is why the claim is "regime-conditional" rather than "proven causal."
- **Granger causality is statistical predictability, not economic causation.** The word "leads" in this writeup means "predictively precedes," not "causes."

---

## WHAT I'D DO WITH ANOTHER SPRINT

In priority order:

1. **Add external drivers** — Brent, USD/AED, mortgage rates, container freight
2. **Regime-conditional models** — use the HDBSCAN regime label as a model switch
3. **Probabilistic forecasts** — quantile regression or conformal prediction
4. **Nowcasting** — predict the current month using partial data (Google Trends, card spend)

---

## REPRODUCIBILITY

Full code, notebook, and pipeline live in this repository:
- `src/` — 5 Python modules (data_loader, analysis, forecast, plots, housing_deep_dive)
- `notebooks/UAE_CPI_Early_Warning.ipynb` — executed end-to-end narrative
- `generate_figures.py` — one-shot pipeline that reproduces every figure from raw data in ~60 seconds
- `docs/dashboard.html` — interactive Plotly dashboard (self-contained, no install)

```bash
git clone https://github.com/AkankshaSwarnim/UAE-CPI-Early-Warning.git
cd UAE-CPI-Early-Warning
pip install -r requirements.txt
python generate_figures.py
```

---

**Links**
- **Interactive dashboard:** [akankshaswarnim.github.io/UAE-CPI-Early-Warning](https://akankshaswarnim.github.io/UAE-CPI-Early-Warning/dashboard.html)
- **LinkedIn:** [linkedin.com/in/akankshaswarnim](https://www.linkedin.com/in/akankshaswarnim/)
