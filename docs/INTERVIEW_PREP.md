# Interview Prep — UAE CPI Early-Warning Project

Target audience: strategy / consulting recruiters (BCG X, McKinsey QuantumBlack, Bain Advanced Analytics, Kearney, Oliver Wyman), plus in-house strategy-analytics teams at Mubadala, ADIA, ADNOC, Emirates NBD.

The frame: you are **not** pitching yourself as a pure ML engineer. You're pitching yourself as someone who **turns messy economic data into decisions** — the profile these teams specifically hire for. The technical work is evidence of that capability, not the point of it.

---

## The 90-second version of the project

**Question:** "Walk me through the UAE CPI project."

**Your answer, ≤ 90 seconds:**

> I wanted to see whether UAE headline inflation could be forecast earlier than the monthly CPI release. The data is FCSA's 13-sector CPI panel, 2014 through December 2025.
>
> First I had to reconstruct a continuous series — FCSA rebased from 2014 to 2021 with no overlap period, so naive concatenation produces a 7% discontinuity. I chained them using the published January 2021 MoM change on the new base, validated to ±0.01 percentage points against FCSA's own published YoY figures.
>
> Then three pieces of analysis: STL decomposition showed the series is driven by structural trend, not seasonality — 55% trend versus 3% seasonal. Granger causality identified Tobacco and Textiles as the only sectors that statistically lead headline at the 5% significance level. HDBSCAN clustering on z-scored sector profiles independently rediscovered the 2018 VAT shock, 2020 COVID deflation, and current 2024-25 regime without supervision.
>
> The forecasting model is a Ridge regression on lagged sector YoYs at horizons of 1, 3, and 6 months. Walk-forward expanding-window backtest across 71 out-of-sample origins. At the 3-month horizon it gets 90% directional accuracy and RMSE 1.16 percentage points, against a naive seasonal baseline that has negative R².
>
> The most interesting finding was that the standard "Housing leads inflation" story didn't hold on a strict unconditional test — but it does hold conditional on being in a high-inflation regime, at p=0.049 at 2-month lead. That's the kind of regime-conditional signal that's genuinely useful for pricing and procurement decisions.

**Why this works:** five specific numbers, one acknowledged limitation, one "most interesting finding" that signals you can do more than run libraries.

---

## The questions you'll get and how to answer them

### Category 1: "Why did you pick this approach?"

**Q: Why Ridge regression and not LSTM, Prophet, or Transformers?**

> "144 monthly observations, roughly 70 features once you include sector lags. That's an n/p ratio where linear shrinkage dominates deep learning empirically. I also ran XGBoost on the same features — it matches Ridge on directional accuracy but overfits on RMSE, which is exactly what you'd predict for a sample this small. The Ridge coefficients are also inspectable, which matters if you're handing this to a policy team. With five more years of monthly data, that calculus changes."

**Q: Why Granger causality? It's not real causation.**

> "Agreed — it's statistical predictive precedence, not economic causation. I use the word 'lead' rather than 'cause' in the writeup for exactly that reason. The reason to run it is to prune the feature set and identify channels worth thinking about economically. Tobacco leading headline isn't a deep economic truth — it's excise announcements telegraphing ahead, which is a real but specific mechanism. The forecast model uses all 13 sectors; Granger just tells me which channels are likely contributing to its performance."

**Q: Why HDBSCAN instead of k-means?**

> "Two reasons. HDBSCAN doesn't require you to pre-specify k, which means I'm not picking the number of regimes to match a narrative I already wanted. It also handles variable-density clusters and identifies transitional points as noise rather than forcing them into a cluster — that's important because UAE inflation has genuine multi-month transitions between regimes, like the back half of 2021. k-means would force those months into whichever regime's centroid is closest, which is misleading."

### Category 2: "Walk me through the result"

**Q: The model forecasts 3 months ahead with 90% directional accuracy. Why should I believe that?**

> "The 90% figure comes from 71 out-of-sample origins using walk-forward expanding-window backtest — at every origin, the model is refit on data up to that date and predicts 3 months ahead. There's no look-ahead bias. I'd push back on anyone who sees a 90% number and doesn't ask about the backtest methodology — the naive seasonal baseline on the exact same 71 origins gets 49% directional accuracy, which is worse than coin-flip."

**Q: What about the 2022 inflation spike — did the model miss it?**

> "It caught the direction but overshot the magnitude. You can see that in the forecast timeline chart — the predicted peak is about one percentage point higher than actual. That's a known failure mode of linear regression on regime-shift data. The model trained on data through 2021 hadn't seen anything like 2022, so it extrapolated the trend aggressively. Two fixes: add external drivers like Brent and USD/AED, or condition the model on HDBSCAN regime assignment. Both are in my 'what I'd do next' list."

**Q: Housing is 34% of the basket but doesn't Granger-lead. How do you explain that?**

> "Because the relationship is mechanical and contemporaneous. Housing moves, headline moves the same month — that's not a 'lead,' that's arithmetic. What does hold up is a regime-conditional test: in the high-inflation subsample, Housing MoM Granger-causes headline MoM at 2-month lag with p=0.049. So the Dubai-specific rental pass-through story is real, but only during inflationary regimes. I think the common commentary that 'Housing leads' conflates the mechanical and the conditional effects."

### Category 3: "What did you learn?"

**Q: What surprised you?**

> "Two things. First, I went in expecting Housing to be the dominant lead indicator. It wasn't — Tobacco was, through excise pass-through. That forced me to rewrite the narrative rather than cherry-pick the result. Second, the naive seasonal baseline has negative R² — which means UAE inflation genuinely has no exploitable annual rhythm. That's a specific fact about this economy that wouldn't hold in, say, food-heavy baskets in South Asia."

**Q: What would you do with more time?**

> "Four things, in priority order. External drivers — Brent, USD/AED, mortgage rates, container freight — the model currently uses only internal CPI lags, which caps its performance ceiling. Regime-conditional models — using HDBSCAN to assign the current regime and then using regime-specific coefficients, which should tighten the 1-month horizon where Ridge currently underperforms. Probabilistic forecasts — point forecasts are weak operationally; quantile regression or conformal prediction intervals would make this decision-ready. And nowcasting — predicting the current month's CPI from partial within-month data like Google Trends and card-spend indices, which is arguably more useful than 6-month-ahead forecasts."

**Q: What's the biggest weakness in the analysis?**

> "Sample size. 144 monthly observations sounds like a lot until you remember that covers only three or four genuinely distinct macro regimes, which means the out-of-sample confidence intervals on any model estimate are wide. A production deployment would need to be honest about that in the user interface — the point forecast is less important than knowing the historical analogue the model is drawing from. That's another reason to like the HDBSCAN regime output as part of the deliverable."

### Category 4: "Make it business"

**Q: If I hired you at a CPG company in Dubai, what would you do with this model?**

> "First week: I'd look at which of the 13 sector contributions matter most for our specific basket. A supermarket chain cares about Food and Beverages, which is 13.8% of the CPI basket but probably 70% of their revenue. So I'd reweight the framework to their pricing-sensitive basket rather than FCSA's. Second week: build a dashboard that shows the current HDBSCAN regime, the 3-month forecast with uncertainty, and the top three sector contributors so the procurement team can see where pressure is coming from. Third week: backtest whether acting on the forecast at our 3-month procurement cycle would have improved margin over the last three years — that's what converts an analytical project into a P&L number."

**Q: If you were advising ADNOC or the UAE Central Bank, what would this tell them?**

> "Three things. First, Transportation is the external-shock amplifier — it transmits oil and global supply-chain pressure into UAE inflation with basically no lag. If you want an early warning of imported inflation, watch Transportation MoM. Second, Housing is the domestic pivot — its behavior tells you what inflation regime you're in, not where it's going. Third, the 2020-21 deflation was almost entirely property-oversupply driven — which means the current inflation is significantly housing-driven as well, and any policy lever that affects rental supply will dominate any lever that affects monetary conditions in the near term. The UAE has limited monetary independence because of the dollar peg, but has enormous control over housing supply."

### Category 5: "Close the loop"

**Q: Why this project? Why does it matter to you?**

> "I live in Dubai. I watched prices swing through COVID and the 2022 spike the way anyone did — and I kept reading commentary about 'Housing driving inflation' and 'VAT was transitory' without anyone showing the work. I wanted to see what the actual monthly sector data said when you took it seriously. Half the things I was told going in turned out to be mechanically-driven rather than causally-meaningful, which is exactly the kind of thing a consulting analytics team gets hired to disentangle for a client."

**Q: What's the one thing about this project you're most proud of?**

> "That I rewrote the narrative when the data didn't support what I expected. The original framing of the project claimed Housing Granger-leads headline at p<0.05. When I ran the test carefully on a stationary series, it was p=0.27. I could have picked a different formulation until I got a significant result, or ignored the test and kept the claim. Instead I wrote up four versions of the test, found the regime-conditional one that does hold up, and rewrote the narrative to match. That's the behavior a consulting hiring manager is actually screening for."

---

## Things to actively volunteer

These are the small signals that distinguish a candidate who's done the work from one who's read the notebook. Drop them unprompted if they fit.

1. **The rebasing chain.** "FCSA rebased from 2014 to 2021 with no overlap. Naive concatenation produces a 7% discontinuity. I chained them using the published Jan-2021 MoM and validated the result to one one-hundredth of a percentage point." Most analysts wouldn't notice the discontinuity exists.

2. **The stationarity check.** "I noticed headline YoY has ADF p-value of 0.45, so I ran Granger on MoM instead of YoY, which has p=0.007. Running Granger on non-stationary series inflates Type I error." Signals you understand the statistical assumptions behind your tests.

3. **The negative R² of the naive baseline.** "The baseline has R² of negative 0.74, which means predicting this year's inflation from last year's same month is worse than predicting the mean. That's a specific fact about UAE inflation — there's no exploitable annual rhythm." This is a one-line finding that sounds more sophisticated than the forecast accuracy number.

4. **Why Ridge wins over XGBoost at this sample size.** "On 144 observations with 70 features, the bias-variance trade-off favors shrinkage. XGBoost matches on directional accuracy but overfits on RMSE — I've seen this pattern in several small-sample time series." Signals you make modeling choices for reasons.

5. **The honest caveat about Granger.** "Granger tests statistical predictability, not economic causation. The Tobacco result is real, but what it's likely picking up is excise-announcement telegraphing, not a structural inflation mechanism." You get ahead of the pushback.

---

## Case interview — how to fold this project into a case

If you're doing a McKinsey-style case on pricing, inflation, or Middle East macro:

**The setup:** "Our client is a Dubai-based [retailer / developer / sovereign fund] concerned about inflation."

**Your move:** Pivot to your project as evidence of your analytical thinking, not as a solution.

> "I've actually spent the last month on a closely related problem — modeling UAE CPI with a 3-month-ahead forecasting horizon. Three framings from that work are directly relevant here. Do you want me to start with the regime question — which era are we in — or the composition question — which sectors are driving the pressure?"

This does three things at once: establishes credibility, signals structured thinking, and gives the interviewer a choice that puts them in the driver's seat.

**Three frames you can offer depending on the case type:**

- **Pricing case** → decomposition analysis ("headline inflation is a weighted blend of sector signals — here are the three sectors actually moving")
- **Investment case** → regime analysis ("HDBSCAN says we're in regime 8 of the last 12 years; the closest historical analogue is regime 2, which behaved as follows")
- **Policy case** → causal analysis ("Granger testing says Housing is regime-conditional — here's what that means for the policy lever")

---

## If they ask for the GitHub link mid-interview

Have it ready to paste. Also have `figures/07_forecast_timeline.png` saved as a phone photo — it's the most convincing single visual and you can show it without needing to share your screen.

## After the interview

Send a follow-up thank-you note within 24 hours that includes a one-paragraph addition to the project — something you thought of in the interview. Signals ongoing engagement, signals you're still thinking about the problem after the meeting ended, and gives you a non-awkward reason to email back.
