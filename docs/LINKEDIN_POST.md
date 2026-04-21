# LinkedIn Post Drafts

Three variants — pick the voice that feels most like you. Each is engineered for LinkedIn's algorithm (hook in the first 2 lines, specific numbers, one clean call-to-action). Attach the forecast timeline chart (`figures/07_forecast_timeline.png`) as the image — it's the strongest single visual in the set.

---

## VARIANT A — Punchy, consulting-voice (recommended for BCG X / strategy recruiters)

> **UAE headline inflation tells you what already happened. Sector-level CPI tells you what's coming.**
>
> I spent the last month building a sector-based early-warning system for UAE inflation. Here's what the data actually shows:
>
> 🎯 **Ridge regression on 13 sector signals forecasts headline CPI 3 months ahead with 90% directional accuracy.**
> RMSE 1.16 pp vs. 3.15 pp for a naive seasonal baseline. R² 0.75. Backtested across 71 out-of-sample origins (2020–2025) with walk-forward expanding windows — no look-ahead bias.
>
> 🧩 **HDBSCAN clustering independently rediscovered 4 macro eras and 9 micro-regimes** — the 2018 VAT shock, 2020 COVID deflation, 2022 oil-led spike, current 2024–2025 regime — without being told they exist.
>
> 📉 **Granger causality identified Tobacco and Textiles as the only sectors that statistically lead headline inflation** (p < 0.05), at 3-month and 1-month horizons respectively. Housing is close (p = 0.08) — its influence runs through magnitude (34% basket weight), not through statistically significant leads. A result I didn't expect going in.
>
> The honest version of the story was more interesting than the version I started with. Along the way I had to:
> → Chain the 2014-base and 2021-base CPI indices (FCSA rebased with no overlap period) — validated to ±0.01 pp against published figures
> → Rerun every claim I'd inherited from prior analyses. Some didn't hold up.
> → Resist the temptation to tune HDBSCAN's min_cluster_size until the number of regimes matched the narrative I wanted.
>
> **Full writeup, code, and reproducible pipeline:** [GitHub link]
>
> Built in Python (pandas, scikit-learn, statsmodels, HDBSCAN, UMAP, XGBoost). Part of my MSc Data Science at @University of Birmingham Dubai.
>
> Open to conversations with anyone working at the intersection of applied ML and real economic questions — particularly strategy consulting and analytics teams in Dubai.
>
> \#UAE #DataScience #MachineLearning #Economics #Inflation #Forecasting #Consulting

---

## VARIANT B — Story-first, technical details below the fold

> In July 2022, UAE inflation hit **7.9%** — the highest in over a decade.
> In May 2020, it bottomed at **−2.7%**.
> Four macro eras, nine micro-regimes, twelve years.
>
> The question I kept circling: **can you see these shifts coming?**
>
> So I built a forecasting engine on 13 sector-level signals from the UAE CPI basket.
>
> **Results** (walk-forward backtest, 71 origins, no look-ahead bias):
> • **3-month horizon:** RMSE 1.16 pp, **90% directional accuracy**, R² 0.75
> • **Naive seasonal baseline:** RMSE 3.15 pp, R² −0.74
>
> In plain English: using sector signals gets the 3-month direction right 9 times out of 10, with error 2.7× smaller than the baseline.
>
> **The unexpected finding:** when I ran Granger causality on each sector, it wasn't Housing that statistically led headline inflation. It was Tobacco (3-month lead, p < 0.001) and Textiles (1-month lead, p = 0.03). Housing's role is in magnitude, not lead — its 34% basket weight means it *is* the base signal rather than the early warning.
>
> And when HDBSCAN clustering independently ran on the 13-sector z-score profiles, it rediscovered the 2018 VAT shock, the 2020 deflation, and the current 2024–2025 regime as distinct clusters — without any supervision.
>
> The honest analysis was more interesting than the one I thought I'd find.
>
> 📂 Full code, notebook, and reproducible pipeline: [GitHub link]
> 🛠️ Python · scikit-learn · statsmodels · HDBSCAN · UMAP · XGBoost
>
> MSc Data Science · University of Birmingham Dubai · 2026
>
> \#DataScience #MachineLearning #UAE #Economics #Consulting

---

## VARIANT C — Short, understated (the "I let the work speak" voice)

> Spent the last month asking a simple question: can sector-level UAE CPI give an early warning on headline inflation?
>
> Built a Ridge regression forecaster on 13 sector signals. Walk-forward backtest, 71 out-of-sample origins. At 3 months ahead: **RMSE 1.16 pp, 90% directional accuracy, R² 0.75.** The naive seasonal baseline gets R² −0.74 on the same data — so this is a real signal.
>
> Two findings I didn't expect:
>
> **1.** Tobacco and Textiles are the only sectors that pass a strict Granger test (p < 0.05) as leading indicators. Housing — the 34% basket heavyweight — is close but doesn't clear the bar. Its role is magnitude, not lead.
>
> **2.** HDBSCAN on z-scored sector profiles independently rediscovers the 2018 VAT shock, 2020 COVID deflation, 2022 oil spike, and current 2024–2025 regime. No supervision, no pre-specified number of clusters.
>
> Along the way I had to chain FCSA's 2014-base and 2021-base CPI indices — they rebased without an overlap, which creates a 7% discontinuity if you concatenate naively. The linking factor is derived from the Jan-2021 MoM on the new base; result validates to ±0.01 pp against published FCSA figures.
>
> Code, notebook, methodology: [GitHub link]
>
> MSc Data Science · University of Birmingham Dubai
>
> \#DataScience #MachineLearning #Economics #UAE

---

## VARIANT D — Contrarian hook (NEW — recommended if the Medium article is your launch piece)

> **"Housing leads UAE inflation." Every analyst note in Dubai says it. The data disagrees.**
>
> I spent the last month stress-testing that claim while building a forecasting model for UAE CPI. Here's what I found.
>
> Standard Granger causality on stationary monthly data: Housing p = 0.08. Doesn't clear the 5% bar.
>
> Cross-correlation analysis: peak at lag zero. Housing and headline move contemporaneously, not with a lead. Which makes sense once you say it out loud — Housing is 34% of the basket, so when Housing moves, headline moves *the same month* by arithmetic, not by economic lead.
>
> But when I split the sample into high-inflation vs. low-inflation subsamples and re-ran the test — Housing Granger-causes headline in the high-inflation subsample at 2-month lag, p = 0.049. In the low-inflation subsample, p = 0.57.
>
> **Housing isn't a universal lead indicator. It's a regime-conditional one.** It becomes an early-warning signal only during inflationary regimes — exactly when you most need one.
>
> That's a more nuanced finding than the standard narrative, and a more operationally useful one. For a procurement team, the framing matters: you can't trade on "Housing leads inflation" unconditionally. You can trade on "Housing leads inflation, conditional on HDBSCAN assigning us to regime X."
>
> The ML part works too — Ridge regression on 13 sector signals forecasts headline 3 months ahead with **90% directional accuracy, RMSE 1.16 pp, R² 0.75**. Walk-forward backtested across 71 out-of-sample origins. Naive seasonal baseline gets R² of −0.74 on the same test set.
>
> The finding I'm most proud of isn't the forecast number. It's that I rewrote the narrative when the data contradicted what I expected to find.
>
> 📂 Full writeup, code, reproducible pipeline: [GitHub link]
> 📝 Long-form: "The Housing Inflation Myth" on Medium: [Medium link]
>
> MSc Data Science · University of Birmingham Dubai · 2026
>
> \#DataScience #UAE #Economics #MachineLearning #Consulting #Inflation

---



**Image:** Attach `figures/07_forecast_timeline.png`. It's visually striking and shows the result instantly. Second choice: `figures/04_regime_map.png` (the UMAP scatter) — more abstract but signals ML depth.

**Timing:** Tuesday or Wednesday 9–11am Dubai time tends to peak GCC engagement on LinkedIn.

**Tag wisely:** Tag University of Birmingham Dubai, and 2–3 Dubai-based analytics / strategy firms whose attention you want (Bain, BCG, McKinsey Dubai, Oliver Wyman, Deloitte Middle East, PwC Middle East Data & Analytics, ADIA, Mubadala, Kearney Dubai). Don't over-tag.

**First comment:** Drop the GitHub link in the first comment rather than the post body — LinkedIn's algorithm penalizes external links in the main post.

**Reply strategy:** Have 2–3 substantive follow-up points ready. If someone asks "why Ridge and not LSTM?" — you want to be able to say "144 monthly observations, n/p ratio favours linear shrinkage, and I wanted something a hiring manager could debug in their head. The full backtest compares Ridge, XGBoost, and the naive baseline at three horizons."
