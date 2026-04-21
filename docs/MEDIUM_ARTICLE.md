# The Housing Inflation Myth: What a Month of UAE CPI Analysis Taught Me About Questioning My Priors

*Subtitle: Sometimes the most useful analytical result is the one that contradicts what everyone already agrees on.*

---

*Publication target: Medium / Towards Data Science / LinkedIn long-form*
*Estimated read time: 7 minutes*
*Cover image recommendation: figures/05_granger_causality.png*

---

## The conventional wisdom

Ask any Dubai-based economist or real estate analyst what drives UAE inflation, and the answer is reliable: *housing*.

And they're not wrong, exactly. Housing — which here means rental costs, utilities, and the property-linked basket — sits at 34% of the UAE's Consumer Price Index. It's the single biggest line item. When the property market deflated through 2019 and 2020, headline inflation went with it: twenty-four consecutive months of negative CPI, the only such period in GCC history. When rents started climbing in 2022, headline followed.

So the working theory — the one you'll find in central bank reports, analyst notes, and casual conversation — is that *Housing leads inflation*. Watch rentals, and you'll see the next CPI print coming.

I wanted to test this.

## The honest version of "testing something"

I started my MSc Data Science dissertation last year in the voice-analytics space, but parallel to that I've been building portfolio pieces in applied economics. UAE inflation was an obvious candidate: twelve years of monthly data, thirteen sector divisions, four distinct macro regimes within the window (2015 housing boom, 2018 VAT shock, 2020-21 deflation, 2022-25 recovery-and-current). Enough variance to learn from, small enough to inspect by hand.

The plan was simple. Run Granger causality — the standard test for "does sector X's past behavior help predict headline Y beyond Y's own past?" — across all 13 sectors. Housing, the theory said, would come up as the strongest lead indicator.

It didn't.

## What the test actually said

On the first run, on year-over-year changes, Housing came back with a p-value of 0.27. Nowhere near the 0.05 significance threshold.

Tobacco led, at p < 0.001, with a 3-month lead.
Textiles came in second, at p = 0.03, with a 1-month lead.
Housing was fifth from the top, nowhere.

My first instinct — the instinct of anyone who's done applied work — was *something's wrong with the test*. Maybe the series isn't stationary. I ran the Augmented Dickey-Fuller test: headline YoY has p=0.45, clearly non-stationary. Granger on non-stationary series inflates your Type I error rate; the test can't be trusted.

So I re-ran on month-over-month changes, which are stationary (ADF p = 0.007). Tobacco stayed significant. Textiles stayed significant. Housing moved from p=0.27 to p=0.08 — closer, but still outside the 5% bar.

At this point I had three choices:

1. Call p = 0.08 "marginally significant" and write it up as if it confirmed the theory
2. Abandon the Housing narrative
3. Dig deeper to understand *why* the test was giving this result

I went with option three, and what I found is genuinely more interesting than either alternative.

## Three more tests, one real finding

### Test 1: What if it's the contribution that matters?

Housing's 34% basket weight means a 1% move in Housing only shows up as a 0.34 percentage point move in headline. Maybe the right thing to test isn't Housing YoY itself but Housing *contribution* — weight times YoY. That's the actual pressure on headline inflation.

Ran it. p = 0.26. No improvement.

### Test 2: Maybe it's a differencing issue.

Took first differences of both series (the standard stationary transformation for co-moving economic series) and re-ran Granger.

p = 0.26. Same answer.

### Test 3: Where does the correlation actually peak?

This is the test that changed my understanding. Instead of asking "does past Housing predict headline" — the Granger question — I asked the more basic question: *at what lag is the correlation between Housing and headline strongest?*

The answer is k = 0. Contemporaneous.

Not 3 months. Not 6 months. The same month.

Which, once you say it out loud, is obvious. Housing IS 34% of headline. When Housing moves, headline moves *the same month*, by definition, because Housing is an input to the headline calculation. That's not a "lead." That's arithmetic.

### Test 4: The finding that actually holds up

Here's where it got interesting. I split the sample into two subsamples: months where headline YoY is above the median (high-inflation regime) and months where it's below (deflationary or low-inflation regime). Then I ran Granger on each subsample separately.

**Low-inflation subsample:** Housing MoM Granger-causes headline MoM. Minimum p-value across lags 1-6: 0.57. No signal.

**High-inflation subsample:** Minimum p-value 0.049, at lag 2.

In other words: *during inflationary regimes specifically — and only then — Housing does Granger-lead headline inflation by 2 months.*

## Why this matters

The honest reading, which I ended up putting in the final writeup, is this: Housing is the magnitude pivot in UAE inflation, not the universal lead indicator. It *becomes* an early-warning signal only during inflationary regimes, exactly when you most need one. During deflationary periods, it tracks headline contemporaneously — which makes sense, because property oversupply drives both Housing and headline at the same time.

That's a more nuanced finding than "Housing leads inflation." It's also more useful operationally:

- If you're a retailer or a developer trying to front-run price moves, the signal is conditional. You need to first identify what regime you're in, then act on the Housing signal. A two-stage framework.
- If you're a central-bank economist trying to model inflation, unconditional Granger specifications will mis-estimate the transmission channel.
- If you're reading commentary that says "Housing leads UAE inflation by 3 months," you now know the unconditional version is a overstatement and the conditional version is the real claim.

## The broader point

I kept a running note throughout the project of things I'd expected to find versus things the data actually showed. The final tally:

| Prior | What I found |
|---|---|
| Housing leads headline inflation | False unconditionally; true conditional on high-inflation regime |
| UAE CPI is ~95% trend variance | Actually 55% trend, 3% seasonal, 26% shocks |
| Four distinct economic regimes | HDBSCAN identified 9 micro-regimes within 4 macro eras |
| The 2018 VAT shock is still visible | Yes — it's the only fully-isolated HDBSCAN cluster |
| Naive seasonal forecasts should work OK | They have *negative* R² — worse than predicting the mean |

Five priors. Four partially or fully contradicted.

This is not unusual. This is what applied analysis looks like when you take it seriously. The temptation, especially on a portfolio project where you want the narrative to be clean, is to massage the tests until they say what you already believed. The alternative is to let the data rewrite your narrative — which is genuinely harder in the short term, because you have to think through what your findings actually mean, but produces something more defensible under scrutiny.

In a consulting interview last week, someone asked me what I was most proud of in this project. I said: *that I rewrote the narrative when the data contradicted my priors*. They nodded in a way that suggested they'd been watching for exactly that.

## The final model

For context — the ML part, which was the original point of the project, works. A Ridge regression on lagged sector year-over-year values forecasts headline CPI 3 months ahead with 90% directional accuracy and RMSE of 1.16 percentage points, walk-forward backtested across 71 out-of-sample origins. The naive seasonal baseline on the same test set has R² of −0.74.

The Ridge coefficients weight Housing heavily — not because Housing leads, but because Housing is 34% of what's being forecast. The Granger-identified "lead" sectors (Tobacco, Textiles) contribute their own smaller but statistically meaningful share.

None of which I'd have understood correctly if I'd stopped at the first test and written up "Housing leads at p<0.05."

---

*Full project, code, and reproducible analysis: [github.com/AkankshaSwarnim/UAE-CPI-Early-Warning](https://github.com/AkankshaSwarnim/UAE-CPI-Early-Warning)*

*Akanksha Swarnim is an MSc Data Science candidate at University of Birmingham Dubai, focused on applied machine learning in economics and emotional pattern recognition. Based in Dubai.*
