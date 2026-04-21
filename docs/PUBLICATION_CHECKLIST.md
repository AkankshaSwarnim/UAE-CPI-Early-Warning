# GitHub & LinkedIn Publication Checklist

## Step 1 — Create the GitHub repo

1. Go to https://github.com/new
2. Repository name: `UAE-CPI-Early-Warning` *(or keep your existing `UAE-CPI-Analysis` — I'd recommend the new name, it's more specific and searchable)*
3. Description: *"Sector-based early-warning system for UAE inflation. Ridge regression forecasts headline CPI 3 months ahead with 90% directional accuracy. Walk-forward backtest, HDBSCAN regime detection, Granger causality."*
4. Public. Add README checkbox = OFF (we have our own).

## Step 2 — Upload the project

From your terminal, in the folder you download from this session:

```bash
cd UAE-CPI-Early-Warning
git init
git add .
git commit -m "Initial release: UAE CPI early-warning system with Ridge forecast + HDBSCAN regimes"
git branch -M main
git remote add origin https://github.com/AkankshaSwarnim/UAE-CPI-Early-Warning.git
git push -u origin main
```

## Step 3 — Verify the repo renders correctly

Open the repo page in a browser and confirm:
- [ ] README displays with all 7 figures inline
- [ ] Figures are visible at full width (not tiny thumbnails)
- [ ] Click through to `notebooks/UAE_CPI_Early_Warning.ipynb` — executed cells with outputs are visible
- [ ] Click `generate_figures.py` — code is syntax-highlighted
- [ ] License shows as "MIT" on the right sidebar

If images don't render: check that `figures/` folder pushed (it should, since there's no `.gitignore` entry for it).

## Step 4 — Add repo topics (critical for discoverability)

On the repo page → "About" gear icon → Topics. Add:
```
data-science · machine-learning · economics · inflation · forecasting · uae
time-series · ridge-regression · hdbscan · granger-causality · python
consulting-case-study · early-warning-system
```

Recruiters search these tags. "consulting-case-study" is the key one for your target.

## Step 5 — Pin the repo on your GitHub profile

Go to your profile page → "Customize your pins" → select this repo. It'll show as the featured project.

## Step 6 — Post on LinkedIn

See `docs/LINKEDIN_POST.md` for three variants. My recommendation:
- Use **Variant A** (punchy consulting voice)
- Attach `figures/07_forecast_timeline.png` as the image
- Post Tuesday or Wednesday, 9–11am Dubai time
- Put the GitHub link in the **first comment**, not the post body
- Tag University of Birmingham Dubai

## Step 7 — After posting: engage for 2 hours

The first 2 hours of engagement determine how far the algorithm pushes the post. When people comment:
- Reply to every comment within 15 minutes
- Ask a follow-up question ("Did you see something similar in the XYZ dataset?") rather than just thanking
- Share the post to 1–2 relevant groups (UAE data community, UoB alumni)

## Step 8 — Follow-up activity

Within a week:
- Write a short Medium article with one specific finding (e.g. "Why I expected Housing to lead UAE inflation — and what the data actually said")
- Link it back to the GitHub repo
- This becomes your second piece of content from the same work

## Common issues and fixes

**"Figures don't show in README when I clone the repo"**
Make sure the figures actually committed. Run `git ls-files figures/` and confirm all 7 PNGs are there.

**"The notebook is too long / takes too long to scroll through on GitHub"**
Fine. GitHub also shows a clickable table of contents in the notebook viewer. The README is the primary entry point anyway.

**"Someone asks why I didn't use LSTM / Prophet / Transformers"**
Your defensible answer:
> "144 monthly observations is too few for deep learning — the n/p ratio favors linear shrinkage. I tested XGBoost (in the repo) and it matched Ridge on directional accuracy but overfit on RMSE. Ridge is the right tool for this problem size. With 5+ years more data, that calculus changes."

**"Someone asks about the honesty caveats"**
Lead with them. That's precisely the signal you want to send to consulting recruiters — you call out the limits of your own work before they do.

---

**Optional: enable GitHub Pages**

If you want the notebook viewable as a webpage:
1. Settings → Pages → Deploy from branch → main → /docs folder
2. Convert the notebook to HTML: `jupyter nbconvert --to html notebooks/UAE_CPI_Early_Warning.ipynb --output-dir docs/`
3. The notebook will be live at `https://akankshaswarnim.github.io/UAE-CPI-Early-Warning/UAE_CPI_Early_Warning.html`
