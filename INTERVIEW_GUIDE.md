# Interview Guide — Credit Risk Analytics Project

> ⚠️ **The credit-risk numbers in this file are OUTDATED** (66%/85% accuracy came from a model
> with data leakage, fixed June 2026). For the credit-risk project use the refreshed guides:
> **INTERVIEW_GUIDE_DATA_ANALYST.md** and **INTERVIEW_GUIDE_DATA_SCIENCE.md**.
> The Mangrove (§9+) and Review Radar (§14+) sections below are still current.

For a **data analyst internship**. Offline / campus interview, no laptop.

> **Read once the night before. Read the one-page cheat sheet (§0) the morning of. Don't try to read the whole document on the day — you'll panic.**

---

## 0. ONE-PAGE CHEAT SHEET (print this, fold it in your pocket)

### The pitch in 7 beats — memorise the *order*, the words will follow

```
Q  →  D  →  F  →  A  →  D  →  I  →  L

Q  Question:    Can we predict which companies will default?
D  Data:        2,144 real public companies, ~12 industries
F  Features:    Engineered 20+ ratios (leverage, liquidity, profitability, Altman Z)
A  Analytics:   Mapped probabilities to AAA–D credit ratings + aggregated risk score
D  Dashboard:   7 views — overview, companies, industries, ratios, trends, stress, VaR
I  Insight:     ~40% investment grade, ~40% distressed; Restaurants/Retail riskiest;
                Basel severe scenario takes mean default 63% → 88%
L  Learning:    Most of analyst work is feature engineering + presentation, not modelling
```

Drill this until you can recite Q–D–F–A–D–I–L from memory. Each letter unlocks one sentence.

### Numbers to drop into answers

| Number | What it means |
|---|---|
| **2,144** | Rows / companies-periods |
| **20+** | Engineered ratios |
| **66% / 85%** | Logistic Regression / XGBoost accuracy |
| **40% / 40%** | Investment-grade / Distressed split |
| **63% → 88%** | Baseline → Basel severe mean default prob |
| **AAA to D** | 10-bucket S&P rating scale |
| **3** | VaR methods |

### Three ratios you must be able to write on paper

```
Leverage ratio       =  Debt / EBITDA            ( <3 healthy, >5 risky )
Current ratio        =  Current Assets / Current Liabilities  ( >1.5 safe )
Interest coverage    =  EBITDA / Interest Expense ( >3 comfortable, <1.5 dangerous )
```

If they push: **Altman Z-score** combines 5 ratios (working-capital/assets, retained-earnings/assets, EBIT/assets, equity/debt, sales/assets). *Z > 2.99 safe, < 1.81 distress.* You don't need to memorise the coefficients.

### One SQL query you must be able to write

"Top 5 riskiest companies per industry" — the classic window-function question:

```sql
WITH ranked AS (
  SELECT industry, company_id, predicted_proba,
         ROW_NUMBER() OVER (
           PARTITION BY industry
           ORDER BY predicted_proba DESC
         ) AS rn
  FROM scored_companies
)
SELECT industry, company_id, predicted_proba
FROM ranked
WHERE rn <= 5;
```

Walk through it out loud: *"CTE for readability → window function partitions by industry → ranks by default prob descending → outer query keeps the top 5."*

### If you blank mid-answer

1. **Pause. Don't filler-word.** Three seconds of silence is fine. "Um" for ten is not.
2. **Restart the structure.** "Let me back up — the project answers one question: …" Buys you 10 seconds.
3. **Anchor in a number.** Say "2,144 companies across 12 industries" while your brain catches up.
4. **Pivot to what you know.** If asked something you don't know, say so: *"I haven't worked with that specifically, but the closest thing I've done is X."* Honest beats invented every time.

### Offline body language

- **Sit forward, not back.** Engaged posture signals competence even if you're nervous.
- **Eye contact with whoever asked the question**, not the most senior person in the room.
- **Slow down on numbers.** "Two thousand one hundred and forty-four," not "twenty-one forty-four."
- **Hands above the table.** Gesturing while you explain ratios actually makes you more articulate.
- **Bring a notebook + pen.** If they ask you to draw or write SQL, you've already got the tools. Don't ask to borrow.

### What to physically bring

- This guide, printed (just this cheat sheet is fine if pages are limited)
- A small notebook + 2 pens (one always fails)
- One printed page with the **3 ratios + the SQL query** as a fallback you can glance at while waiting
- Your resume, 2 copies

**Do not bring a laptop**, even if allowed — it telegraphs that you can't talk about your work without it.

---

## 1. The pitch (2–3 minutes — your default answer)

Memorise the **structure** Q-D-F-A-D-I-L (from §0), not the words. Speak in short sentences. Pause between sections. Eye contact, not the floor.

> *(Q)* The project I'd like to walk through is a **credit-risk analytics dashboard**. The question it answers is simple: out of a few thousand companies, which ones are likely to default — and *why*?
>
> *(D)* The dataset is around **2,144 companies** with real public financials. Apple, Microsoft, McDonald's, Deere — about a dozen industries: tech, restaurants, manufacturing, retail. The raw data is what you'd see on an income statement and balance sheet — revenue, EBITDA, debt, cash, working capital.
>
> *(F)* The first thing I did was **feature engineering**. In finance that means computing the ratios analysts actually use. Leverage ratios like Debt-to-EBITDA. Liquidity ratios like current ratio and quick ratio. Profitability ratios — EBITDA margin, ROA, ROE. And the **Altman Z-score**, which is a classic bankruptcy-prediction metric. That took the data from about 20 columns to over 40.
>
> *(A)* On top of that I built an analytical layer. I mapped each company's predicted default probability to a credit rating on the **AAA-to-D scale** that S&P and Moody's use. So instead of seeing a probability number, a business user just sees *"this company is BB, this one is D."* I also built an aggregated risk score weighted across financial, operational, and compliance dimensions.
>
> *(D)* The dashboard has seven views. Portfolio overview with KPIs. A searchable company table. Industry benchmarks. Ratio-correlation heatmaps. Time-series trends. **Basel III stress-test scenarios**. And Value-at-Risk analysis.
>
> *(I)* Two findings stood out. The portfolio is barbell-shaped — about **40% investment grade**, about **40% distressed**, very little in between. **Restaurants and retail** are the riskiest sectors on average, **technology** the safest. And under Basel's *severely adverse* scenario the mean default probability climbs from about **63% to nearly 88%**.
>
> *(Tools)* On tooling: **Python with pandas** for the data work. **Plotly** for interactive charts. **SQL** with window functions and CTEs for the time-series and percentile-based benchmarks. The dashboard is a web app, but I also export everything as CSV so the same data flows into **Excel or Power BI**.
>
> *(L)* The biggest thing I learned: most of credit analysis is **feature engineering and presentation**, not modelling. Getting the right ratios right and visualising them so a non-technical reader can act on them — that's the part I want to do more of.

### Shorter variants

**30-second** (if they say "briefly tell me about a project"):

> I built a credit-risk dashboard that scores about 2,100 real public companies for default risk. It computes around 20 standard financial ratios — leverage, liquidity, profitability, Altman Z-score — maps each company to an S&P-style AAA-to-D rating, and runs Basel III stress scenarios. Built in Python with pandas and Plotly, with SQL underneath and a Power BI / Excel export layer.

**1-minute**: trim the pitch above to question + data + ratios + dashboard + one insight. Drop the "what I learned" line.

---

## 2. Key terms — finance

You'll be asked to explain at least two of these. Be ready to define them in **one or two sentences plus a number**.

### Credit risk / default probability
The chance a borrower fails to repay. *Default probability* is the number our model predicts — between 0 and 1.

### EBITDA
Earnings Before Interest, Taxes, Depreciation, and Amortisation. A measure of a company's core operating profitability before the effects of financing decisions and accounting choices. Used because it's comparable across companies with different capital structures.

### Leverage ratio (Debt-to-EBITDA)
How many years of EBITDA the company would need to pay off all its debt. *Under 3× is healthy, above 5× starts to look risky, above 8× is distressed.*

### Debt-to-Equity ratio
Total debt divided by total equity. Tells you how much the company is funded by borrowing vs by shareholders. *Above 2 is high leverage.*

### Current ratio
Current assets ÷ current liabilities. Can the company pay its short-term bills? *Above 1.5 is safe, below 1.0 is a warning sign.*

### Quick ratio
Same idea as current ratio but excludes inventory — only counts cash, receivables, and short-term investments. Stricter test of liquidity.

### Interest coverage ratio
EBITDA ÷ interest expense. How many times over can the company cover its interest payments. *Above 3× is comfortable, below 1.5× is dangerous.*

### EBITDA margin / net margin
EBITDA (or net income) ÷ revenue. How much of every dollar of sales becomes profit. *Tech companies routinely hit 30%+, restaurants are often under 10%.*

### ROA / ROE
Return on Assets / Return on Equity — net income divided by total assets, or by shareholders' equity. How efficiently a company generates profit from what it owns / what shareholders have put in.

### Altman Z-score
A bankruptcy-prediction formula from 1968 that combines five financial ratios (working capital/assets, retained earnings/assets, EBIT/assets, equity/debt, sales/assets) into a single score.
- **Z > 2.99** → safe zone
- **1.81 < Z < 2.99** → grey zone
- **Z < 1.81** → distress zone (high probability of bankruptcy within 2 years)

### Credit rating (AAA to D)
Letter grade for creditworthiness, same scale as S&P, Moody's, Fitch.
- **AAA–BBB** = **investment grade** (low risk, big institutions can buy the bonds)
- **BB–C** = **speculative / "junk"**
- **D** = in default

### Basel III
International banking regulation framework from the Basel Committee. Requires banks to hold enough capital to survive stress scenarios. *Adverse* and *Severely Adverse* are the two standard stress scenarios — recession, credit losses, liquidity tightening. My project includes both.

### IFRS 9
Accounting standard for financial instruments. The relevant piece is **Expected Credit Loss (ECL)** — banks must reserve for losses *before* they happen, based on forward-looking default probabilities. My project simulates an IFRS-9 impairment scenario.

### Stress testing
Running the portfolio through hypothetical bad scenarios (e.g. 30% revenue collapse, debt costs +25%) to see how risk scores change. Banks are required to do this regularly.

### Value at Risk (VaR)
The maximum loss you'd expect over a given period at a given confidence level. *"5% VaR of 2% on daily returns"* means: on the worst 5% of days, you'd lose at least 2% of portfolio value.

Three ways to calculate it:
- **Parametric (variance-covariance)** — assumes returns are normally distributed. Fast but misses fat tails.
- **Historical** — uses the actual past return distribution. No assumptions, but needs enough history.
- **Monte Carlo** — simulates thousands of scenarios. Most flexible, most expensive.

### Investment grade vs distressed
**Investment grade** (AAA–BBB) = institutional investors can hold. **Distressed** (CCC–D) = high default risk, prices reflect that.

---

## 3. Key terms — data / ML

### Feature engineering
Creating new columns from raw data to make patterns easier for a model (or a human) to spot. In this project: turning raw debt and EBITDA into a leverage ratio, raw revenue and net income into a margin, etc.

### Logistic Regression
A simple, interpretable classification model. Output is a probability between 0 and 1. Used here as a **baseline** because it's transparent — every coefficient maps to a single feature, so auditors can trace exactly how a prediction was made.

### XGBoost
A gradient-boosted tree model. Builds many small decision trees, each correcting the previous one's mistakes. Usually wins ML competitions on tabular data. Used here as the **production model** because it's significantly more accurate.

### Train/test split
Holding back ~20% of the data so you can measure how well the model performs on companies it hasn't seen. Prevents "memorising the answer key."

### Accuracy / precision / recall
- **Accuracy** = % of predictions correct overall
- **Precision** = of the companies we flagged as high-risk, how many really were
- **Recall** = of the companies that really were high-risk, how many did we catch

For imbalanced data (which credit risk is — most companies *don't* default), precision and recall matter more than raw accuracy.

### Classification report
The sklearn output that gives you precision, recall, and F1-score per class.

### Pipeline (sklearn)
A wrapper that chains preprocessing + model into a single object. Means the same transformations apply automatically at inference, you can't accidentally forget to scale a feature.

### SHAP
**SH**apley **A**dditive ex**P**lanations. A method from cooperative game theory that tells you how much each feature contributed to a specific prediction. Important in regulated industries — you can show *why* a company was flagged as high-risk, not just that it was.

### Correlation matrix / Pearson correlation
A grid showing how strongly each pair of variables moves together. Values from −1 (move opposite) to +1 (move identically). In my dashboard the ratio-correlation heatmap shows, for example, that quick ratio and current ratio are 0.92 correlated — which makes sense, they measure similar things.

### Time-series aggregation
Grouping data by a date column and computing summary stats per period — mean default probability per year, per quarter, etc.

---

## 4. Key terms — tech (light, in case they ask)

### Pandas
The Python library for tabular data. Think Excel but programmable.

### Plotly
JavaScript-based interactive charting library. Hover tooltips, zoom, pan — without writing custom JavaScript.

### SQL window functions
SQL functions that operate on a "window" of related rows — for example, computing a rolling 3-period average without collapsing the rows down. Used in this project for rolling risk trends.

### CTE (Common Table Expression)
A `WITH … AS (…)` block in SQL. Lets you break a complex query into named, readable steps instead of one giant nested subquery.

### PERCENTILE_CONT
SQL function for continuous percentiles — used here to compute, say, the 75th-percentile leverage in each industry as a benchmark.

### FastAPI
A modern Python web framework. I use it to expose the model's output as REST endpoints (JSON and CSV). You probably don't need to mention it unless they push you on how the dashboard works.

### MLflow
Tool for tracking ML experiments — logs hyperparameters, metrics, and trained models from every run so you can compare them. (Only mention if asked about reproducibility.)

---

## 5. Common questions and how to answer

### About the project itself

**"Walk me through the project."**
→ Deliver the 2–3 minute pitch from §1. Practise it out loud.

**"What was the most challenging part?"**
→ "Data quality. Many of the financial ratios have a divide-by-zero problem — a company with no debt has infinite interest coverage, a company breaking even has an undefined margin. I had to add safe-division logic everywhere and decide whether to clip extreme values or drop them. That cleanup took longer than building the dashboard itself."

**"What would you do differently?"**
→ "Two things. First, the risk-flag labels in the dataset are partly synthetic — in production I'd want to train against actual historical default events from a credit bureau. Second, I'd add a proper data-validation layer up front so bad rows are caught before they hit the feature pipeline."

**"What did you learn?"**
→ "The thing I didn't expect: how much of credit analysis is presentation, not modelling. I spent days on the dashboard layout, conditional colour-coding, ratio definitions — making the output usable by a non-technical reader. The actual model training is a few lines of code. That re-shaped how I think about analyst work."

**"Why this project?"**
→ "I wanted to combine something domain-specific — finance — with the data and Python skills I'd been building. Credit risk has clear business logic, well-defined ratios, and public data, so I could check my work against industry benchmarks instead of guessing."

**"What's the business value?"**
→ "A credit officer at a bank could use this to triage a portfolio of borrowers — see which industries are most exposed, which individual companies are sliding into distress, and how the book would look under a recession scenario. Today that work happens across multiple spreadsheets and tools; this consolidates it."

### About the data

**"Where did the data come from?"**
→ "Real public financial filings — companies like Apple, Microsoft, Walmart — combined with synthetic operational-incident counts because those aren't publicly disclosed at company level."

**"How big is it?"**
→ "About 2,144 rows across roughly a dozen industries. Small by ML standards but realistic for a portfolio analyst — a single bank's commercial-lending book."

**"How did you clean it?"**
→ "Three steps: enforce a schema check up front so missing columns fail loudly; coerce types — dates to datetime, numerics to floats; and add `+ 1e-6` denominators on every ratio to prevent divide-by-zero. Anything that came out as ±infinity got clipped to a sensible range for charting."

### About the model

**"How accurate is the model?"**
→ "XGBoost hits about 85% test accuracy, Logistic Regression about 66%. But accuracy can mislead with imbalanced classes — I look at the full classification report with precision and recall."

**"Why two models?"**
→ "Logistic Regression is the interpretable baseline — every coefficient maps directly to a feature, which is what auditors and regulators want. XGBoost is the production model because it's much more accurate. Having both gives stakeholders a choice depending on whether they need transparency or performance."

**"How do you handle imbalanced data?"**
→ "I check precision and recall separately rather than rely on overall accuracy. XGBoost also has a `scale_pos_weight` parameter you can tune if you need to push the model toward recall. I haven't used SMOTE here but I know it's an option for upsampling the minority class."

### About specific ratios

**"Explain the Altman Z-score."**
→ "It's a bankruptcy-prediction formula from 1968 that combines five ratios into one number. Above 2.99 the company is safe, below 1.81 it's in the distress zone. In my dashboard I've got a histogram with dashed lines at both thresholds so you can see what proportion of the portfolio is in each zone."

**"Why current ratio *and* quick ratio?"**
→ "Both measure short-term liquidity, but quick ratio is stricter — it excludes inventory. So if a manufacturer has a healthy current ratio but a weak quick ratio, you know it's only liquid on paper because of unsold stock. The correlation between them in my dataset is about 0.92, so they usually agree, but the gaps are where it gets interesting."

### About SQL

**"How did you use SQL?"**
→ "Three patterns mostly. Window functions for rolling time-series — `AVG() OVER (PARTITION BY company_id ORDER BY date)` for trend lines. CTEs to keep multi-step aggregations readable. And `PERCENTILE_CONT` for industry benchmarks — for each ratio I compute the 25th, 50th, and 75th percentile per industry so each company can be plotted against its peer group."

**"Write a query to find the top 5 riskiest companies per industry."**
→ "I'd use `ROW_NUMBER()` partitioned by industry:
```sql
WITH ranked AS (
  SELECT industry, company_id, predicted_proba,
         ROW_NUMBER() OVER (PARTITION BY industry ORDER BY predicted_proba DESC) AS rn
  FROM scored_companies
)
SELECT * FROM ranked WHERE rn <= 5;
```
The CTE keeps it readable; the window function does the ranking inside each industry."

### About visualisation

**"Tell me about one chart you built."**
→ Pick the ratio-correlation heatmap. *"It's a 12×12 grid showing how strongly each financial ratio moves with every other one. Colour goes from red (negative correlation) through neutral to green (positive). The interesting cells are the unexpected ones — like predicted default probability being only weakly correlated with leverage on its own, because the model has learned the interaction effects across multiple ratios, not just one."*

Alternatively: the leverage-vs-liquidity scatter, where each dot is a company, the size is default probability, and the colour is the credit rating. Tells the whole story of the portfolio in one chart.

**"Have you used Excel or Power BI?"**
→ "Yes — I designed this project specifically so an analyst could pull the same data into either tool. Every endpoint has a CSV variant. Excel via *Data → From Web*, Power BI via *Get Data → Web*. I also wrote a setup guide for connecting Power BI Desktop to the API."

### Behavioural

**"Tell me about a time something didn't work."**
→ "Early on the dashboard styling was completely broken — I'd used a CSS technique that depends on a build step, but I was loading the styling library straight from a CDN with no build step. The tabs and cards just showed up as plain text. Took me a while to realise the framework was silently dropping all my custom rules. Once I figured it out I rewrote the styles in plain CSS — slightly more verbose but it actually works, and now I know to test the styling on a fresh browser before assuming it's done."

**"How do you decide what to prioritise?"**
→ "I start with the question I'm trying to answer, then work backwards to what data I need. In this project the question was *which companies are likely to default,* so the first hour was checking that the dataset had a default flag and enough financial detail to compute the standard ratios. Everything after that — the model, the dashboard, the stress tests — only makes sense once that data foundation is in place."

**"Why do you want this internship?"**
→ Don't fake it. A version of: "I've been doing self-directed projects like this one, but I want to learn what analyst work looks like inside a real team — how problems get scoped by stakeholders, how findings get presented up the chain, what data sources you actually have access to in industry."

---

## 6. Numbers to memorise

Drop these specifics during the pitch — they make you sound credible.

| Number | What it is |
|---|---|
| **2,144** | Rows / company-period observations |
| **~12** | Distinct industries |
| **20+** | Financial ratios engineered |
| **66% / 85%** | LR / XGBoost test accuracy |
| **~40%** | Share of portfolio that's investment grade (AAA–BBB) |
| **~40%** | Share of portfolio that's distressed (CCC–D) |
| **~63% → ~88%** | Mean default prob, baseline → Basel severely adverse |
| **AAA to D** | Rating scale (10 buckets) |
| **3** | VaR methods (parametric, historical, Monte Carlo) |

---

## 7. What *not* to say

- **Don't oversell.** Say "production-ready architecture," not "deployed in production." Say "designed to scale," not "scales to millions of users."
- **Don't bluff numbers.** If you don't remember the exact XGBoost accuracy, say "around 85%" — never invent a number.
- **Don't list every library.** Mention 4–5 with intent (pandas, scikit-learn, XGBoost, Plotly, SQL). A buzzword dump signals insecurity.
- **Don't pretend the labels are real.** If asked, be straight: "The default-risk labels are partly synthetic — in production I'd train against actual historical defaults."
- **Don't lead with the tech stack.** Lead with the *question*. Analysts answer questions; engineers ship stacks.

---

## 8. Closing line if asked "anything else?"

> The piece I'd want to extend in a real role: connect this to a feature store and a model registry so the same engineered ratios serve both the dashboard and any other downstream tool — credit memo generation, an early-warning alerting system, a regulatory report. Right now everything's in one process; the natural next step is making it composable.

That's a *senior-thinking* close to a *junior* interview and lands well.

---

Good luck. Read it twice; don't memorise verbatim, internalise the structure. The interviewer wants to see that *you understand the project*, not that you can recite a script.

---
---

# PROJECT 2 — Mangrove Monitoring & Loss Prediction System

## 9. Cheat sheet (Q-D-F-A-D-I-L)

```
Q  Question:    Where is mangrove forest loss happening, and can we predict it
               to support conservation decisions?
D  Data:        JAXA Global Mangrove Watch v3.0 satellite imagery —
               65,536 pixels (256×256 study grid), 9 ecological features per pixel
F  Features:    9 satellite-derived ecological signals (NDVI, elevation, canopy,
               proximity to water, etc.); engineered + SMOTE-balanced
A  Algorithm:   XGBoost classifier — 96% accuracy, 0.931 AUC
               Validated with Spatial Block Cross-Validation, not random CV
D  Dashboard:   15-endpoint FastAPI REST API + Next.js 14 / TypeScript frontend
               with Leaflet.js maps; SHAP waterfall plots per prediction
I  Impact:      Quantified ecosystem services at ~INR 2.57 lakh/ha/year and
               carbon stock at 3,302 tCO₂/ha for evidence-based restoration
L  Learning:    Spatial data needs spatial validation — random splits leak signal
               because neighbouring pixels are correlated
```

## 10. The pitch (2–3 minutes)

> *(Q)* The second project is an **end-to-end geospatial machine-learning system for predicting mangrove forest loss**. Mangroves are one of the most efficient carbon sinks on the planet, but they're disappearing globally. The question was: can we predict *where* loss is most likely so conservation budgets get allocated to the right places?
>
> *(D)* The data comes from **JAXA's Global Mangrove Watch v3.0**, which is the gold-standard satellite product for mangrove extent and change. I worked on a **256×256 pixel study grid — 65,536 pixels total** — each with **nine ecological features** like NDVI, elevation, proximity to water, canopy density.
>
> *(F)* The interesting problem in this kind of geospatial data is **spatial autocorrelation** — pixels that sit next to each other on a map are almost always correlated. If you do a random train-test split, your "test set" contains pixels that are literally next to pixels in your training set, so your accuracy looks great but you're really just memorising local patterns. So instead of random cross-validation I used **Spatial Block Cross-Validation** — divide the map into geographic blocks and hold out *entire blocks* for validation. That gives you an honest accuracy estimate. I also used **SMOTE** to handle class imbalance, because the "loss" class was the minority.
>
> *(A)* The model is an **XGBoost classifier**. It hits **96% accuracy and a 0.931 AUC** on held-out blocks, which means it's very well-calibrated — AUC is the more honest metric on imbalanced data. I used **SHAP TreeExplainer** for explainability — for every individual prediction, the dashboard shows a **waterfall plot** of which features pushed the probability up or down. That was important because conservation stakeholders aren't machine-learning people; they need to see *why* the model flagged a parcel.
>
> *(D)* The deployment side: **FastAPI backend with 15 endpoints**, **Pydantic validation** on the inputs, structured logging, Docker-packaged. The frontend is a **Next.js 14 dashboard in TypeScript** with **Leaflet.js maps**, so a user can zoom around the grid and click any pixel to see its prediction, the SHAP explanation, and the underlying features.
>
> *(I)* The applied piece: I tied the model output to ecological-economics literature to **quantify ecosystem services at about INR 2.57 lakh per hectare per year** and **carbon stock at 3,302 tonnes of CO₂ per hectare**. That makes the model's output actionable for conservation planning — you can prioritise the hectares that maximise carbon and ecosystem value.
>
> *(L)* The biggest takeaway: **the choice of cross-validation strategy matters more than the choice of model**. With random CV the same model showed near-perfect numbers that were completely misleading. Switching to spatial blocks dropped apparent accuracy but produced numbers I could actually trust.

## 11. Key terms — be ready to explain these

### Spatial autocorrelation
The tendency of nearby locations to have similar values. *Tobler's first law of geography: "everything is related to everything else, but near things are more related than distant things."* It's why random train-test splits on spatial data give artificially high accuracy.

### Spatial Block Cross-Validation
Instead of random k-fold, you divide the map into geographic blocks (squares, hexagons, or natural regions) and hold out entire blocks for validation. Forces the model to generalise to genuinely unseen geography.

### SMOTE (Synthetic Minority Over-sampling Technique)
A method to fix class imbalance. Instead of duplicating minority-class samples, SMOTE generates *new* synthetic ones by interpolating between existing minority points in feature space. Better than naive over-sampling because it doesn't just memorise the same rows.

### AUC (Area Under the ROC Curve)
A number between 0.5 (random) and 1.0 (perfect) that summarises classifier performance across all decision thresholds. **0.931 is excellent.** AUC is more honest than raw accuracy on imbalanced datasets — accuracy can be misleadingly high if you just predict "no loss" for everything.

### NDVI (Normalised Difference Vegetation Index)
A satellite-derived index of vegetation health, computed from red and near-infrared bands. Values from −1 to +1; healthy dense vegetation is +0.6 to +0.9.

### SHAP TreeExplainer
A fast, exact SHAP implementation specifically for tree-based models (Random Forest, XGBoost, LightGBM). For each prediction, it tells you how much each feature contributed.

### Waterfall plot
A visualisation that shows how a model arrived at a single prediction. Starts at the baseline (average prediction), then each feature pushes the prediction up or down until the final value. SHAP's standard per-prediction view.

### Pydantic
A Python library for runtime data validation and parsing. In FastAPI, you declare request/response shapes as Pydantic models and validation is automatic — bad input gets a clean 422 error instead of crashing your handler.

### Leaflet.js
An open-source JavaScript library for interactive maps. Lightweight (~40 KB), works with OpenStreetMap tiles by default, great for thematic mapping.

### tCO₂/ha
Tonnes of CO₂ per hectare — the standard unit for measuring carbon stock in an ecosystem. Includes above-ground biomass (trees), below-ground biomass (roots), and soil carbon.

### Ecosystem services
The benefits humans get from ecosystems — for mangroves: carbon sequestration, coastal storm protection, fish nursery habitat, water filtration. Often monetised per hectare per year for policy work.

## 12. Common questions — Mangrove project

**"Why XGBoost and not a neural network or CNN?"**
→ "Two reasons. First, the feature set is tabular — nine engineered ecological features per pixel, not raw imagery — so a tree model is a natural fit. Second, with 65,536 pixels we're not in a data regime where deep learning wins. XGBoost is faster to train, better-calibrated out of the box, and TreeExplainer gives me exact SHAP values, which I needed for the explainability layer."

**"What's spatial autocorrelation and why did it matter?"**
→ "Nearby pixels on a map are highly correlated — if one pixel has dense canopy, its neighbours usually do too. With random cross-validation, the test set ends up next to the training set, so the model gets credit for memorising local patterns. I switched to Spatial Block CV — holding out whole geographic blocks — which gives you an honest estimate of how well the model generalises to new geography."

**"Why SMOTE specifically? Why not class weights?"**
→ "Both are valid. SMOTE creates synthetic minority samples by interpolating in feature space, so the decision boundary has more data to learn from. Class weighting just amplifies the loss on minority samples without adding new examples. For tree models, SMOTE often gives better recall on the minority class. I'd benchmark both in production but SMOTE was the cleaner first pass."

**"What does an AUC of 0.931 mean?"**
→ "If I picked one positive (loss) pixel and one negative (no-loss) pixel at random, the model would rank the positive one higher 93% of the time. It's a much more honest metric than accuracy on imbalanced data, because accuracy alone could be high by just predicting 'no loss' for everything."

**"What were the most important features?"**
→ Have a real answer ready — look at your SHAP summary plot. A safe answer: *"From the SHAP summary, **proximity to water** and **elevation** dominated — which makes ecological sense, because mangroves grow in intertidal zones at low elevation. NDVI trend over time was also a strong signal — declining greenness foreshadows loss."* (Adjust to what your actual plot showed.)

**"How did you validate the ecosystem-services valuation?"**
→ "I anchored the numbers to peer-reviewed estimates from the ecological-economics literature — the standard reference is the **Costanza et al.** valuation of global ecosystem services, with mangrove-specific adjustments from more recent studies. I'm not claiming I derived those numbers from scratch; I applied published per-hectare valuations to the predicted loss area."

**"Why did you build it as a full-stack app rather than a notebook?"**
→ "Conservation stakeholders aren't going to read a Jupyter notebook. The point of the dashboard is that a non-technical user — a forestry officer or an NGO — can zoom into their region of interest, click a pixel, and immediately see both the prediction and the *reason*. The waterfall plot is what makes the model trustable for people outside ML."

**"How many endpoints and what do they do?"**
→ "Fifteen total — grouped into prediction endpoints (single pixel, batch grid), data endpoints (features, metadata, time-series), explainability (SHAP per-prediction), and metrics (model performance, feature importance). The split is deliberate — frontend pages each hit just the endpoints they need."

**"What did you struggle with?"**
→ "The cross-validation strategy. The first version had random k-fold and showed something like 99% accuracy, which I knew was too good to be true. Reading more, I found out about spatial autocorrelation in geospatial ML literature, switched to block CV, and the realistic numbers came out. That was the moment I understood why people say *the validation strategy IS the experiment*."

## 13. Resume bullet review — Mangrove

Your current bullets are strong. Two micro-improvements:

- **Bullet 1** is excellent — keep as is.
- **Bullet 2** — consider tightening: *"Addressed class imbalance with SMOTE and prevented spatial autocorrelation leakage using Spatial Block Cross-Validation."* (Active verb at the front.)
- **Bullet 4** — you currently say *"15-endpoint FastAPI REST API"*. Consider adding throughput or response time if you measured them: *"…serving predictions in <100 ms"*. Numbers help.
- **Bullet 5** — ecosystem-services and carbon-stock numbers are great. **Double-check you can defend the source** (Costanza-style valuation, IPCC carbon factors) — if asked "where did 3,302 tCO₂/ha come from?" you must have a real answer.

---
---

# PROJECT 3 — Review Radar (NLP / sentiment analysis)

## 14. Cheat sheet (Q-D-F-A-D-I-L)

```
Q  Question:    Can we automatically tell what customers like, dislike, and lie about
               in product reviews — at a scale a human analyst couldn't read manually?
D  Data:        Product reviews (text + star rating)
F  Features:    Token-level via DistilBERT for sentiment;
               TF-IDF + topic modelling for aspect extraction;
               linguistic signals for fake-review detection
A  Algorithm:   DistilBERT for sentiment (~92% accuracy);
               TF-IDF + topic modelling for aspect mining;
               classifier on linguistic features for fake reviews
D  Dashboard:   Modular preprocessing + inference components — built for rapid
               experimentation rather than a polished frontend
I  Impact:      End-to-end pipeline that turns unstructured review text into
               structured product insight + a noise-reduced dataset
L  Learning:    Most NLP value comes from preprocessing and aspect extraction,
               not the sentiment model itself
```

## 15. The pitch (2–3 minutes)

> *(Q)* The third project is **Review Radar**, an NLP pipeline for analysing product reviews. The question: how do you extract useful business insight from thousands of unstructured reviews, when no human is going to read them all?
>
> *(D)* The data is product reviews — text plus star rating — and the challenge is that the text says a lot more than the rating. A 5-star review can still mention shipping problems; a 2-star review can rave about the product itself.
>
> *(F)* I broke the problem into three parts. First, **sentiment classification** — is the overall review positive or negative — using **DistilBERT**, which is a smaller, faster version of BERT that keeps about 97% of BERT's performance with about 40% fewer parameters. I got around **92% sentiment accuracy** on the test set. Second, **aspect extraction** — what product features are being talked about — using **TF-IDF combined with topic modelling** to surface the high-impact terms and clusters. Third, **fake-review detection** to clean up the dataset before downstream analysis.
>
> *(A)* The pipeline is **modular** — separate preprocessing, sentiment-inference, aspect-extraction, and fake-detection components — so I could swap a piece without rebuilding the whole thing. That made experimentation fast.
>
> *(I)* The output is a structured view of an unstructured corpus — instead of "23 negative reviews," you can say *"23 negative reviews, mostly about delivery delays and packaging, mostly genuine, two flagged as suspicious."* That's the difference between a number and an insight.
>
> *(L)* The big lesson: **the sentiment model is the easy part**. Pre-trained transformers are essentially commodities now. The work that creates value is the aspect extraction and the data-cleaning — figuring out *what* customers are talking about and filtering out fake noise. That's what turns a sentiment score into something a product team can act on.

## 16. Key terms — be ready to explain these

### BERT and DistilBERT
**BERT** (Bidirectional Encoder Representations from Transformers) is a 2018 transformer-based language model from Google. It reads a sentence in both directions at once, giving rich context-aware token representations.

**DistilBERT** is a "distilled" smaller version — trained to mimic BERT but with ~40% fewer parameters and 60% faster inference. Retains around 97% of BERT's performance on common tasks. Used widely in production where latency matters.

### Knowledge distillation
The training technique that produces DistilBERT. A small "student" model is trained to match the output probabilities of a large "teacher" model. Lets you compress models without retraining on raw labels.

### TF-IDF (Term Frequency × Inverse Document Frequency)
A weighting scheme for words. **TF** = how often a word appears in this document. **IDF** = how rare the word is across all documents. Multiplying them highlights words that are *characteristic* of a document — common in that doc, rare overall. Used here to surface aspect words ("battery", "shipping", "screen") that distinguish reviews from each other.

### Topic modelling
Unsupervised algorithms that cluster words into "topics" based on which words co-occur in documents. The classic algorithm is **LDA (Latent Dirichlet Allocation)** — it assumes each document is a mix of topics and each topic is a distribution over words. Useful for finding the natural themes in a review corpus without labelling.

### Aspect-based sentiment analysis
Sentiment scoring not for the whole review but for individual *aspects* — battery life: negative, screen: positive, shipping: negative. Much more actionable than document-level sentiment.

### Fake review detection
Classifying which reviews are suspicious. Signals: extreme polarity language, repeated phrases across reviewers, reviewer history, posting bursts, mismatch between text sentiment and star rating, copy-paste detection.

### Fine-tuning vs zero-shot
**Fine-tuning** means continuing to train a pre-trained model on your specific task data. **Zero-shot** means using the pre-trained model without further training. Be ready to say which one you did for DistilBERT. (Most likely you fine-tuned on a labelled sentiment dataset.)

### Tokenization
Splitting text into pieces ("tokens") the model can ingest. BERT-family models use **WordPiece** — words can be broken into subwords (`unhappily → un + happi + ly`), which lets the model handle rare or compound words.

## 17. Common questions — Review Radar

**"Why DistilBERT and not full BERT?"**
→ "Speed and memory. DistilBERT keeps about 97% of BERT's accuracy with 40% fewer parameters. For a pipeline that has to score thousands of reviews, that's the right trade-off. I'd revisit if I had latency budget — for instance, RoBERTa-large would be a small accuracy bump at a big compute cost."

**"Did you fine-tune DistilBERT or use it pre-trained?"**
→ *Have a real answer.* If you fine-tuned: "I fine-tuned on a labelled sentiment dataset — added a classification head, trained for a few epochs with a small learning rate. The pre-trained weights handle language understanding; the fine-tuning adapts to our domain." If you used a pre-trained sentiment model directly: "I used a pre-trained sentiment model and validated against a held-out test set." **Don't bluff.**

**"How does TF-IDF actually work?"**
→ "It scores each word in each document with two parts. **TF** is the frequency of the word in that document. **IDF** is the log of total documents divided by documents containing the word — so a word that's everywhere gets a low IDF, a rare word gets a high one. Multiplied, the score is high for words that are common *in this doc* but rare overall. That's exactly the kind of word that characterises the document."

**"What topic-modelling algorithm did you use?"**
→ *Have a real answer.* If LDA: "I used LDA — Latent Dirichlet Allocation. Each document is a mixture of topics, each topic is a distribution over words. I tried a few values of *k* (number of topics) and picked the one where the topics were most interpretable." If it was something else like NMF or BERTopic, be honest.

**"How did you detect fake reviews?"**
→ "I used a combination of signals: linguistic features like extreme polarity and repeated phrasing, distributional features like text-rating mismatch (a 5-star review with negative text), and reviewer-level features where available. Then a classifier on top of those features." *(Adjust to what you actually did. If you used an off-the-shelf fake-review dataset, say so honestly.)*

**"How did you measure 92% accuracy?"**
→ "On a held-out test split of labelled reviews — standard train/validation/test split. I also looked at precision and recall per class because positive and negative classes are usually imbalanced in review datasets."

**"What's the limit of your approach? Where would it fail?"**
→ "Sarcasm and negation are the classic failure modes of transformer sentiment models — *'great, just what I needed, another broken charger'* gets read as positive. They also struggle when a review mixes positive and negative aspects in one sentence — that's why I'd argue aspect-based extraction is where the next iteration goes."

**"Why is fake review detection important?"**
→ "Because downstream analytics inherit the noise. If 15% of your reviews are fake but you don't know it, your sentiment averages are skewed, your topic models pick up the fake patterns, and your product team is making decisions based on partly fabricated data. Removing or flagging them improves every downstream step."

**"How would you scale this to millions of reviews?"**
→ "Three changes. Run DistilBERT inference on a GPU and batch the inputs. Move the topic modelling from full retraining to incremental updates. And cache the embeddings so I'm not re-encoding the same reviews on every run. Beyond that, distribute the preprocessing across workers — preprocessing is embarrassingly parallel."

## 18. Resume bullet review — Review Radar

The bullets are clean, but they're vaguer than the Mangrove ones. Two notes:

- **Bullet 2** is light on specifics. *"Extracted high-impact product aspects using TF-IDF and topic modeling"* — strengthen by naming the algorithm and the corpus size: *"…using TF-IDF and LDA topic modelling across N product reviews."*
- **Bullet 3** — "Implemented fake review detection" is doing a lot of work. Add the *how*: *"Implemented fake review detection using linguistic and behavioural signals, removing ~X% of suspicious entries."* If you don't have a percentage, omit; otherwise it strengthens the bullet a lot.

---
---

# 19. Multi-project strategy — which one do you lead with?

You have three projects. The interviewer will usually say "tell me about a project" — which means **you pick**. Choose deliberately:

| Lead with… | When |
|---|---|
| **Credit risk** | The role mentions finance, banking, analytics, dashboards, BI tools, Excel/Power BI |
| **Mangrove** | The role mentions ML engineering, deployment, geospatial, sustainability, full-stack, FastAPI/React |
| **Review Radar** | The role mentions NLP, customer analytics, product analytics, e-commerce |

If the role doesn't lean any specific way, **lead with credit risk**. It's the most "data analyst" — heavy on ratios, SQL, dashboarding, business interpretation — and the bullets are the strongest in that direction.

### When they ask "any other projects?" after your first one

Pick the one that contrasts most with what you just discussed:

- Led with credit risk → next mention **Mangrove** (shows you also do production ML and full-stack)
- Led with Mangrove → next mention **credit risk** (shows you also do business-facing analytics)
- Either way → keep **Review Radar** as the third — it's the most generic NLP project and the easiest to summarise quickly

### A 30-second summary of each (for the "anything else?" moment)

> **Credit risk:** "A credit-risk analytics dashboard scoring 2,144 real public companies on default probability — feature-engineered 20+ ratios, XGBoost classifier, AAA-to-D rating mapper, plus Basel III stress tests and VaR analysis."

> **Mangrove:** "A geospatial ML system for predicting mangrove loss — XGBoost on JAXA satellite features, 96% accuracy / 0.931 AUC validated with Spatial Block CV, served via a FastAPI + Next.js dashboard with SHAP explainability per pixel."

> **Review Radar:** "An NLP pipeline for product reviews — DistilBERT for sentiment at 92% accuracy, TF-IDF and topic modelling for aspect extraction, plus a fake-review detection layer to clean the dataset."

Practise saying all three in under 30 seconds each. If you can rattle through three projects in 90 seconds total, you sound *prepared*, not just lucky.

---

# 20. Honest-defence audit — read this carefully

Some bullets on the resume are technical claims that an interviewer will probe. **For each of these, make sure you have a real, concrete answer before the interview** — don't get caught bluffing.

### Mangrove

- **"Spatial Block Cross-Validation"** — be ready to explain *why* (autocorrelation) and *how* (geographic blocks, not random rows). If you didn't actually implement this and only read about it, say "I split by geographic region for validation" instead.
- **"SMOTE"** — know the alternatives (class weighting, undersampling) and why you chose SMOTE.
- **"INR 2,57,500/ha/year"** — know the source. Costanza et al. is the standard reference; admit if you're applying published per-hectare values rather than deriving them.
- **"3,302 tCO₂/ha"** — same. Carbon-stock figures usually come from IPCC factors or peer-reviewed local studies.
- **"15-endpoint FastAPI"** — be ready to list 4–5 of the endpoints. Don't say a number you can't back up.

### Review Radar

- **"92% accuracy"** — on what dataset, what split? Know the answer.
- **"DistilBERT"** — did you fine-tune it or use a pre-trained sentiment head? Know the answer.
- **"Topic modelling"** — which algorithm? LDA? NMF? BERTopic? Know the answer.
- **"Fake review detection"** — what signals, what classifier? This is the bullet most likely to draw a probing follow-up.

If you can't defend a number or a technique, **soften the bullet now**. Better to say "applied published valuations" than to invent a methodology under pressure.

---

# 21. Updated 3-day plan now that you have 3 projects

The original STUDY_PLAN.md schedules Day 2 morning for pandas. With three projects to drill, that needs to shift:

| Day | Old plan | Updated plan |
|---|---|---|
| Day 1 morning | SQL cheat sheet | SQL cheat sheet (unchanged) |
| Day 1 afternoon | SQL practice problems | SQL practice problems (unchanged) |
| **Day 2 morning** | Pandas cheat sheet | **Drill 3 project pitches** — recite each Q-D-F-A-D-I-L out loud 3× |
| **Day 2 midday** | Resume rewrite | **Resume rewrite** (unchanged, but covers all 3 projects) |
| **Day 2 evening** | Behavioural prep | Pandas cheat sheet — *abbreviated, just SQL→pandas table* |
| Day 3 morning | Pitch practice | Mock yourself on any of the 3 projects (let dice/chance pick) |
| Day 3 afternoon | Weak-spot review | Weak-spot review |
| Day 3 evening | Rest | Rest |

Behavioural prep gets compressed — pick **one** story (the styling bug works for credit risk; the spatial-CV realisation works equally well for Mangrove) and reuse it across all behavioural questions.

You don't need three perfect projects. You need **one solid lead + two competent backups**. Spend 60% of project-prep time on credit risk, 25% on whichever you'd lead with second, 15% on the third.
