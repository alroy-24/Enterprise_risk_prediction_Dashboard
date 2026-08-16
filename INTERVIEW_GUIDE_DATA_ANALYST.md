# Interview Guide — Data Analyst Role
### Enterprise Risk Intelligence Platform (this repo)

> **Use the numbers in THIS guide. They are the real, current numbers** (after the June 2026 fix).
> The older `INTERVIEW_GUIDE.md` says 85% accuracy — **do not use that number.** It came from a broken
> model that was "cheating" (data leakage). It has been fixed. If you say 85% and they dig, you'll be stuck.

---

## How to read this guide

- Read **§0 first** — it decodes your résumé bullet, which is the only thing the interviewer sees.
- Then read the whole thing **once**, slowly, the night before.
- The morning of, read only **§0.2**, **§1 (the intro)**, **§2 (numbers)**, and **§3 (the pipeline picture)**.
- Don't try to memorise word-for-word. Learn the **idea** of each answer, then say it in your own words.
- Every hard word is explained in plain English the first time it appears.

---

## 0. Your résumé bullet, decoded (READ THIS FIRST)

The interview is offline, so **the résumé bullet below is the only thing the interviewer sees.**
Every bold phrase in it is a door they can open. This section opens each one for you.

> **Enterprise Risk Intelligence Platform (ERIP)** — XGBoost, FastAPI, Plotly, SQL — GitHub
> - Engineered **20+ industry-standard financial ratios** (Altman Z-score, leverage, liquidity,
>   profitability, coverage) on **2,144 real public companies**; trained Logistic Regression baseline
>   and **XGBoost (85% test accuracy)** for default-probability scoring.
> - Mapped raw probabilities to **AAA–D S&P-style credit ratings** and built a **weighted aggregated
>   risk score** across financial, operational, and compliance dimensions for portfolio-level analytics.
> - Estimated worst-case losses using three **Value-at-Risk** methods (parametric, historical, Monte
>   Carlo) and **regulatory stress scenarios**, with per-company **SHAP** explanations.
> - Built a **7-view analytics dashboard** (FastAPI + Plotly) backed by **SQL window functions and
>   CTEs**; exposed **CSV endpoints** so the same data feeds **Excel and Power BI**.

### 0.1 — Three numbers on the résumé do not match the code. Fix these before you go.

I re-ran the model and counted the data. These three claims will not survive a follow-up question.

| Résumé says | The code says | What to do |
|---|---|---|
| **85% test accuracy** | **73.7%** accuracy (AUC 0.778) | **Change the résumé to 0.78 ROC-AUC.** See §0.2 |
| **2,144 real public companies** | 2,144 **rows** from **48** real companies | Say "2,144 records". See §0.3 |
| **7-view dashboard** | **8** views | Change to 8, or just say "eight screens" |

### 0.2 — The 85% problem (most important thing in this guide)

**85% is the old, broken number.** It came from a version of the model that was accidentally being
shown the answer while it learned — that's called **data leakage**, and it makes a model look great
in testing and fail in real life. That bug was found and fixed. The honest number is **73.7% accuracy**,
or better, **0.78 ROC-AUC**.

**If you can still edit the résumé, do it now.** Replace `XGBoost (85% test accuracy)` with:

> `XGBoost (0.78 ROC-AUC, 74% recall)`

That is a *stronger* bullet for a risk role, not a weaker one — recall is the metric that matters when
you're catching bankruptcies, and quoting AUC signals you know accuracy is the wrong headline metric
for unbalanced data (§7).

**If the résumé is already printed and you cannot change it**, do not hide from it. If they ask about
the 85%, say this calmly — it turns a mistake into your best answer of the interview:

> "I need to correct that number — it's out of date on my résumé. That 85% came from an early version
> where I found a data-leakage bug: a column was leaking the answer into the training data. After I
> fixed it the honest figures are 0.78 ROC-AUC and about 74% recall. I'd rather quote the real number.
> Catching the leak was the most valuable thing I learned on the project."

An interviewer who hears that learns you (a) audit your own work, (b) know what leakage is, and (c)
don't oversell. That is worth more than a fake 85%.

### 0.3 — "2,144 real public companies" — say "records", not "companies"

The file has **2,144 rows**, but they are not 2,144 different companies. The real structure is:

- **48 real public companies** (Apple, McDonald's, Deere, …) with genuine reported financials,
- across **3 reporting periods**,
- then **expanded to 2,144 rows** by generating variations of those real records
  (`src/expand_dataset.py`), so the model had enough data to train on.

So say **"2,144 records"** or **"a 2,144-row dataset"** — never "2,144 companies". If they ask how many
real companies, the honest answer is strong:

> "48 real companies across 3 reporting periods, expanded to 2,144 records with generated variations so
> the model had enough rows to train on. The financial ratios are computed the same way on every row.
> With more time I'd pull the full set from a proper filings API rather than expanding a smaller sample."

### 0.4 — "20+ ratios" is fine — here's the exact nuance

I counted: **28 ratio columns are engineered** in `src/features.py`, and **18 of them feed the model**
after dropping redundant and leaky ones. So "20+ engineered" is true. If they ask how many the model
uses, say:

> "I engineer about 28 ratios; 18 go into the model after I drop the ones that were redundant or leaking."

### 0.5 — The three terms the rest of this guide doesn't cover

Everything else in the bullet is already explained later (Altman Z-score, leverage, liquidity,
coverage → §6 · VaR and stress scenarios → §6 · AAA–D ratings → §6 · window functions and CTEs → §9 ·
CSV endpoints, Excel, Power BI → §10). These three are **not**, so learn them here.

**SHAP** *(the résumé says "per-company SHAP explanations")*
SHAP answers *"why did the model give THIS company THIS score?"* It splits a single prediction into a
list of pushes: this company's high debt pushed its risk **up** 12 points, its strong cash position
pushed it **down** 5. The pieces add up to the final score, so nothing is hidden.

> Say it as: *"SHAP breaks one company's score into which ratios pushed it up and which pushed it down.
> It turns the model from a black box into something an analyst can defend to a client or a regulator."*

**Know where it lives.** SHAP is in `src/explainability.py` and the **Streamlit** dashboard (`src/app.py`).
It is **not** in the newer FastAPI dashboard. If they ask to see it, open the Streamlit app — don't get
caught looking for a tab that isn't there. Honest line: *"SHAP is in the Streamlit version; the FastAPI
rebuild has the model-quality view and I haven't ported the SHAP view across yet."*

**Weighted aggregated risk score** *(financial / operational / compliance)*
One company has three separate worries, so I score each and blend them into one number
(`src/aggregation.py`, weights in `config/weights.yaml`):

| Dimension | Weight | What it measures |
|---|---|---|
| **Financial** | **0.45** | The model's default probability plus how leveraged the company is |
| **Operational** | **0.35** | Incident rate — how often things go wrong in the business |
| **Compliance** | **0.20** | How heavy the company's regulatory/compliance burden is |

> Say it as: *"Financial risk is the biggest driver, so it carries 45%. Operational is 35%, compliance
> 20%. They're deliberately kept as a config file, not hard-coded, so a risk manager can re-weight them
> without touching the code."*

If asked *"why those weights?"* — be honest: *"They're a sensible starting point, not calibrated. With
real loss data I'd fit the weights to what actually predicted losses instead of choosing them."*

**Plotly**
The charting library used for every graph in the dashboard. Chosen because the charts are
**interactive** — hover for a tooltip, zoom, click a legend to hide a series — which matters when an
analyst wants to inspect one company inside a 2,000-dot scatter. (Your chart reasoning is in §8.)

---

## 1. Your opening answer (about 60 seconds — practise with a timer)

When they say *"tell me about your project,"* say this. Slowly. In your own words.

> "I built a tool that looks at a group of companies and answers one question: **which of these
> companies is likely to go bankrupt, and why?**
>
> The dataset is about **2,100 records**, built from the real published financials of **48 public
> companies** — Apple, McDonald's, Deere — across **28 different industries** and three reporting
> periods. For each one I have the basic money numbers you'd find in their financial reports:
> revenue, profit, debt, cash.
>
> From those raw numbers I calculated **more than 20 financial ratios** — the same measurements a
> real credit analyst uses to judge how healthy a company is. Things like how much debt a company
> has compared to its profit, and whether it has enough cash to pay its bills.
>
> Then a **machine-learning model** reads those ratios and gives each company a **risk score** — the
> chance it defaults. I turn that score into a **credit rating from AAA down to D**, the same letter
> grades S&P uses, so a normal business person sees 'BB' instead of '0.17'.
>
> I also blend that into one **overall risk score** that combines financial, operational, and
> compliance risk, and for any single company I can show **which ratios drove its score** — so an
> analyst can defend the number rather than just trust it.
>
> Finally, I put it all into a **dashboard with 8 screens** — an overview, company details, industry
> comparisons, ratios, trends, stress tests, value-at-risk, and a model-quality page. And every table
> can also be downloaded as a CSV, so the same numbers open straight in **Excel or Power BI**."

Then stop. Let them ask about whatever interests them. Every sentence above is a door they can open.

---

## 2. Numbers to memorise

These are the numbers you can safely say out loud. All of them are real.

| Number | What it means |
|---|---|
| **2,144** | **Records** in the dataset — say "records", not "companies" (see §0.3) |
| **48** | Real public companies behind those records, across 3 reporting periods |
| **28** | Different industries covered |
| **20+** | Financial ratios engineered (28 built, **18** feed the model — see §0.4) |
| **43%** | Share of rows marked "high risk" — this is the base rate (see note below) |
| **0.78** | The model's ROC-AUC score on the test set (its main quality score, see §7) — **lead with this** |
| **74%** | Accuracy (73.7%). Real, but **don't lead with it** — and it is *not* 85% (see §0.2) |
| **74% / 68%** | Recall / precision (how many risky companies it catches, and how often it's right) |
| **80 / 20** | Train/test split — 1,715 rows to learn from, 429 rows to test on |
| **45 / 35 / 20** | Risk weights — financial / operational / compliance (see §0.5) |
| **AAA → D** | The 10-step credit rating scale (AAA is safest, D means already failed) |
| **9** | Stress-test scenarios (bad situations I test the portfolio against) |
| **3 × 3** | VaR grid — 3 calculation methods × 3 confidence levels (90/95/99%) |
| **8** | Dashboard screens |

> **"Base rate" in plain words:** 43% of the rows are labelled high-risk. So if you just guessed
> "high risk" for everyone, you'd be right 43% of the time. That's the bar the model has to beat.

**If you forget a number, round it and say so.** "Around 2,000 companies." "AUC in the high seventies."
Never make up an exact number — a wrong exact number is worse than an honest rough one.

---

## 3. How the whole thing works (be able to draw this on paper)

This flow is the heart of the project. If you can draw and explain this, you've shown you understand it.

```
1. Raw data (a CSV of company money numbers)
        │
2. Data cleaning  ── fix missing values, stop divide-by-zero, remove impossible numbers
        │
3. Feature engineering  ── turn raw numbers into 20+ ratios       (src/features.py)
        │
4. Machine-learning model  ── gives each company a risk score     (src/models.py)
        │
5. Credit rating  ── turn the score into a letter grade AAA–D     (src/risk_rating.py)
        │
6. Risk aggregation  ── combine everything into one overall score (src/aggregation.py)
        │
7. API layer  ── serves the numbers as JSON and CSV               (src/api/main.py)
        │
8. Dashboard  ── charts and tables (also feeds Excel / Power BI)  (src/api/static/)
```

**Say it in one breath:** *"Raw financial numbers get cleaned, turned into ratios, scored by a model,
graded into a credit rating, combined into one risk score, and served through an API to the dashboard
and to Excel or Power BI."*

The word **"pipeline"** just means this — a chain of steps where the output of one step is the input
to the next. Data goes in one end, a finished risk score comes out the other.

---

## 4. Data cleaning — what I actually did (they WILL ask this)

Data cleaning is fixing messy raw data so it's safe to use. Real data is never perfect, and interviewers
love this topic because it's where the real work is. Here are the exact problems I hit and how I fixed
each one. Learn these — they're your strongest, most honest answers.

**Problem 1 — Divide-by-zero.**
Many ratios are one number divided by another. If the bottom number is zero, the computer gives an
error or infinity. Example: interest coverage = profit ÷ interest paid. A company with no debt pays
zero interest, so you'd be dividing by zero.
- **Fix:** I added a tiny number (`+ 0.000001`) to the bottom of every division. That makes the maths
  safe without meaningfully changing the answer. This is called **safe division**.

**Problem 2 — Infinity and crazy values.**
Even after that, a few ratios came out as huge or infinite numbers that would break the charts and
confuse the model.
- **Fix:** I **clipped** them — meaning I capped extreme values at a sensible maximum and minimum, so
  one weird company doesn't ruin a whole chart.

**Problem 3 — Missing numbers.**
Some companies didn't report every line item. A blank cell can't be used in a calculation.
- **Fix:** Where it was reasonable, I **estimated** the missing value from something related. For
  example, if interest expense was missing, I estimated it as about 4% of the company's debt. Where I
  couldn't estimate honestly, I left it out rather than guess.

**Problem 4 — Wrong data types.**
Dates sometimes came in as plain text, and numbers as text too. You can't do maths on text.
- **Fix:** I **converted types** — dates into real date format, numbers into real numbers.

**Problem 5 — Checking the columns are all there.**
- **Fix:** Before anything else, I check the data has all the columns I expect. If a column is missing,
  the program stops loudly with an error instead of quietly producing wrong results. This is called a
  **schema check** ("schema" just means the expected shape of the data).

**One-line summary you can say:**
> "The cleaning was the longest part. Safe division to avoid divide-by-zero, clipping to remove
> impossible values, estimating a few missing items from sensible proxies, fixing data types, and a
> schema check up front so bad data fails loudly instead of silently."

---

## 5. Feature engineering — turning raw numbers into ratios

**Feature engineering** means creating new, more useful columns from the raw data. A model (or a human)
finds patterns much more easily in "this company's debt is 6× its profit" than in two separate raw
numbers.

In this project it means calculating the financial ratios. The raw data had about 20 columns; after
feature engineering it had over 40. The ratios fall into four groups:

- **Leverage** (how much debt) — e.g. Debt ÷ Profit.
- **Liquidity** (can it pay short-term bills) — e.g. current ratio, quick ratio.
- **Profitability** (how much profit per sale) — e.g. profit margin, return on assets.
- **Bankruptcy risk** — the Altman Z-score, a classic formula combining five ratios.

Each ratio is defined in §6. That's the finance you must be able to explain.

---

## 6. Finance terms you WILL be asked

Explain each in two simple sentences plus one number. Practise the first three on paper from memory.

| Term | Plain-English meaning + healthy range |
|---|---|
| **EBITDA** | A company's core operating profit, before interest, tax, and accounting deductions. It lets you compare companies fairly even if they carry different amounts of debt. |
| **Leverage (Debt-to-EBITDA)** | Debt ÷ profit — how many years of profit it would take to pay off all the debt. *Under 3 = healthy, over 5 = risky.* |
| **Current ratio** | Current assets ÷ current bills — can the company pay what it owes this year? *Over 1.5 = safe, under 1 = warning.* |
| **Quick ratio** | Same idea but tougher — it ignores unsold stock (inventory). A stricter cash test. |
| **Interest coverage** | Profit ÷ interest owed — how easily it covers its loan payments. *Over 3 = comfortable, under 1.5 = danger.* |
| **Debt-to-equity** | Debt ÷ owner's money — is it funded more by borrowing or by owners? *Over 2 = a lot of debt.* |
| **Profit margin** | Profit ÷ sales — how much of each sales dollar is profit. Tech firms ~30%, restaurants under 10%. |
| **ROA / ROE** | Profit compared to what the company owns (ROA) or to owners' money (ROE) — how efficiently it makes money. |
| **DSO (Days Sales Outstanding)** | How many days, on average, customers take to pay their bills. Higher = slower cash coming in. |
| **Altman Z-score** | A 1968 formula that mixes five ratios into one bankruptcy score. *Over 2.99 = safe, under 1.81 = danger zone.* You don't need to memorise the formula. |
| **Credit rating (AAA–D)** | Letter grades for safety. AAA to BBB = "investment grade" (safe). BB to C = "junk" (risky). D = already failed. |
| **Default probability** | The model's output — a number from 0 to 1 giving the chance the company fails to pay its debts. |
| **Basel III** | International banking rules that force banks to test themselves against bad economic scenarios. My worst scenario cuts revenue 40% and raises interest costs 50%. |
| **IFRS 9 / ECL** | An accounting rule that says banks must set aside money for **expected** losses *before* a company actually defaults. |
| **Stress testing** | Re-scoring every company under a pretend bad situation (e.g. "what if revenue drops 30%?") to see how much riskier the portfolio gets. |
| **Value at Risk (VaR)** | The most you'd expect to lose, at a chosen confidence level. "95% VaR of 2%" means: on the worst 5% of days you'd lose at least 2%. Three ways to calculate it: **historical** (look at the past), **parametric** (assume a normal bell-curve), **Monte Carlo** (simulate thousands of possible futures). |

---

## 7. Data & machine-learning terms (explained simply)

You don't need to be a data scientist for an analyst role, but you must handle these basics.

**Model.** A model is just a formula the computer *learns* from examples. You show it lots of companies
where you already know who failed, and it learns which ratio patterns go with failure.

**Two models I used, and why two.**
- **Logistic Regression** — a simple, see-through model. You can read exactly how it made each decision.
  I used it as the **baseline** (the simple version to compare against).
- **XGBoost** — a more powerful model that builds many small decision trees, each fixing the last one's
  mistakes. It's more accurate, so it's the **main model**. Having both means I can offer either
  simplicity or accuracy depending on what the business needs.

**Train/test split.** I split the data 80/20. The model **learns** from 80% (the training set) and is
**tested** on the other 20% it has never seen (the test set). This checks it can handle new companies,
not just memorise the answers. I split it in a **stratified** way — meaning both halves keep the same
43% mix of risky companies, so the test is fair.

**Accuracy, precision, recall** (know the difference — this is a favourite question):
- **Accuracy** = of all predictions, how many were right overall.
- **Precision** = of the companies I flagged as risky, how many really were risky. (Avoids false alarms.)
- **Recall** = of the companies that really were risky, how many I caught. (Avoids missing real risks.)

For risk work, **recall matters most** — missing a company that fails is far worse than a false alarm.
My model catches **74%** of the risky companies (recall) and is right **68%** of the times it raises a
flag (precision).

**ROC-AUC (the main score — my model gets 0.78).** This is the fairest single score for this kind of
problem. Plain meaning: *if you hand the model one risky company and one safe company, AUC is how often
it correctly ranks the risky one as riskier.* 0.78 means 78% of the time. 0.5 would be pure guessing,
1.0 would be perfect.

**Why not just quote accuracy?** Because the classes are unbalanced. When one group is bigger than the
other, plain accuracy can look good while the model is actually doing badly on the group you care about.
AUC, precision, and recall are more honest here.

**Pipeline (in the model sense).** A pipeline bundles the cleaning steps and the model into one object,
so the *exact same* preparation is applied to new data automatically. It stops mistakes like forgetting
to clean a column before scoring it.

**Correlation.** A number from −1 to +1 showing how two things move together. +1 = move together, 0 =
unrelated, −1 = move opposite. In my data, current ratio and quick ratio are about 0.9 correlated —
they measure almost the same thing — which is how I decide two ratios are redundant.

---

## 8. Charts in the dashboard (and why each one)

This is the section where an analyst interview is won or lost. Anyone can say "I made a bar chart."
What they're listening for is: **did you pick the chart because of the question, or because it looked
nice?** So this section goes chart by chart — 21 Plotly charts across 8 screens — and for each one gives
you the data behind it, the reason for the choice, and what you say if they push.

Everything here lives in `src/api/static/app.js` (the drawing) and `src/api/main.py` (the numbers).

### 8.1 — The one rule, and the cheat table

**Pick the chart by what the viewer needs to do.** Memorise this table; it answers 80% of chart questions.

| The viewer needs to… | Chart | Where in my project |
|---|---|---|
| Know one number *now* | KPI card | Overview, Stress tests, VaR, Model quality |
| Compare categories | Horizontal bar | Risk by industry, top 15 riskiest, feature importance |
| See the *shape* of a spread | Histogram | Default probability, Altman Z, interest coverage, returns |
| See a trend over time | Line chart | Risk over time, risk by industry over time |
| See if two things move together | Scatter plot | Leverage vs liquidity, profitability vs risk |
| See parts of a whole (few, ordered) | Donut | AAA–D rating split |
| See parts of a whole (nested, weighted) | Treemap | Sector → industry, sized by revenue |
| Spot a pattern across a whole grid | Heatmap | Ratio correlations, VaR grid, confusion matrix |
| Look up an exact value | Table | Companies, industry benchmarks, scenarios, model comparison |

> **A table is a visualisation choice too.** If the viewer's job is "find *this* company's number", a
> table beats any chart. That's why the Companies screen is a filterable table, not a chart — and saying
> that out loud shows more judgement than adding a 22nd chart would.

### 8.2 — Seven design rules I applied across every chart

This is the part almost no candidate has. Each of these is a real decision in the code.

1. **One shared style object.** Every chart is drawn with the same `PL_LAYOUT` and `PL_CONFIG` constants —
   same font, same grid colour (`#252a37`), transparent background so charts sit cleanly on their cards,
   toolbar hidden, `responsive: true` so they reflow on resize. *Why it matters:* 22 charts look like one
   product, not 21 experiments.
2. **Colour means the same thing everywhere.** One `RATING_COLORS` dictionary (green AAA → dark red D)
   drives the donut, the top-risky bars, both scatter plots, *and* the little rating pills in the tables.
   A viewer learns the colour code once and it never lies to them.
3. **Every chart has a reference line.** A number alone doesn't tell you if it's good. So: the default-
   probability histogram gets a dashed mean line; the Altman histogram gets the textbook 1.81 (distress)
   and 2.99 (safe) cut-offs; the returns histogram gets the 95% VaR line; ROC gets the "chance" diagonal;
   the precision–recall chart gets the 43% base-rate line; calibration gets the perfect diagonal.
   **Say this cold:** *"A reference line turns a chart into a verdict."*
4. **Outliers are clipped for display, never in the data.** One company with 400× leverage would flatten
   every other dot into a smear. So the view clips (`clip(v, 0, 15)`) and the axis label says
   *"(clipped)"*. The underlying CSV still has the true value. *Honest, and it keeps the chart readable.*
5. **Bars are horizontal and sorted by value, never alphabetical.** Industry names are long ("Semiconductors
   & Semiconductor Equipment"), so horizontal bars with `automargin` avoid rotated, unreadable labels.
   Sorting by value means the answer is at the end of the bar, not hunted for.
6. **Redundant encoding.** On the industry-risk chart, risk is shown by bar *length* and by *colour*
   (green→amber→red). Two channels for one fact is deliberate: it reads correctly at a glance and it
   survives being printed in greyscale.
7. **Hover carries the precision, so the chart stays clean.** Nearly every trace has a custom
   `hovertemplate` — the axis shows a rounded number, the hover shows the exact one plus context (e.g.
   hovering an industry bar also reveals how many companies are in it). *Clean by default, precise on demand.*

---

### 8.3 — Screen 1: Overview *(the 10-second answer)*

**4 KPI cards** — companies analysed, mean default probability, investment grade (AAA–BBB) count,
distressed (CCC–D) count. Each has a big value plus a small subtitle giving context (e.g. "36% of
portfolio"), and the card colours itself green/amber/red against a threshold in the code.
*Why:* **answer first, detail below.** A risk manager who reads only the top strip should still leave
knowing the state of the portfolio.

**Donut chart — rating distribution.** Slices in fixed AAA→D order, coloured by `RATING_COLORS`,
empty ratings dropped, labels and percentages printed on the slices.
*Why a donut and not a bar:* there are at most 10 ordered categories and they genuinely sum to a whole
(every company has exactly one rating), which is the one case where a pie/donut is the right tool. The
hole in the middle makes the slices read as arc *length* rather than confusing wedge area.
*If they push:* *"Pie charts are rightly criticised, but the criticism is about many unordered slices.
Ten ordered categories that sum to 100% is the legitimate case — and I label every slice with its
percentage so nobody has to estimate an angle."*

**Histogram — default probability density.** All predicted probabilities, 30 bins, with a dashed red
mean line and a "mean X%" annotation.
*Why:* the average alone hides the story. The shape is the story — and mine is **two-humped**: companies
pile up at clearly-safe and clearly-risky with a thin middle. *That is a genuinely useful finding*: it
means the model is decisive rather than parking everyone near 50%, and it means a review threshold in
the middle only pulls in a small number of borderline cases.

**Treemap — sector exposure.** Two nested levels (sector → industry), each rectangle's **area = total
revenue**, `branchvalues: 'total'` so parents equal the sum of their children.
*Why a treemap:* two reasons a pie can't cover — it's **hierarchical** (click a sector, see its
industries) and it's **weighted by a money value**, not by a count. This is a concentration-risk chart:
one giant rectangle means the portfolio is exposed to one sector.

**Horizontal bar — top 15 riskiest companies.** Each bar coloured by its rating, with the rating letter
printed at the end of the bar.
*Why:* the single most operational chart on the dashboard — "who do I call today?" One detail worth
mentioning: the data has one row per company *per reporting date*, so a naive top-15 would show the same
company three times. The code de-duplicates first (`dedupeByCompany`), keeping each company at its
**riskiest** observation. *That's the kind of detail interviewers remember.*

**Watchlist mini-table (top 8).** Clicking a row opens a detail modal with the company's identity,
financials, all its ratios, and its risk scores.
*Why:* charts show the pattern, the drill-down answers "…but why *this* company?"

---

### 8.4 — Screen 2: Companies *(the lookup screen)*

No charts — a searchable, filterable table (free-text search, industry dropdown, rating dropdown),
sorted by risk, capped at 300 rendered rows with a footer that says so honestly
(*"Filter: 62 match · showing first 62"*).
*Why no chart:* the job here is lookup and export, not pattern-spotting. **The cap is a performance
decision** — rendering 2,144 rows of HTML makes the page stutter, and nobody scrolls past 300 anyway;
the full set is one click away as CSV.

---

### 8.5 — Screen 3: Industries *(the comparison screen)*

**Horizontal bar — mean risk by industry**, sorted so the riskiest industry sits at the top, coloured on
a green→amber→red scale, hover reveals the company count.
*Why the hover count matters:* an industry with 2 companies and one with 300 look identical as bars. The
count on hover is the guard against over-reading a tiny sample — mention this and you've shown you think
about **sample size**, which is a real analyst instinct.

**Horizontal bar — industry composition** (company count per industry), single colour.
*Why single colour:* count isn't good or bad, so colouring it would imply a judgement that isn't there.
Two charts side by side then answer both halves of the question — *how risky* and *how much of it we hold*.

*Good "what would you improve?" answer:* the risk chart puts the worst industry at the **top** and the
composition chart puts the biggest at the **bottom** — the sort directions don't match. It's a one-line
fix, and noticing it yourself is worth more than pretending the dashboard is perfect.

**Benchmarks table** — mean leverage, current ratio, EBITDA margin and Altman Z per industry.
*Why:* a leverage of 4× is alarming for software and normal for utilities. Benchmarks are what make a
ratio interpretable.

---

### 8.6 — Screen 4: Financial ratios *(the relationships screen)*

**Scatter — leverage vs liquidity.** X = leverage ratio (debt ÷ EBITDA, clipped to 0–15), Y = current
ratio (current assets ÷ current liabilities, clipped to 0–8), **dot colour = credit rating**, **dot size
= predicted default probability**.
*Why:* four variables in one picture, and it maps to how credit analysts actually think — the danger
zone is bottom-right (**lots of debt, little cash to pay it with**), and you should see red dots cluster
there. Saying *"the chart is only useful because the danger quadrant is a real credit concept"* is much
stronger than *"I plotted two ratios."*

**Scatter — profitability vs risk.** X = EBITDA margin, Y = predicted default probability, coloured by
rating.
*Why:* this one is a **sanity check on the model, not on the companies.** The cloud should slope
downward — higher margins, lower predicted risk. If it didn't, the model would be learning something
backwards. Using a chart to audit your own model is exactly the instinct interviewers are probing for.

**Heatmap — ratio correlation matrix.** 12 fields (the ratios plus `predicted_proba` and
`aggregated_risk`), Pearson correlation, printed to 2 decimals in every cell.
Three deliberate choices worth naming:
- **A diverging colour scale** (red at −1 → dark at 0 → green at +1), because correlation has a
  meaningful zero point in the middle. Risk scales are one-directional, so those use a sequential
  green→amber→red instead. *"Diverging when zero is meaningful, sequential when it isn't"* is a
  one-sentence answer that sounds like experience.
- **The scale is pinned to −1…+1**, not auto-fitted. If you let it auto-scale, a matrix whose strongest
  correlation is 0.3 lights up as if it were 1.0 and you draw a false conclusion.
- **The number is printed in every cell**, so the colour is the fast path and the digits are the truth.

*What I found with it:* current ratio and quick ratio sit around 0.9 — they're measuring nearly the same
thing, which is how I decided which ratios were redundant. The `predicted_proba` row also doubles as a
quick read on which ratios move with risk at all.

**Histogram — Altman Z-score**, with the textbook zones marked: a red dashed line at **1.81** (distress)
and a green one at **2.99** (safe).
*Why:* the chart isn't really about the distribution — it's about **how much of the portfolio sits left
of 1.81.** The reference lines convert a shape into a headline.

**Histogram — interest coverage** (EBITDA ÷ interest expense), clipped to −5…60.
*Why clipped so hard:* a debt-free company has an effectively infinite ratio, which would compress every
real company into the first bin. The interesting region is near zero, where a company can't cover its
interest bill — so that's the region the axis is spent on.

---

### 8.7 — Screen 5: Trends *(the direction screen)*

**Dual-axis line chart — risk over time.** Mean aggregated risk and mean default probability on the left
axis; **mean leverage on a second right-hand axis, drawn dotted**.
*Why two axes:* risk sits between 0 and 1, leverage is a multiple like 3.5×. On one axis the risk lines
would flatten into the floor. **The honest caveat, which you should volunteer:** dual axes can be used to
fake a relationship by choosing convenient scales, so I made the second series visually distinct (dotted,
thinner) to signal *"this one is on the other axis."* Volunteering a chart's weakness is a credibility move.

**Multi-line — risk by industry over time.** Mean default probability per industry per date, limited to
the **top 10 industries by number of companies**, one colour each.
*Why the limit:* 28 lines is the classic "spaghetti chart" — technically complete, practically useless.
Ten is roughly the ceiling for distinguishable lines, and choosing them by company count means you're
showing the industries the portfolio is actually exposed to.

---

### 8.8 — Screen 6: Stress tests *(the "what if" screen)*

**9 scenario cards** — one per scenario, showing mean default probability, the P95, and how many
companies breach the risk threshold; card colour flips amber above 25% and red above 50%.

**Combined bar + dot chart.** Horizontal bars = the **mean** default probability under each scenario;
red diamond markers on the same row = the **P95** (the bad-tail outcome).
*Why combine them:* this is the best chart on the dashboard to talk about. The bar is the typical
outcome; the diamond is the outcome in a bad case. **The gap between them is the point** — two scenarios
can have the same average while one has a far worse tail, and a plain bar chart would hide exactly the
thing a risk team cares about. Putting both on one row makes the comparison automatic.
*Plain-English P95:* "in the worst 1 in 20 cases, it's at least this bad."

---

### 8.9 — Screen 7: Value at Risk

**Heatmap — VaR matrix.** 3 methods (historical, parametric, Monte Carlo) × 3 confidence levels
(90/95/99%), with the percentage printed in each cell. The 9 values come from 9 parallel calls to
`/var`.
*Why a heatmap for 9 numbers:* it's really a small table that colours itself. The pattern you want the
viewer to see instantly is that **loss grows as you move right** (higher confidence = a rarer, worse
case) and that **the three methods broadly agree** — agreement across methods is the credibility check.
If one row were wildly different, the colour would show it before anyone read a digit.

**Histogram — returns distribution.** 252 daily portfolio returns with a dashed red line at the 95%
historical VaR.
*Why:* this is the chart that *defines* VaR for a non-technical viewer. The line sits where the worst 5%
of days begin — you can literally point at the tail to the left of it and say *"that's the 5% of days
this number is about."* A definition you can point at beats a definition you recite.

---

### 8.10 — Screen 8: Model quality *(the trust screen)*

This screen exists to answer *"why should I believe the scores?"* — and having built it at all is a
strong signal.

**4 model KPI cards** — ROC-AUC, PR-AUC, recall/precision, Brier score, each colour-coded against a
threshold (e.g. AUC above 0.8 is green, above 0.65 amber).

**ROC curve** — both models plotted together with their AUC in the legend, the active model drawn
thicker, plus the grey dashed **"chance" diagonal**. The diagonal is what random guessing looks like;
distance above it is the model's skill.

**Precision–recall curve** — both models, with a dashed line at the **43% base rate**.
*Why both curves and not just ROC:* ROC can look flattering when classes are imbalanced. The PR curve
compares against "what you'd get by guessing", which is the honest bar. *"I show both because they fail
in different ways"* is a strong sentence.

**Calibration curve** — mean predicted probability on X, observed default rate on Y, against a perfect
diagonal.
*Why it matters more than accuracy here:* if the model says "20% chance of default", roughly 20 out of
100 such companies should actually default. Ranking correctly isn't enough when the number gets used to
price risk — **the probability has to mean what it says.** The Brier score card is the same idea as a
single number.

**Confusion matrix heatmap** — a labelled 2×2 with the counts written in (TP/FP/TN/FN).
*Why a heatmap for four numbers:* colour makes the imbalance jump out, and the plain-English axis labels
("Actually risky" / "Pred. safe") mean nobody has to remember what a false negative is. **Talk about the
asymmetry:** a false negative here is a risky company we called safe — far more expensive than a false
positive, which just costs an analyst an unnecessary review. That's the argument for tuning the
threshold away from 0.5 toward recall.

**Horizontal bar — feature importance**, as a share, biggest at the top.
*Why:* it's the "does this make sense?" chart. If leverage and coverage rank high, the model agrees with
credit intuition; if something meaningless ranked first, that's a leakage warning. *This chart is
literally how the leakage bug in §0.2 would have been caught.*

**Model comparison table** — every metric for both models side by side, active one flagged.

---

### 8.11 — Charts I deliberately did *not* use (a great question to be ready for)

- **3D charts** — they look impressive and they're unreadable; occlusion and perspective make it
  impossible to compare values. The leverage/liquidity scatter carries four variables in 2D instead
  (position ×2, colour, size).
- **Stacked bars for anything that needs comparing** — only the bottom segment sits on a common baseline,
  so the rest can't be compared reliably. Grouped bars or a small table instead.
- **A pie chart for 28 industries** — this is where the treemap goes; area with nesting beats 28 wedges.
- **Truncated axes on bar charts** — bar length *is* the value, so starting the axis at anything but zero
  exaggerates differences. (Line charts are the exception: they show change, so a zoomed axis is fair.)
- **Dual axes anywhere except the trend chart**, where the units genuinely differ by orders of magnitude.

**One honest limitation to volunteer:** the rating palette is red-to-green, which is the hardest pairing
for colour-blind viewers. It's mitigated because the value is always written out too — the rating letter
sits on each bar, the correlation number is in each cell, percentages are on the donut slices — but on a
production dashboard I'd switch to a colour-blind-safe scale. Naming a real weakness and its mitigation
lands better than claiming the design is flawless.

### 8.12 — Two quick "how" answers

- **Why Plotly?** It's interactive in the browser with no extra work — hover for exact values, zoom,
  pan — which is what a dashboard needs. Matplotlib would be my choice for a static report or a PDF.
  I also vendor `plotly.min.js` locally in `src/api/static/` rather than loading it from a CDN, so the
  dashboard works on a locked-down network with no internet access.
- **Where does the data come from?** Every chart reads the same FastAPI endpoints that back the CSV
  downloads — `/companies`, `/industry-benchmarks`, `/correlations`, `/time-series`, `/scenarios`,
  `/var`, `/model/metrics`. **One source of truth**, whether you're looking at the web dashboard, Excel,
  or Power BI. Nobody can bring a spreadsheet to a meeting that disagrees with the dashboard.

---

**Say this cold (the 20-second version):** *"I chose charts by the job. Compare things → bar. Shape of
data → histogram. Trend → line. Two things related → scatter. Parts of a whole → donut if it's a few
ordered categories, treemap if it's nested and weighted. A pattern across a grid → heatmap. And every
chart has a reference line on it — a mean, a threshold, a chance diagonal — because a number on its own
doesn't tell you whether it's good."*

---

## 9. SQL (they will test this — write it once by hand before the interview)

**The classic question: "top 5 riskiest companies in each industry."** This tests window functions.

```sql
WITH ranked AS (
  SELECT industry, company_id, predicted_proba,
         ROW_NUMBER() OVER (
           PARTITION BY industry
           ORDER BY predicted_proba DESC) AS rn
  FROM scored_companies
)
SELECT industry, company_id, predicted_proba
FROM ranked
WHERE rn <= 5;
```

**Read it out loud like this:** *"A WITH block (a CTE) to keep it readable. A window function that ranks
companies inside each industry, riskiest first. Then the outer query keeps only the top 5 per industry."*

Three SQL terms to know:
- **CTE** — a `WITH ... AS (...)` block. It's just a named, temporary result that makes a big query
  readable by breaking it into steps.
- **Window function** — does a calculation across a group of related rows *without* squashing them into
  one. Good for rankings and running averages. Example: `AVG() OVER (PARTITION BY company_id ORDER BY date)`.
- **PERCENTILE_CONT** — finds percentiles, e.g. the median (P50) or 75th-percentile leverage per industry,
  which I use for industry benchmarks. These patterns live in `sql/risk_analytics_queries.sql`.

---

## 10. The Excel / Power BI story (your edge for analyst roles)

> "Every table in my dashboard is also a **CSV download** — like `/csv/companies` and
> `/csv/model-metrics`. Power BI connects with *Get Data → Web*, Excel with *Data → From Web*. So the
> same trusted numbers feed the web dashboard, Excel, **and** Power BI — one source of truth, three ways
> to view it. I wrote a setup guide (POWERBI_SETUP.md) covering the connection and scheduled refresh. The
> charts carry over directly: the correlation heatmap becomes a Power BI matrix with colour formatting,
> the treemap is a native Power BI treemap, and the KPI cards become card visuals. So I can rebuild the
> whole thing in whatever tool the team already uses."

---

## 11. Questions you'll get, and answers that work

**"Walk me through one insight you found."**
> "The portfolio leans risky — about 43% of records are flagged high-risk, and only a small slice is
> investment grade. When I run the worst Basel stress scenario — revenue down 40%, interest costs up 50%
> — the average default probability jumps a lot and the worst-case tail gets ugly. Restaurants and heavy
> machinery come out as the most exposed industries."

**"How good is the model?"** *(keep it simple, lead with AUC)*
> "Its main score, ROC-AUC, is about 0.78 on data it had never seen. In plain terms, if you hand it one
> risky and one safe company, it ranks the risky one higher 78% of the time. It catches about 74% of the
> truly risky companies. I use AUC instead of plain accuracy because the two groups aren't the same size,
> and accuracy would be misleading."

**"What was the hardest part?"**
> "The data cleaning. Ratios divide one number by another, and lots of them tried to divide by zero — a
> debt-free company has no interest to cover. I added safe division, capped the impossible values, and
> estimated a few missing items from sensible proxies. That cleanup took longer than building the charts."

**"How did you clean the data?"** *(give the concrete list from §4)*
> "Five things: a schema check up front so missing columns fail loudly; converting dates and numbers to
> the right types; safe division to avoid divide-by-zero; clipping extreme and infinite values; and
> estimating a few missing items, like interest expense as roughly 4% of debt."

**"What would you improve?"**
> "Three things. The risk labels are partly made-up — in a real setting I'd train on actual past defaults
> from a credit bureau. I'd add stronger data checks at the point the data comes in. And I'd re-tune the
> rating thresholds to this portfolio, because with a 43% risk rate the current AAA-to-D mapping puts very
> little into the safe grades."

**"Where did the data come from?"**
> "Real public financial filings, spread across several reporting periods, plus some made-up operational
> and compliance fields, because companies don't publish those."

**"Tell me about one chart."**
> "The leverage-vs-cash scatter. Each dot is a company: debt-to-profit across the bottom, cash position up
> the side, colour is the credit rating, and dot size is the risk. The bottom-right corner — lots of debt,
> little cash — is where the red dots gather. One chart tells the whole story."

**"How did you design the dashboard?"**
> "Like a newspaper — headline first, detail below. The overview answers 'how risky is the whole portfolio?'
> in four big numbers before you scroll. Each tab after that goes one level deeper: companies, industries,
> ratios, time, stress tests, VaR, model quality. Risk colours mean the same thing on every chart, and the
> watchlist is clickable so you go from the big picture to one company in a single click. My test: can
> someone get the headline in ten seconds and the full story in two minutes?"

**"What's a pipeline?"** *(in case they ask the word directly)*
> "A chain of steps where each step's output feeds the next. In my project: raw data → cleaning → ratios →
> model → rating → dashboard. Bundling those steps together means the same preparation always runs before
> scoring, so nothing gets skipped by accident."

---

## 12. Honesty rules (you built this with AI help — that's fine)

- It's normal in 2026 to use AI tools to write code faster. You can say so. But you **must** be able to
  explain every idea in this guide in your own words. They're testing whether you understand it, not
  whether you typed every character.
- Never say a number you can't defend. Everything in §2 is real — open the **Model quality** tab and
  re-check it before the interview if you want.
- If you don't know something: *"I haven't used that, but the closest thing in my project is ___."*
- Don't say "deployed in production." Say "production-style setup — an API, tests, CI, and Docker."

---

## 13. Closing line if they ask "anything else?"

> "The natural next step is connecting it to live data — the loading layer already supports a Postgres
> database — and adding alerts. So instead of an analyst checking the dashboard, the dashboard would warn
> the analyst the moment a company drops two grades down the rating scale."
