# 3-Day Study Plan — Data Analyst Internship Interview

You have 3 days. The trap is trying to "cover all of SQL" and "all of Python." Don't.
Cover the **20% of each that comes up in 80% of analyst interviews.** That's what this plan is.

> **How to use:** print the four cheat sheets (§3, §4, §5, §6). Work through the day-by-day in §2. Treat §7 as the night-before checklist.

---

## 1. What an analyst interview actually tests

A campus / internship interview for a data-analyst role almost always hits five buckets:

| Bucket | Weight | What gets asked |
|---|---|---|
| **SQL** | ~30% | A query on paper. Usually JOIN + GROUP BY, sometimes a window function. |
| **Python / pandas** | ~15% | "How would you do this in pandas?" — filter, groupby, merge. |
| **Project deep-dive** | ~25% | Walk-through + follow-ups on your project. Already covered in INTERVIEW_GUIDE.md. |
| **Resume questions** | ~15% | "Tell me about this bullet." Every line on your resume is a question waiting to be asked. |
| **Behavioural** | ~15% | "Tell me a time you …" — failure, conflict, deadline, learning. |

If you can answer SQL + project + resume cleanly, you'll be above the bar for an internship. The other two are bonus.

---

## 2. The 3-day schedule

Each day = ~6 productive hours. Adjust to your reality. The order matters — SQL first because it's the highest-yield, resume last because it just needs polishing.

### **Day 1 — SQL day**

| Time | What |
|---|---|
| Morning (2h) | Read §3 — the SQL cheat sheet. Write out every example query *by hand* on paper. Do not type. |
| Midday (2h) | 10 practice problems (§8). Write the query first, then check. Aim for 7/10 correct. |
| Evening (2h) | Re-read your wrong answers. Re-write those queries from scratch the next morning. |

**Day 1 success criteria:** you can write a JOIN + GROUP BY + ORDER BY query on paper without thinking, and you can explain what a window function is in one sentence.

### **Day 2 — Python + Resume**

| Time | What |
|---|---|
| Morning (2h) | Read §4 — pandas cheat sheet. Write out the "SQL → pandas translation table" by hand. |
| Midday (2.5h) | Rewrite your resume using §5 framework. Every bullet gets the action-verb + metric treatment. |
| Late afternoon (1h) | Re-do yesterday's wrong SQL problems from memory. |
| Evening (1h) | Read §6 — behavioural answers. Pick *one* failure story, *one* teamwork story, *one* learning story. Write them out as 4-sentence STAR answers. |

**Day 2 success criteria:** resume is one page, every bullet has a verb + a number, and you have three behavioural stories you can tell without thinking.

### **Day 3 — Integration + rest**

| Time | What |
|---|---|
| Morning (1.5h) | Recite the project pitch (Q-D-F-A-D-I-L) out loud three times. Time it. Should be 2 min 30 sec. |
| Midday (1.5h) | Mock yourself: pick 5 questions from §6 + INTERVIEW_GUIDE.md §5. Answer out loud, no notes. |
| Afternoon (1h) | Identify your *one weakest spot* from the morning and re-read just that section. |
| Evening | **Stop studying by 7 PM.** Iron your clothes. Pack your bag (resume copies, notebook, two pens, this guide). Sleep early. |

**Day 3 success criteria:** you are calm and bored, not panicking. If you're still studying at 10 PM the night before, you'll do worse, not better.

---

## 3. SQL cheat sheet (print this)

### The 6 SQL patterns that cover 90% of interview questions

#### a) Filter + sort
```sql
SELECT name, salary
FROM employees
WHERE department = 'Sales'
  AND salary > 50000
ORDER BY salary DESC
LIMIT 10;
```

#### b) GROUP BY + aggregation
```sql
SELECT department, COUNT(*) AS headcount, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 5            -- HAVING filters AFTER aggregation
ORDER BY avg_salary DESC;
```

**Remember:** `WHERE` filters rows *before* grouping. `HAVING` filters *after*.

#### c) JOIN (most-asked SQL question, period)
```sql
SELECT o.order_id, c.name, o.amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
WHERE o.amount > 100;
```

**Know cold:**
- **INNER JOIN** — rows that match on both sides
- **LEFT JOIN** — all rows from the left table, NULLs where right has no match
- **RIGHT JOIN** — mirror of LEFT (rarely used; flip the tables and use LEFT)
- **FULL OUTER JOIN** — all rows from both sides (rare in interviews)

If asked "what's the difference between INNER and LEFT JOIN?" — say:
*"INNER returns only matching rows from both tables. LEFT returns every row from the left table, with NULLs where the right table has no match."*

#### d) CASE WHEN (conditional column)
```sql
SELECT name, salary,
  CASE
    WHEN salary < 30000 THEN 'Low'
    WHEN salary < 80000 THEN 'Mid'
    ELSE 'High'
  END AS salary_band
FROM employees;
```

#### e) Subquery vs CTE (Common Table Expression)

Both do the same job — break a complex query into steps. **CTEs are more readable**, prefer them.

```sql
-- CTE version (preferred)
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department
)
SELECT e.name, e.salary, d.avg_salary
FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_salary;
```

The query above answers: *"Find employees earning more than their department's average."* Classic interview question.

#### f) Window functions

The pattern that separates "knows SQL" from "really knows SQL." Three you must know:

```sql
-- ROW_NUMBER: rank rows within a group, no ties
SELECT name, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
FROM employees;

-- RANK: same as ROW_NUMBER but ties share the same rank (skips next)
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) AS rk
FROM employees;

-- Running total
SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;
```

**Top-N per group** (your project's "top 5 riskiest companies per industry" question):
```sql
WITH ranked AS (
  SELECT industry, company_id, predicted_proba,
         ROW_NUMBER() OVER (PARTITION BY industry
                            ORDER BY predicted_proba DESC) AS rn
  FROM scored_companies
)
SELECT industry, company_id, predicted_proba
FROM ranked
WHERE rn <= 5;
```

### Order of operations (memorise!)

SQL is *written* in this order:
```
SELECT  →  FROM  →  WHERE  →  GROUP BY  →  HAVING  →  ORDER BY  →  LIMIT
```

But *executed* in this order:
```
FROM  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  ORDER BY  →  LIMIT
```

This is why you **can't** use a column alias from SELECT in a WHERE clause — at the time WHERE runs, the alias doesn't exist yet.

---

## 4. Pandas cheat sheet (SQL → pandas translation)

Write this out on paper. Every analyst job uses pandas; every SQL operation has a pandas equivalent.

| SQL | pandas |
|---|---|
| `SELECT col1, col2 FROM df` | `df[['col1', 'col2']]` |
| `SELECT * FROM df WHERE x > 5` | `df[df['x'] > 5]` |
| `SELECT * FROM df WHERE x > 5 AND y = 'A'` | `df[(df['x'] > 5) & (df['y'] == 'A')]` |
| `ORDER BY x DESC` | `df.sort_values('x', ascending=False)` |
| `LIMIT 10` | `df.head(10)` |
| `SELECT DISTINCT x FROM df` | `df['x'].unique()` |
| `COUNT(*)` | `len(df)` or `df.shape[0]` |
| `GROUP BY x` + agg | `df.groupby('x').agg({...})` |
| `INNER JOIN ON x = y` | `df1.merge(df2, on='x')` |
| `LEFT JOIN` | `df1.merge(df2, on='x', how='left')` |
| `CASE WHEN x > 5 THEN 'high' ELSE 'low' END` | `np.where(df['x'] > 5, 'high', 'low')` |
| `IS NULL` | `df['x'].isna()` |

### The pandas operations you'll be asked to do

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Inspect
df.shape                          # (rows, cols)
df.head()
df.info()
df.describe()                     # summary stats for numeric cols

# Filter
high = df[df['salary'] > 50000]
sales = df[df['dept'] == 'Sales']

# Group + aggregate
df.groupby('dept')['salary'].mean()
df.groupby('dept').agg(
    headcount=('id', 'count'),
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
)

# Top N per group (the famous one)
df.sort_values('salary', ascending=False).groupby('dept').head(3)

# Join
merged = orders.merge(customers, left_on='customer_id', right_on='id', how='left')

# Missing values
df.isna().sum()                  # count NaNs per column
df['x'].fillna(0)                # replace NaNs with 0
df.dropna(subset=['x'])          # drop rows where x is NaN

# Sorting & ranking
df.sort_values(['dept', 'salary'], ascending=[True, False])
df['rank'] = df.groupby('dept')['salary'].rank(ascending=False)

# New column
df['band'] = df['salary'].apply(lambda x: 'High' if x > 80000 else 'Mid')
# Or vectorised (faster + more pandas-y):
df['band'] = pd.cut(df['salary'], bins=[0, 30000, 80000, 1e9],
                    labels=['Low', 'Mid', 'High'])

# Pivot
df.pivot_table(index='dept', columns='gender',
               values='salary', aggfunc='mean')

# Date columns
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

### If they ask "which is faster, pandas or SQL?"

*"Depends on data size. For anything that fits in memory and you're doing one-shot analysis, pandas is faster to iterate. For huge datasets or anything multi-user, SQL on a real database is faster and more reliable because the database has indexes and a query planner."*

---

## 5. Resume framework (do this in one sitting on Day 2)

### The one-page rule

Internship resume = **one page**, no exceptions. If it's longer, cut.

### The 5-section structure

```
1. NAME + contact (phone, email, GitHub, LinkedIn — that's it)
2. EDUCATION   (institution, degree, dates, CGPA if > 7.5)
3. PROJECTS    (2-3 strongest, including this one)
4. SKILLS      (grouped: Languages / Data / Tools)
5. ACHIEVEMENTS / EXTRACURRICULAR (if relevant)
```

Skip: photo, address, marital status, "objective" sentence. None of these help.

### Every bullet follows this formula

```
ACTION VERB  +  WHAT YOU DID  +  TOOL / METHOD  +  RESULT / NUMBER
```

Bad bullet:
> *Worked on a machine learning project to predict credit risk.*

Good bullet:
> *Built credit-risk model on **2,144 public-company financials**, engineering **20+ ratios** (Altman Z, leverage, liquidity); XGBoost achieved **85% test accuracy** vs Logistic Regression baseline of 66%.*

### Strong verbs for analyst resumes

| Use | Avoid |
|---|---|
| Built, designed, analysed, computed | Worked on, helped with, was involved in |
| Reduced, increased, improved | Tried to, attempted |
| Automated, queried, visualised | Did, used, made |
| Identified, diagnosed, validated | Looked at, checked |

### Your project gets 3 bullets, not one. Suggested:

> - **Built credit-risk analytics platform** scoring 2,144 real public companies (Apple, McDonald's, etc.) for default probability; engineered 20+ financial ratios incl. Altman Z-score, leverage, liquidity, profitability.
> - **Trained Logistic Regression and XGBoost classifiers** on engineered features; XGBoost reached **85% test accuracy** vs **66%** baseline. Implemented SHAP for prediction-level feature attribution.
> - **Designed interactive dashboard** with 7 views (overview, peer benchmarks, ratio correlations, time-series trends, Basel III stress tests, Value-at-Risk); exposed CSV endpoints so output integrates with Excel and Power BI.

### Skills section template

```
Languages:   Python, SQL
Data:        pandas, NumPy, scikit-learn, XGBoost
Visualisation: Plotly, Matplotlib, Power BI, Excel
Database:    PostgreSQL, MySQL
Tools:       Git, VS Code, Jupyter
```

Don't list a skill you can't be quizzed on. If "Power BI" is on there and they ask "have you built a dashboard in it?" you need a real answer.

### What to cut

- Vague soft-skill bullets ("strong team player," "good communication"). Show, don't tell.
- Bullets that don't mention a tool or a result.
- Anything from before college unless it's genuinely exceptional.
- Hobbies/interests unless they're conversation-worthy (a real chess rating, a published blog, etc.). Avoid "reading, music, travel."

### When to share the resume with me

If you want me to review it, paste it as plain text. I'll mark it bullet-by-bullet.

---

## 6. Behavioural answers — STAR format

Pick **three stories**, one for each common theme. Then map every behavioural question to one of those three.

```
S — Situation: 1 sentence of context
T — Task:      what you specifically had to do
A — Action:    what you did (the meat — 2-3 sentences)
R — Result:    what happened, with a number if possible
```

### The three stories you need

1. **Failure / something that didn't work** — and what you learned.
2. **Teamwork / disagreement** — how you handled it.
3. **Learning / picked up a new skill quickly** — under time pressure.

### Example: failure story using the broken-styling bug from this project

> **S:** When I was building the dashboard for the credit-risk project, I integrated a UI styling library and assumed it was working.
> **T:** I needed the tab navigation, cards, and rating badges to render correctly.
> **A:** The first time I opened the dashboard, the tabs showed up as plain text — no styling at all. I'd used a CSS feature that required a build step, but I was loading the library straight from a CDN. Took me a while to realise the library was silently ignoring my custom rules. I rewrote everything in plain CSS — slightly more code, but predictable.
> **R:** Dashboard now renders correctly across browsers. The lesson was test the styling on a fresh, no-cache browser session before assuming any UI is "done."

This is true, specific, and shows you debug rather than blame the tool. Use it.

### Common behavioural questions and which story to use

| Question | Story |
|---|---|
| "Tell me about a time something didn't work." | Failure |
| "Describe a challenging project." | Failure or Learning |
| "Tell me about a time you had to learn something new." | Learning |
| "Tell me about a disagreement with a teammate." | Teamwork |
| "How do you handle pressure?" | Learning (the time-pressure version) |
| "Tell me about your biggest weakness." | Pick a real one + how you mitigate it. Not "I work too hard." |

### "Why this company?" / "Why this role?"

Have a 2-sentence answer. Do **30 minutes of research** on Day 3 morning — read their About page, find one specific thing (a product, a team, a value) and reference it.

> *"I want to work on real-world analyst problems where the data is messy and the stakeholders are non-technical — and from what I read about your [team / product / process], that's exactly the kind of environment you have."*

---

## 7. The night-before checklist

- [ ] Resume printed × 2 copies
- [ ] INTERVIEW_GUIDE.md §0 (cheat sheet) printed and folded in pocket
- [ ] STUDY_PLAN.md §3, §4 printed (just in case)
- [ ] Notebook + two pens in bag
- [ ] Clothes ironed and laid out
- [ ] Route to venue checked, leaving time for delays
- [ ] Phone charged + small backup
- [ ] Water bottle
- [ ] Watch (so you don't pull out a phone to check time)
- [ ] Asleep by 10:30 PM. Studying at midnight makes you worse, not better.

---

## 8. SQL practice problems (do these on Day 1 afternoon)

Tables:
```
employees(id, name, department, salary, hire_date, manager_id)
departments(id, name, location)
```

Write the query for each. Check your answers against the patterns in §3.

1. List all employees in the 'Engineering' department earning over 60,000, sorted by salary descending.
2. Count employees per department.
3. Show only departments with more than 10 employees, sorted by headcount.
4. List each employee alongside their department name.
5. List all departments and the count of employees in each — including departments with **zero** employees. (Hint: LEFT JOIN)
6. Find employees earning more than their department's average salary.
7. For each department, find the top 3 highest-paid employees.
8. Find the employee hired most recently in each department.
9. Categorise each employee as 'Junior' (< 50k), 'Mid' (50k-100k), or 'Senior' (≥ 100k). Show counts by category.
10. Show the running headcount of hires over time, ordered by hire date.

**Answers** — write yours first, then peek. Solutions in §9.

---

## 9. Solutions

```sql
-- 1
SELECT name, salary FROM employees
WHERE department = 'Engineering' AND salary > 60000
ORDER BY salary DESC;

-- 2
SELECT department, COUNT(*) AS headcount
FROM employees GROUP BY department;

-- 3
SELECT department, COUNT(*) AS headcount
FROM employees GROUP BY department
HAVING COUNT(*) > 10
ORDER BY headcount DESC;

-- 4
SELECT e.name, d.name AS department
FROM employees e
JOIN departments d ON e.department = d.id;

-- 5
SELECT d.name, COUNT(e.id) AS headcount
FROM departments d
LEFT JOIN employees e ON e.department = d.id
GROUP BY d.name;

-- 6
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_sal
  FROM employees GROUP BY department
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_sal;

-- 7
WITH ranked AS (
  SELECT name, department, salary,
         ROW_NUMBER() OVER (PARTITION BY department
                            ORDER BY salary DESC) AS rn
  FROM employees
)
SELECT * FROM ranked WHERE rn <= 3;

-- 8
WITH latest AS (
  SELECT name, department, hire_date,
         ROW_NUMBER() OVER (PARTITION BY department
                            ORDER BY hire_date DESC) AS rn
  FROM employees
)
SELECT name, department, hire_date FROM latest WHERE rn = 1;

-- 9
SELECT
  CASE
    WHEN salary < 50000  THEN 'Junior'
    WHEN salary < 100000 THEN 'Mid'
    ELSE 'Senior'
  END AS band,
  COUNT(*) AS n
FROM employees
GROUP BY band;

-- 10
SELECT hire_date,
       COUNT(*) OVER (ORDER BY hire_date) AS running_headcount
FROM employees
ORDER BY hire_date;
```

If you get 7/10 first try, you're ready. If less, redo wrong ones the next day before moving on.

---

Good plan. Stick to it. Don't optimise beyond it.
