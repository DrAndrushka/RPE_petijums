# 🏥 RPE analysis pipeline (v2)

Turn your Excel cohort into **tables, plots, and an HTML report**.  
Run everything from **`v2_RPE_notebook.ipynb`**.

```
📂 load data
  → 🏷️ guess column types
  → 🧹 clean
  → 📊 describe (DDA)
  → ❓ missing data + MICE
  → 🔍 screen associations (EDA)
  → 📈 logistic regression (pooled)
  → 📄 HTML report
```

---

## 🚀 Quick start

```bash
cd RPE_petijums/dev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook v2_RPE_notebook.ipynb
```

Put **`RPE 2020-2025.xlsx`** here (or change `DATA_PATH` in the notebook).  
⚠️ **Do not commit the Excel file** — it may have patient IDs.

After the notebook finishes:

```bash
python report.py --output-root output
```

Open **`output/report/report.html`** in a browser.

Tests: `python -m unittest test_pipeline_fixes.py -v`

---

## 📁 What each file does

| File | Does what |
|------|-----------|
| `schema_infer.py` | Guesses column type (number, category, yes/no, ID, …) |
| `cleaning.py` | Fixes types, drops dupes, makes bins & flags |
| `dda.py` | Summary stats + charts per column |
| `missingness_resolution.py` | Who is missing, MICE fill-in |
| `eda.py` | One test per outcome × predictor |
| `inferential.py` | Adjusted logistic model + forest plot |
| `report.py` | Builds the HTML from `output/` |

All results go to **`output/<step>/figures/`** and **`tables/`** (gitignored).

---

## 🔢 Stats cheat sheet (plain language)

### 🏷️ Column types (`schema_infer.py`)

Auto-labels each column, e.g. `continuous`, `ordinal`, `binary`, `id`, `skip`.  
You can edit the schema in the notebook if the guess is wrong.

---

### 🧹 Cleaning (`cleaning.py`)

| Thing | How |
|-------|-----|
| % missing | `(# empty cells) / (all rows) × 100` |
| z-score | `(value − average) / spread` — if spread is 0, z = 0 |
| Yes/no | Maps yes/no, 1/0, true/false to True/False |
| Duplicates | Same patient ID twice → flagged |

*Note: the final model uses sample spread for z-scores; the cleaning helper may use population spread.*

---

### 📊 Descriptions (`dda.py`)

**Numbers:** min, 5th %, median, mean, 95th %, max, spread (SD), IQR, skew, “trimmed” mean (drops extreme 10% each side).

**Categories:** most common level, % in top bucket, how balanced classes are (entropy), median level (ordered only).

**Dates:** earliest, latest, span in days.

**Missing together:** Jaccard score — “how often do column A and B go missing on the same rows?”  
`both missing / either missing` (0 = never together, 1 = always together).

---

### ❓ Missing data (`missingness_resolution.py`)

| Idea | What we do |
|------|------------|
| **Structural NA** | e.g. “lesion 2” empty because there was only 1 lesion → count lesions, take max PIRADS, skip raw columns |
| **MNAR flag** | Extra yes/no column: “was this value missing on purpose?” |
| **Quick fill** | Median (numbers) or mode (categories) — **exploration only** |
| **MICE** | Build **10** full datasets; random forest guesses each gap; used for final models |

---

### 🔍 Screening (`eda.py`)

Compares each **outcome** to each **predictor** (needs ≥ 5 complete rows).

| Outcome type | vs number | vs category |
|--------------|-----------|-------------|
| yes/no | Mann–Whitney | χ² or Fisher (small cells) |
| number | Spearman | Kruskal–Wallis |
| category | Spearman / χ² | χ² |

**Effect sizes (rough guide):**

- Spearman ρ, rank-biserial **r**, Cramér’s **V** → strength of link  
- ε² after Kruskal–Wallis → how much groups differ  

**FDR:** adjusts p-values **separately per outcome** (default α = 0.05).  
Not one big correction across all outcomes.

**Plots (binary outcome):** % “event” per group + Wilson 95% CI.

---

### 📈 Final model (`inferential.py`)

**Only for yes/no outcomes.**

1. Predictors → z-scores, dummy categories, etc.  
2. Drop predictors with **VIF > 5** (too correlated with each other).  
3. Logistic regression on each MICE dataset.  
4. **Rubin pool** — average coefficients across the 10 imputations and combine uncertainty.

**Odds ratio (OR):** `exp(coefficient)`. OR = 1 → no effect; OR > 1 → higher odds of outcome; OR < 1 → lower odds.

**Forest plot:** OR + 95% CI on log scale; dashed line at 1.

**Pooling (short):**

- Average the 10 model coefficients  
- Add “between-dataset” noise to the standard error  
- p-value and CI from a t-style rule (Barnard–Rubin df when possible)

---

### 📄 Report (`report.py`)

Does **not** re-run stats — only reads CSVs/SVGs.

| Badge | Rule of thumb |
|-------|----------------|
| Weak / moderate / strong (correlation) | \|ρ\| or \|r\| or V: 0.10 / 0.30 / 0.50 |
| Weak / moderate / strong (OR) | same idea on log(OR) scale |
| ✅ FDR significant | adjusted p < 0.05 |
| 🟡 Nominal only | raw p < 0.05 but FDR not |
| Missing 🟢🟡🟠🔴 | &lt;5% / 5–20% / 20–40% / ≥40% |

---

## 📂 Where files land

```
output/
  schema/          column types
  cleaning/        what changed
  dda/             describe everything
  missingness/     gaps + MICE settings
  eda/             associations.csv + plots
  inferential/     OR tables + forest plots
  report/          report.html  ← open this
```

---

## 📚 Papers behind the methods

**❓ Missing data**
- **MICE** — van Buuren & Groothuis-Oudshoorn, 2011, *J Stat Softw*
- **Rubin pooling** — Rubin, 1987, Wiley
- **Pooled p / CI** — Barnard & Rubin, 1999, *Biometrika*
- *sklearn RF imputer (not R `mice`) — cite van Buuren + Rubin*

**🔍 Screening (EDA)**

- **FDR (per outcome)** — Benjamini & Hochberg, 1995, *JRSS B*
- **Mann–Whitney U** — Mann & Whitney, 1947, *Ann Math Stat*
- **Rank-biserial r** — Kerby, 2014, *Compr Psychol*
- **Spearman ρ** — Spearman, 1904, *Am J Psychol*
- **Kruskal–Wallis** — Kruskal & Wallis, 1952, *JASA*
- **ε²** — Tomczak & Tomczak, 2014, *Balt J Health Phys Act*
- **χ²** — Pearson, 1900, *Philos Mag*
- **Fisher exact** — Fisher, 1935, *JRSS*
- **Cramér’s V** — Cramér, 1946, Princeton UP
- **Wilson CI** — Wilson, 1927, *JASA*

**📈 Multivariable**
- **Logistic / OR** — Hosmer et al., 2013, Wiley
- **VIF ≤ 5** — O’Brien, 2007, *Qual Quant*

**📊 Descriptives (optional)**
- **Trimmed mean** — Huber, 1981, Wiley
- **Entropy** — Shannon, 1948, *Bell Syst Tech J*
- **Jaccard (co-missing)** — Jaccard, 1912, *New Phytol*

**✍️ Methods sentence** — MICE-style imputation (van Buuren & Groothuis-Oudshoorn, 2011; m = 10), univariate tests with Benjamini–Hochberg FDR **per outcome** (Benjamini & Hochberg, 1995), logistic regression with Rubin pooling (Rubin, 1987; Barnard & Rubin, 1999) and VIF pruning (O’Brien, 2007).

---

## ⚖️ License & data

- Code: [MIT](LICENSE)  
- Patient Excel: **your responsibility** — keep private, follow hospital rules  
- In publications: say you used MICE (m=10), Rubin pooling, and **FDR per outcome**
