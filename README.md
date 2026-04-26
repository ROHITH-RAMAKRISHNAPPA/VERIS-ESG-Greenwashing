# VERIS: Detecting Greenwashing Through Satellite-Verified NLP

A forensic NLP pipeline that cross-validates corporate sustainability disclosures against satellite-calibrated facility-level CO₂e emissions from Climate TRACE v5.4.1.


## Live Dashboard

**[Launch VERIS Dashboard](https://veris-esg-greenwashing.streamlit.app)** — hosted on Streamlit Community Cloud


## Project Report & Data

**[Google Drive - Full Report, Data & Appendix](https://drive.google.com/drive/folders/1gmEkyJPpQO3aK7Ij0Lz9mPKUFhZFLizG?usp=sharing)**

Includes the submitted PDF report, appendix (AI declaration, data dictionary, roles), and supporting data files that are too large for GitHub.


## Overview

ESG rating systems rest on two assumptions the data consistently undermine — that firms report emissions honestly, and that rating providers interpret the same disclosures consistently. Berg, Koelbel and Rigobon (2022) document pairwise provider correlations between 0.38 and 0.71 for identical firms. More fundamentally, no existing mechanism connects what a firm writes in a sustainability report to what a satellite independently observes about its emissions.

**VERIS** (Verified Emissions vs. Reported Information Score) addresses this gap. It computes four year-pair NLP signals across 119 sustainability reports from 12 globally listed firms (2014-2024, 4.0 million tokens), aggregates them via AHP-weighted TOPSIS against facility-level CO₂e estimates from Climate TRACE v5.4.1, and produces a single composite disclosure-reality gap score per firm-year.


## Key Results

| Metric | Value |
|---|---|
| Corpus | 12 firms, 119 reports, 4.0M tokens, 2014-2024 |
| Within-corpus discriminant AUC | 0.8924 (Q2 vs Q4 logistic regression) |
| Kruskal-Wallis H across 4 quadrants | 58.82, p < 0.001 |
| DoubleML CSRD effect (corrected) | θ = +0.0047, p = 0.61, n = 79 |
| EU-obligated firms | θ = +0.0086 (p = 0.38, n = 52) |
| Non-EU firms | θ = +0.0091 (p = 0.60, n = 27) |
| Placebo falsification (fake 2018) | θ = -0.0183, p = 0.0096 (passes: opposite sign) |
| Strongest Q2 greenwashing signal | ExxonMobil 2022-2023 (VERIS = 0.8944) |
| Second strongest Q2 signal | Rio Tinto 2018-2019 (VERIS = 0.8434) |



## NLP Signals and AHP Weights

| Signal | Library | Weight | Direction |
|---|---|---|---|
| SBERT semantic drift | sentence-transformers (all-mpnet-base-v2) | 46.15% | High = meaning change (benefit) |
| LDA-JSD topic divergence | scikit-learn + scipy, k=7 | 23.08% | High = topic reshuffling (benefit) |
| Jaccard trigram overlap | Python built-in | 23.08% | High = copy-paste recycling (cost) |
| VADER sentiment delta | vaderSentiment | 7.69% | High = tone escalation (benefit) |

AHP weights derived via Saaty's (1987) eigenvector method. CR = 0.00 by construction. Five sensitivity scenarios tested: Balanced, Equal-Weights, SBERT-Heavy, LDA-JSD-Heavy, Jaccard-Heavy.


## Greenwashing Quadrants

| Quadrant | VERIS | Emissions | Interpretation |
|---|---|---|---|
| Q1 Genuine Improvement | > median | Falling | Real progress, real narrative |
| Q2 Greenwashing Signal | > median | Rising/stable | Language change without operational change |
| Q3 Greenhushing | < median | Falling | Real progress, silent narrative |
| Q4 Stagnant | < median | Rising/stable | No change anywhere |

Corpus median VERIS = 0.162.


## Repository Structure

```
.
├── notebooks/
│   ├── 00_config.ipynb            # paths, constants, firm roster
│   ├── 01_pdf_extraction.ipynb    # 3-layer OCR: pdfplumber -> pdfminer -> docTR
│   ├── 02_forensic_features.ipynb # Fog Index, TTR, controversy density, social pillar
│   ├── 03_semantic_drift.ipynb    # SBERT drift, LDA-JSD (k=7), Jaccard, VADER
│   ├── 04_veris_scoring.ipynb     # AHP weights, TOPSIS, quadrant classification
│   ├── 05_causal_inference.ipynb  # FE panel, DoubleML, EU stratification, placebos
│   ├── 06_visualisations.ipynb    # 16 publication figures
│   ├── VERIS_ESG_Greenwashing.py  # Streamlit interactive dashboard
│   └── .streamlit/
│       └── config.toml            # Light theme configuration
│
├── outputs/
│   ├── csv/                       # All pipeline outputs
│   └── figures/                   # fig01 through fig16
│
├── src/
│   ├── ct_download_countries.py
│   └── ct_country_backfill.py
│
└── requirements.txt
```



## Run Order

```bash
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute notebooks/00_config.ipynb
jupyter nbconvert --to notebook --execute notebooks/01_pdf_extraction.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_forensic_features.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_semantic_drift.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_veris_scoring.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_causal_inference.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_visualisations.ipynb

streamlit run notebooks/VERIS_ESG_Greenwashing.py
```



## Data Sources

- **Sustainability Reports**: Collected from corporate investor-relations portals (2014-2024). 119 PDFs across 12 firms. Available via the [Google Drive folder](https://drive.google.com/drive/folders/1gmEkyJPpQO3aK7Ij0Lz9mPKUFhZFLizG?usp=sharing).
- **Emissions Data**: [Climate TRACE v5.4.1](https://climatetrace.org/downloads) under Creative Commons Attribution 4.0. Pre-2021 uses a country-share proxy. The 2020-2021 year-pairs are excluded from all causal panels due to a measurement-regime transition artefact (66-99% apparent emissions drop).

Raw PDFs and Climate TRACE CSVs are not tracked in this repository due to file size. See the Google Drive folder above.



## Causal Specification

Y_it = θ * D_it + g(X_it) + e_it     [DoubleML Partially Linear Model]
D_it = m(X_it) + v_it

Y  = year-on-year CO2e delta (%)
D  = 1 if year_to >= 2022 AND firm is EU-obligated, else 0
X  = {lagged VERIS score, log CO2e, EU firm flag, sector dummies}
```

Nuisance functions estimated by LassoCV with 5-fold cross-fitting and RidgeCV fallback. HC3-robust standard errors.



## Team

| Member | Contribution |
|---|---|
| **Courtney Fernando** | Confirmed 12-company corpus and Climate TRACE v5.4.1 mappings. Wrote the core methodology (VERIS design, AHP/TOPSIS, CSRD treatment, DoubleML, placebo). Defined sensitivity criteria. Developed the 2x2 greenwashing framework (Q1-Q4). Documented the 90-page filter, OCR pipeline, and Obfuscation Flag. |
| **Fida Hussain Abbas Rao** | Built GPU-accelerated PDF crawler for all 118 reports. Implemented docTR OCR pipeline for image-only documents (Eni 2015-2017, early Equinor) with page-level batch clearing. Applied uniform 90-page filter across corpus. Computed Gunning-Fog Index, Type-Token Ratio, and Obfuscation Flag per document. |
| **Weiyi Yan** | Ran Sentence-BERT embeddings across all consecutive year-pairs to measure semantic drift (46.15% VERIS weight). Applied LDA modelling and computed Jensen-Shannon Divergence scores for topic distributional shift (23.08% VERIS weight). Documented emergent topic clusters for qualitative interpretation. |
| **Shu-Han Hsu** | Computed Jaccard Overlap for copy-paste detection (23.08% VERIS weight) and VADER Sentiment Delta for tone escalation (7.69% VERIS weight). Extracted Controversy Density, Social Pillar Index, and Circularity Index. |
| **Adrika Navas** | Loaded Climate TRACE v5.4.1 sector bundles and linked facility records via rapidfuzz fuzzy ownership join (default threshold=62, per-firm overrides). Ran AHP weight derivation (CR=0.00 by construction) and TOPSIS. Executed five-scenario sensitivity analysis. |
| **Koushik Chowdhury** | Ran fixed-effects panel regressions with HC3-robust standard errors. Applied DoubleML with cross-fitted LassoCV to estimate the ATE of the 2021 CSRD on emissions. Conducted placebo falsification test using counterfactual treatment year 2018. |
| **Rohith Ramakrishnappa** | Built all 16 visualisations and the interactive Streamlit dashboard. Integrated all phase outputs into the final report. Managed Harvard referencing, formatting, and final proofreading. Compiled and submitted the report and appendix. |


## References

- Berg, F., Koelbel, J.F. and Rigobon, R. (2022). Aggregate confusion. *Review of Finance*, 26(6).
- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *The Econometrics Journal*, 21(1).
- Climate TRACE Coalition (2026). Climate TRACE Emissions Inventory v5.4.1. https://climatetrace.org
- Kim, E.H. and Lyon, T.P. (2015). Greenwash vs. brownwash. *Organization Science*, 26(3).
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- Robinson, P.M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4).
