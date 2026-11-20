# Reddit Sentiment Analysis: AI Accusations vs AI-like Content

Big Data Analytics (CSE557) Project — Analyzing 2.6 million Reddit posts from 2019 vs 2024–2026 using Apache Spark and VADER sentiment analysis.

## Overview

This project investigates the dramatic rise in AI-related accusations on Reddit and compares it against actual AI-like writing patterns. We discovered a **24:1 accusation gap** — users accuse others of AI use 24 times more often than our heuristic detects AI-like style.

### Key Findings

- **2,616,810 posts analyzed** from 10 subreddits
- **13× rise in AI accusations** (2019 → 2024–2026)
- **0.016% AI-like posts** detected in 2024–2026 (lower bound estimate)
- **24:1 gap** between accusations (0.064%) and measured AI-like content (0.0026%)
- **Neutral sentiment collapsed** from 14.51% to 7.93% across periods
- **72% of sentiment shift** explained by subreddit composition changes

## Dataset

**Source:** Arctic Shift Reddit Archive (Dataset 2)

**Subreddits analyzed (10 total):**
- tifu, photography, legaladvice, summonerschool, dating_advice
- buildapc, Guitar, offmychest, depression, TalesFromRetail

**Time periods:**
- Control: 2019 posts
- Treatment: 2024–2026 posts

**Raw data:** 2,888,325 posts → **2,616,810 clean posts** (90.6% retention)

## Pipeline Steps

### 1. Data Ingestion & Cleaning
- **Script:** d2_prep_trim.py, i_comments_pipeline.py
- Read Arctic Shift JSONL files (posts only, comments excluded)
- Remove duplicates, invalid entries, posts with insufficient text
- Convert to Parquet for efficient processing
- **Output:** output_ai/posts_scored.parquet/

### 2. Sentiment Scoring
- **Script:** i_comments_pipeline.py
- VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon-based scoring
- Classifies each post as positive, neutral, or negative
- **Runtime:** 2,219.87 seconds (98% of total processing time)
- **Throughput:** 1,156.6 rows/second

### 3. AI-likeness Detection
- **Script:** i_comments_pipeline.py
- Heuristic based on lexical features:
  - Hedging language patterns
  - Passive voice frequency
  - Sentence uniformity
- **Threshold:** 3.0 (configurable: 2.5, 3.5 tested)
- **Precision:** 16.6% (33 of 199 flagged posts validated as AI-written)
- High recall, low precision by design — a screening tool

### 4. Aggregation & Analysis
- **Scripts:** d2_charts.py, d2_chart_extra.py, d2_fill_report_tables.py
- Time-period comparisons (2019 vs 2024–2026)
- Subreddit-level sentiment distribution
- Composition-controlled sentiment adjustment
- Threshold sensitivity analysis

### 5. Visualization
- **Scripts:** make_charts.py, i_charts.py
- Generated figures saved to output/ and output_ai/
- Interactive demo page: index.html

## Project Structure

`
BDA Project/
├── ai_comments_pipeline.py      # Main Spark pipeline for AI analysis
├── sentiment_pipeline.py        # Base sentiment analysis pipeline
├── d2_prep_trim.py             # Data cleaning and preprocessing
├── d2_charts.py                # Primary visualization generation
├── ai_charts.py                # AI-specific charts
├── d2_paths.py                 # Path configuration
├── run_d2.py                   # Orchestration script
├── scalability.py              # Performance benchmarking
├── validate_sample.py          # Manual validation of AI detections
│
├── output/                     # General analysis outputs
│   ├── fig_dist.png           # Sentiment distribution
│   ├── fig_subreddits.png     # Subreddit comparison
│   ├── fig_runtime.png        # Performance metrics
│   ├── scalability.csv        # Scalability results
│   └── *.json                 # Summary statistics
│
├── output_ai/                  # AI-specific analysis outputs
│   ├── fig_posts_ai_share_monthly.png
│   ├── fig_posts_composition_control.png
│   ├── fig_posts_neutral_collapse.png
│   ├── fig_posts_sentiment_ai_vs_human.png
│   ├── fig_posts_sentiment_mix.png
│   ├── posts_sentiment_accusations.csv
│   ├── posts_sentiment_by_subreddit.csv
│   ├── posts_threshold_sensitivity.csv
│   ├── posts_summary.json
│   └── validation_sample_labelled.csv
│
├── index.html                  # Interactive demo page
├── requirements_locked.txt     # Python dependencies
└── README.md                  # This file
`

## Key Results

### Accusation vs Detection Gap
| Metric | 2019 | 2024–2026 | Change |
|--------|------|-----------|--------|
| Accusation language | 0.005% | 0.064% | +13× |
| AI-like style | 0.006% | 0.016% | +2.7× |
| Automated accounts | 0.118% | 0.133% | +1.1× |

### Sentiment Shift
| Class | 2019 | 2024–2026 | Change |
|-------|------|-----------|--------|
| Positive | 47.49% | 54.11% | +6.62 pp |
| Neutral | 14.51% | 7.93% | −6.58 pp |
| Negative | 38.00% | 37.97% | −0.03 pp |

**Composition effect:** 72% of the sentiment score rise is explained by shifting subreddit mix.

### Subreddit Analysis
9 of 10 subreddits saw neutral sentiment collapse:
- **tifu:** −22.7 percentage points
- **photography:** −22.6 pp
- **legaladvice:** −10.9 pp

Only **TalesFromRetail** moved against the trend (+0.6 pp).

### Validation
- **Flagged posts precision:** 16.6% (33/199 confirmed AI-written)
- **Unflagged false negatives:** 0.52% (1/193 reads as AI but not flagged)
- Design prioritizes recall over precision for screening

## Technologies

- **Apache Spark 3.5.0** — Distributed data processing
- **PySpark** — Python API for Spark
- **VADER (vaderSentiment)** — Sentiment analysis lexicon
- **Pandas** — Data manipulation for aggregation
- **Matplotlib** — Visualization
- **Python 3.11+**

## Usage

### Prerequisites
`ash
pip install -r requirements_locked.txt
`

### Run Full Pipeline
`ash
python ai_comments_pipeline.py posts "data/*.jsonl" local[*] 1.0 3.0
`

Parameters:
- posts — analyze posts (vs comments)
- data/*.jsonl — input file glob pattern
- local[*] — Spark master (use all CPU cores)
- 1.0 — fraction of data to process (1.0 = 100%)
- 3.0 — AI-likeness threshold

### Generate Charts
`ash
python ai_charts.py
python d2_charts.py
`

### View Results
Open index.html in a browser to see the interactive demo page.

## Performance

- **Total runtime:** 2,262.5 seconds (~38 minutes)
- **Processing rate:** 1,156.6 posts/second
- **Bottleneck:** VADER sentiment scoring (98% of runtime)
- **Data retention:** 90.6% after cleaning

Scalability tested with multiple configurations (local[1], local[2], local[4]).

## Methodology Notes

- **Posts only** — Comments excluded from analysis
- **No rounding artifacts** — All reported numbers are exact
- **Convergent evidence** — Accusation findings align with different corpus studies
- **Lower bound estimate** — 0.016% AI-like detection is conservative, not prevalence
- **Threshold sensitivity tested** — Results hold at 2.5, 3.0, and 3.5 thresholds

## Contributors

Big Data Analytics CSE557 Course Project

## License

Dataset: Arctic Shift Reddit Archive  
Analysis code: Educational use
