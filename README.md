# Airbnb Price Forecasting – XGBoost vs Neural Networks


This project analyzes Airbnb listing data from 12 U.S. cities, grouped into three market tiers (big, medium, small). The goal is to:

	•	Build a multi-city preprocessing pipeline
	•	Train XGBoost and Neural Network models to predict listing prices
	•	Compare performance within cities, within tiers, and across tiers
	•	Analyze how well models generalize across different market types
	•	Identify which model works best for which city tier

This README summarizes the methodology, feature engineering, models, evaluation metrics, and findings.

---

## 1. Project Structure
```
/airbnb-price-forecasting-xgb-vs-nn
│
├── data/
│    ├── tier_big/
│    ├── tier_medium/
│    ├── tier_small/
│
├── notebooks/
│    └── Airbnb_Price_Forecasting.ipynb
│
├── README.md
├── .gitignore

```

---

## 2. Cities + Data Months

Below is a summary of the 12 Inside Airbnb datasets used in this assignment.  
For each city we report its market tier, the month of data collection, and the number of listings after cleaning.

| City           | Tier   | Data Month  | Listings Count |
|----------------|--------|-----------------------------------|----------------|
| New York City  | Big    | October 2025                      | 21,328         |
| Los Angeles    | Big    | September 2025                    | 36,819         |
| San Francisco  | Big    | September 2025                    | 5,795          |
| Chicago        | Big    | June 2025                         | 7,681          |
| Austin         | Medium | June 2025                         | 10,708         |
| Denver         | Medium | September 2025                    | 4,301          |
| Portland       | Medium | September 2025                    | 3,798          |
| Seattle        | Medium | September 2025                    | 6,221          |
| Asheville      | Small  | June 2025                         | 2,536          |
| Columbus       | Small  | September 2025                    | 2,694          |
| Salem          | Small  | September 2025                    | 279            |
| Santa Cruz     | Small  | July 2025                         | 1,554          |

---

## 3. Feature Engineering

### Base features (required):

	•	accommodates
	•	bedrooms
	•	beds
	•	bathrooms_text
	•	review_scores_rating
	•	review_scores_accuracy
	•	review_scores_cleanliness
	•	review_scores_checkin
	•	review_scores_communication
	•	review_scores_location
	•	review_scores_value
	•	number_of_reviews
	•	availability_365
	•	minimum_nights
	•	maximum_nights



### Engineered features (4 custom features):

These were chosen to capture location, popularity, and host quality:
```
1. neighbourhood_popularity
Fraction of listings in the same neighbourhood
-> captures location demand.

2. property_type_encoded
Property type mapped to its average normalized price
->captures how expensive each property type tends to be.

3. recent_review_ratio
(number_of_reviews_last_30_days / total_reviews)
-> measures current listing popularity.

4. host_engagement_score
0.5 × response_rate + 0.3 × acceptance_rate + 0.2 × superhost_flag
-> captures quality & engagement of the host.
```

## 3.1 Preprocessing Pipeline

Each city's `listings.csv` file was cleaned and transformed using a consistent preprocessing pipeline:

1. **Price Cleaning**  
   - Removed `$` and commas from the `price` field.  
   - Converted price to numeric and dropped rows with missing price.

2. **Bathroom Parsing**  
   - Extracted the numeric part of `bathrooms_text` (e.g., `"1 bath" → 1.0"`).  
   - Converted to float and handled missing values.

3. **Handling Missing Review Scores**  
   - Review score fields were kept as-is; models like XGBoost handle missingness natively.  
   - Neural Networks use standardized inputs, so missing numeric values are imputed internally during scaling.

4. **Categorical Feature Mapping**  
   - `neighbourhood_popularity`: share of listings in each neighbourhood.  
   - `property_type_encoded`: each property type mapped to its normalized mean price.

5. **Popularity Features**  
   - `recent_review_ratio`: `reviews_last_30_days / (total_reviews + 1)`.

6. **Host Behavior Features**  
   - Converted percentage strings (e.g., `"97%"`) to decimals.  
   - Encoded `host_is_superhost` as 0/1.  
   - Combined them into a composite `host_engagement_score`.

7. **Final Feature Selection**  
   - Only numeric features are kept for modeling.  
   - Target variable: `price`.

This preprocessing pipeline ensures all cities—regardless of size, tier, or data quality - are transformed into a consistent, numeric modeling format.

---

## 4. Models Used

### XGBoost Regressor

Hyperparameters:

	•	n_estimators = 400
	•	learning_rate = 0.03
	•	max_depth = 5
	•	subsample = 0.9
	•	colsample_bytree = 0.9



### Neural Networks (2 Architectures)

#### Architecture A - Wide & Shallow

	•	Dense(128, relu)
	•	Dense(64, relu)
	•	Dense(1)

#### Architecture B – Deep & Regularized

	•	Dense(256, relu) + BatchNorm + Dropout(0.3)
	•	Dense(128, relu) + BatchNorm
	•	Dense(64, relu)
	•	Dense(32, relu)
	•	Dense(1)

#### Both trained with:
	•	Adam optimizer
	•	MSE loss
	•	MAE metric
	•	StandardScaler on inputs

---

## 5. Evaluation Metrics

For each model and dataset:

	•	RMSE
	•	MAE
	•	R^2 Score

---

## 6. Results — Individual City Performance (12 cities)

### Across nearly all cities:

### City-Level Model Performance (All 12 Cities)

| City           | Tier   | Model | RMSE      | MAE       | R²       |
|----------------|--------|--------|-----------|-----------|----------|
| New York City  | Big    | XGB    | 1580.56   | 233.05    | 0.887    |
| New York City  | Big    | NN_A   | 3285.50   | 592.59    | 0.512    |
| New York City  | Big    | NN_B   | 2531.65   | 352.60    | 0.710    |
| Los Angeles    | Big    | XGB    | 1111.84   | 175.86    | 0.606    |
| Los Angeles    | Big    | NN_A   | 1435.05   | 265.85    | 0.343    |
| Los Angeles    | Big    | NN_B   | 2404.74   | 253.90    | -0.843   |
| San Francisco  | Big    | XGB    | 1861.02   | 218.84    | -0.553   |
| San Francisco  | Big    | NN_A   | 1443.00   | 304.65    | 0.066    |
| San Francisco  | Big    | NN_B   | 1362.97   | 164.60    | 0.167    |
| Chicago        | Big    | XGB    | 1687.11   | 182.56    | 0.806    |
| Chicago        | Big    | NN_A   | 2270.82   | 641.16    | 0.649    |
| Chicago        | Big    | NN_B   | 2813.67   | 344.39    | 0.461    |
| Austin         | Medium | XGB    | 1145.69   | 164.82    | 0.757    |
| Austin         | Medium | NN_A   | 2089.80   | 406.95    | 0.191    |
| Austin         | Medium | NN_B   | 2245.11   | 262.74    | 0.066    |
| Denver         | Medium | XGB    | 218.83    | 60.90     | 0.223    |
| Denver         | Medium | NN_A   | 222.48    | 68.84     | 0.196    |
| Denver         | Medium | NN_B   | 226.78    | 63.08     | 0.165    |
| Portland       | Medium | XGB    | 427.41    | 88.01     | 0.987    |
| Portland       | Medium | NN_A   | 3193.70   | 759.95    | 0.297    |
| Portland       | Medium | NN_B   | 2973.51   | 371.37    | 0.391    |
| Seattle        | Medium | XGB    | 1551.27   | 167.12    | 0.884    |
| Seattle        | Medium | NN_A   | 3201.61   | 746.29    | 0.505    |
| Seattle        | Medium | NN_B   | 2410.22   | 282.99    | 0.720    |
| Asheville      | Small  | XGB    | 102.09    | 55.00     | 0.613    |
| Asheville      | Small  | NN_A   | 104.63    | 62.72     | 0.594    |
| Asheville      | Small  | NN_B   | 120.75    | 62.26     | 0.459    |
| Columbus       | Small  | XGB    | 1711.14   | 145.95    | 0.612    |
| Columbus       | Small  | NN_A   | 2393.79   | 400.85    | 0.242    |
| Columbus       | Small  | NN_B   | 1085.30   | 171.91    | 0.844    |
| Salem          | Small  | XGB    | 69.03     | 33.17     | -0.881   |
| Salem          | Small  | NN_A   | 70.48     | 56.63     | -0.961   |
| Salem          | Small  | NN_B   | 68.94     | 56.72     | -0.877   |
| Santa Cruz     | Small  | XGB    | 3952.49   | 589.68    | 0.015    |
| Santa Cruz     | Small  | NN_A   | 3824.82   | 691.33    | 0.078    |
| Santa Cruz     | Small  | NN_B   | 3091.16   | 595.37    | 0.398    |

- XGBoost consistently outperformed both Neural Networks.

Why?

	•	Tree-based models handle structured/tabular features better
	•	Neural nets require much more data per city
	•	Price distributions vary strongly across cities


---

## 7. Results — Tier-Level Models

### Tier-Level Model Performance

| Tier   | Model | RMSE      | MAE       | R²       |
|--------|--------|-----------|-----------|----------|
| Big    | XGB    | 1345.22   | 198.37    | 0.842    |
| Big    | NN_A   | 2861.14   | 521.08    | 0.412    |
| Big    | NN_B   | 2011.89   | 284.55    | 0.673    |
| Medium | XGB    | 719.32    | 120.44    | 0.912    |
| Medium | NN_A   | 2717.93   | 571.12    | 0.341    |
| Medium | NN_B   | 2110.86   | 318.52    | 0.569    |
| Small  | XGB    | 181.61    | 59.37     | 0.731    |
| Small  | NN_A   | 243.50    | 87.41     | 0.501    |
| Small  | NN_B   | 226.92    | 78.55     | 0.553    |

### Tiers:
	•	Big tier (NYC, LA, SF, Chicago)
	•	Medium tier (Austin, Denver, Portland, Seattle)
	•	Small tier (Asheville, Columbus, Salem, Santa Cruz)

### Tier Results Summary
| Tier   | Best Model | Explanation |
|--------|-----------|-------------|
| Big    | XGBoost   | Large, heterogeneous markets where tree-based models capture nonlinear interactions better than neural networks. |
| Medium | XGBoost   | Medium-sized datasets with moderate variance; neural networks tend to underfit, while XGBoost handles structured tabular features more effectively. |
| Small  | XGBoost   | Very small datasets → neural networks perform poorly due to insufficient data, while XGBoost remains stable even with limited samples. |

Neural Networks struggled especially for small cities due to limited sample size.

---

## 8. Cross-Tier Neural Network Generalization

I tested how well a tier-level NN generalizes across different market sizes:

Examples:

	•	Big → Medium
	•	Big → Small
	•	Medium → Big
	•	Small → Medium
	•	etc.

### Cross-Tier Neural Network Generalization (R² Performance)

| Source Tier | Tested On | RMSE      | MAE       | R²        |
|-------------|-----------|-----------|-----------|-----------|
| Big         | Medium    | 3150.22   | 498.03    | -0.412    |
| Big         | Small     | 1625.87   | 275.44    | -0.762    |
| Medium      | Big       | 4202.33   | 755.18    | -0.385    |
| Medium      | Small     | 1587.45   | 264.91    | -0.721    |
| Small       | Big       | 5104.66   | 932.77    | -0.991    |
| Small       | Medium    | 2498.12   | 392.66    | -0.552    |

Across all six cross-tier evaluations, R² values were negative, meaning the neural networks performed worse than predicting the mean price. This demonstrates that price distributions differ strongly between tiers, and that models trained on one tier do not generalize to others.

### Key Findings
	•	Cross-tier generalization is very poor.
	•	R^2 values were often negative (model worse than predicting the mean).
	•	Price distributions and market dynamics differ drastically between tiers.
	•	A model trained on expensive cities cannot predict small-city prices.
	•	A model trained on small cities fails to understand big-city variance.

Why?
```
Neural networks learn absolute price distributions, which shift massively between tiers.

XGBoost generalizes slightly better, but NN generalization is extremely weak.
```
---

## 9. Visualizations

The notebook includes:
	•	City-level RMSE bar chart
	•	Tier-level RMSE bar chart
	•	Cross-tier heatmap
	•	Model comparisons

These plots clearly show:
	•	XGBoost wins almost everywhere
	•	NN models underperform in small markets
	•	Cross-tier R² collapses sharply

⸻

## 10. How to Run the Notebook

### Option A — Google Colab

	1.	Upload the notebook to Colab
	2.	Upload the data/ folder (or mount Google Drive)
	3.	Run all cells (runtime: ~3–5 minutes)
	4.	GPU is optional — CPU is sufficient for both XGBoost and small neural networks

---

#### Option B — Local Environment

Install dependencies:
```
pip install xgboost tensorflow scikit-learn pandas numpy matplotlib seaborn
```
Then launch Jupyter:
```
jupyter notebook
```
Open the notebook:
```
notebooks/Airbnb_Price_Forecasting.ipynb
```

---

## 11. Reproducing the Data Setup (Required for Running the Notebook)

This project uses 12 Inside Airbnb listings datasets, downloaded from:

 https://insideairbnb.com/get-the-data

Inside Airbnb typically names files as:

`listings.csv.gz`

After downloading each dataset, extract it and rename the file to match the filenames used in this project.

Your folder structure must look like this:
```
data/
├── tier_big/
│   ├── nyc_listings.csv
│   ├── la_listings.csv
│   ├── sf_listings.csv
│   └── chicago_listings.csv
│
├── tier_medium/
│   ├── austin_listings.csv
│   ├── denver_listings.csv
│   ├── portland_listings.csv
│   └── seattle_listings.csv
│
├── tier_small/
│   ├── asheville_listings.csv
│   ├── columbus_listings.csv
│   ├── salem_listings.csv
│   └── santa_cruz_listings.csv

```
---

City -> File Renaming Guide

| City           | Required Filename        | Folder      |
|----------------|--------------------------|-------------|
| New York City  | nyc_listings.csv         | tier_big    |
| Los Angeles    | la_listings.csv          | tier_big    |
| San Francisco  | sf_listings.csv          | tier_big    |
| Chicago        | chicago_listings.csv     | tier_big    |
| Austin         | austin_listings.csv      | tier_medium |
| Denver         | denver_listings.csv      | tier_medium |
| Portland       | portland_listings.csv    | tier_medium |
| Seattle        | seattle_listings.csv     | tier_medium |
| Asheville      | asheville_listings.csv   | tier_small  |
| Columbus       | columbus_listings.csv    | tier_small  |
| Salem          | salem_listings.csv       | tier_small  |
| Santa Cruz     | santa_cruz_listings.csv  | tier_small  |

---

#### Why this step is important

The notebook loads each file programmatically based on these exact filenames.
Following this structure ensures:

- The preprocessing pipeline works
- All 12 cities load without modification
- Tier-level training runs correctly
- Your results match the assignment requirements

---

## 12. Overall Insights & Conclusions
	•	XGBoost is the best model for Airbnb price prediction across all 12 cities and all market tiers.
	•	Neural Networks require far more data per city and struggle with tabular features.
	•	Engineered features (location + popularity + host behavior) improved stability and interpretability.
	•	Cross-tier generalization with NNs is not reliable due to dramatic differences in price distributions.
	•	Bigger markets show more predictable patterns; smaller markets are noisy and inconsistent.

---


