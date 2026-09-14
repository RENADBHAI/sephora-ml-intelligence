# Machine Learning-Based Sentiment Analysis and Product Recommendation for Sephora Skincare Reviews

This project predicts whether Sephora skincare reviews are recommended or not recommended using NLP and machine learning, then uses the model outputs and product-level review statistics to recommend skincare products.

## Dataset

Sephora Products and Skincare Reviews Dataset from Kaggle.

Raw files are stored in `data/raw/`:

- `product_info.csv`
- `reviews_0-250.csv`
- `reviews_250-500.csv`
- `reviews_500-750.csv`
- `reviews_750-1250.csv`
- `reviews_1250-end.csv`

## Main ML Task

Binary classification with `is_recommended`:

- `1` = recommended
- `0` = not recommended

Rows with missing `is_recommended` are excluded from supervised training.

## Notebooks

1. `notebooks/01_data_audit_and_eda.ipynb`
   Advanced data audit, missing values, duplicates, text quality checks, target imbalance, product/brand analysis, and EDA plots.

2. `notebooks/02_preprocessing_and_feature_engineering.ipynb`
   Text cleaning, `full_review_text`, `clean_text`, length features, sentiment features, leakage review, `ColumnTransformer`, TF-IDF, scaling, encoding, and stratified training/validation/testing split.

3. `notebooks/03_model_training_and_evaluation.ipynb`
   Logistic Regression, Complement Naive Bayes, Linear SVM, Decision Tree, Random Forest, XGBoost, oversampling, class weights, validation-based threshold tuning, model comparison, ROC/PR curves, confusion matrices, feature importance, and error analysis.

4. `notebooks/04_product_recommendation_demo.ipynb`
   Product recommendation system using model-predicted product sentiment, average rating, recommendation rate, review count, helpfulness, text similarity, price preference, avoided ingredients, and optional brand filtering.

5. `notebooks/05_helpful_review_ranking_optional.ipynb`
   Optional extension that predicts which reviews are likely to be helpful for shoppers.

## Saved Model and Comparison Results

The saved/deployed classifier is an XGBoost model selected using validation performance. The full comparison tables in `reports/validation_model_comparison.csv` and `reports/model_comparison.csv` show that several models are very close. XGBoost is the saved model, but it is not claimed to be best on every test metric.

- Split strategy: 70% training, 15% validation, 15% testing
- Tuned threshold: 0.49
- Saved XGBoost test accuracy: 0.9633
- Saved XGBoost test precision: 0.9839
- Saved XGBoost test recall: 0.9722
- Saved XGBoost test F1-score: 0.9780
- Saved XGBoost test macro F1: 0.9335
- Saved XGBoost test weighted F1: 0.9638
- Saved XGBoost test ROC-AUC: 0.9882
- Saved XGBoost test PR-AUC: 0.9970

For comparison, the test table shows close or slightly higher F1 values for other models, such as Logistic Regression at 0.9799, Random Forest at 0.9800, and Soft Voting at 0.9798. These differences are small, so the report should discuss model comparison rather than saying one model is best for every metric.

Accuracy is not used alone because the classes are imbalanced. The validation split is used for model-selection support and threshold tuning, while the test split is kept untouched for final evaluation.

## Recommendation Logic

The recommendation notebook aggregates review-level model predictions per product as `positive_sentiment_ratio`, then ranks products with:

`final_score = 0.22 * positive_sentiment_ratio + 0.22 * recommendation_rate + 0.18 * normalized_average_rating + 0.15 * text_similarity + 0.10 * normalized_review_count + 0.08 * helpfulness_score + 0.05 * price_fit_score`

The user can enter skin type, one or more concerns, avoided ingredients, max price, price preference, rating preference, minimum review count, optional brand, and extra request details.

The optional Flask app uses all products from `data/raw/product_info.csv` and merges in the processed model/review metrics when they are available. Missing recommendation metrics are handled with safe defaults so products are not removed just because they do not have every processed field.

The Flask app also has a more complete demo scoring system. It hard-filters products above the maximum price, removes out-of-stock products when that metadata is available, removes products that contain avoided ingredients when ingredient data exists, and filters unrelated product types for face skincare concerns. Conflict filtering is based on ingredient/product metadata instead of review mentions, so products are not removed only because a review discusses an ingredient. It then ranks remaining products using product category match, concern-specific keyword rules, reviewer skin-type match, ingredient compatibility, brand preference, price fit, rating/review reliability, model sentiment, helpfulness, positive/negative feedback, Sephora loves count, TF-IDF review relevance, and a small product-text clustering signal. Review snippets are selected by comparing review sentences to the user's request, so the evidence shown in the table is requirement-specific.

For the live demo, the app is designed to be explainable rather than overly complex. Review classification uses the review text as the main decision signal and treats star rating as a confidence modifier. At startup, the app trains a small in-memory TF-IDF + Logistic Regression text model from the existing processed training split, blends it with the saved model, calibrates the blended probability on the existing validation split, and chooses a validation-based F1 threshold. These app metrics are validation/demo metrics, not final test metrics. It also prints accuracy, precision, recall, F1, and confusion-matrix counts in the terminal and shows the validation snapshot in the UI.

Product recommendations show matched concerns, matched features, ingredient checks, relevant snippets, cluster labels, and a short ranking reason. Avoided ingredients are optional personal constraints; if they conflict with a concern, the app filters those ingredients and shifts to reasonable alternatives instead of failing. If no concern is selected, the app treats it as a general recommendation and avoids over-prioritizing strong actives by default. Recommendation scores are scaled across the current candidate set so concern/category match, skin match, ingredient compatibility, review relevance, reliability, and sentiment produce differentiated results.

## Outputs

- `models/best_model.pkl`
- `models/preprocessing_pipeline.pkl`
- `models/helpful_review_ranker.pkl`
- `reports/data_audit_report.md`
- `reports/validation_model_comparison.csv`
- `reports/model_comparison.csv`
- `reports/final_summary.md`
- `reports/helpful_review_ranking_report.md`
- figures in `reports/figures/`

## How to Run

Run all commands from the project root folder:

```bash
cd "path/to/sephora-ml-project (demo)"
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run notebooks in order from the project root:

```bash
jupyter notebook
```

Optional simple Flask app:

```bash
python app.py
```

You can also run the same app as a module:

```bash
python -m simple_app_optional.app
```

Then open the local Flask URL printed in the terminal, usually `http://127.0.0.1:5000`.

The app builds file paths from the project folder, so model and data files are found reliably even when the app is launched through the root `app.py` file or the module command above.
