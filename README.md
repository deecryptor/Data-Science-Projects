# General Assembly Data Science Projects

Projects done mainly during my time in the General Assembly Data Science Immersive program, from exploratory analysis through regression, NLP classification and time series forecasting. I have done a revised capstone to attempt combining machine learning with generative AI.

**Featured: [Capstone: Short Video Hook Lab](05_Capstone-Hook-Lab).** A model that predicts which YouTube Shorts will beat their channel's usual views from decisions made before posting, plus an AI title rewriter (Google Gemini) scored by that model.

| # | Project | Summary | Key libraries / tools | Models |
|---|---------|---------|-----------------------|--------|
| 01 | [Exploring Climate Data of Singapore](01_Climate-and-Electricity) | Explores whether rainfall, temperature, sunshine and humidity are linked to household electricity use in Singapore. | Python, Jupyter, pandas, NumPy, Matplotlib, Seaborn | None (EDA) |
| 02 | [Predicting HDB Resale Prices](02_HDB-Resale-Prices) | Predicts Singapore HDB resale flat prices from flat, location and amenity features. Ridge regression reaches a test R² of 0.93 (RMSE about S$36,400). Submitted to a Kaggle competition. | pandas, NumPy, scikit-learn, statsmodels, SciPy, Seaborn | Linear, Ridge and Lasso regression |
| 03 | [Toxic Comment Detection on YouTube](03_Toxic-Comment-Detection) | Classifies 40,000 YouTube comments, collected with the YouTube Data API and labelled with Detoxify. 79% precision and 74% recall on the toxic class, checked against 300 comments labelled by hand. | YouTube Data API, Detoxify (PyTorch), NLTK, scikit-learn, pandas | Naive Bayes, Logistic Regression, Random Forest |
| 04 | [Predicting Dengue Cases](04_Dengue-Forecasting) | Forecasts weekly dengue cases in Singapore from case, weather and Google Trends data (MAPE 0.10), with a cost-benefit analysis of outreach manpower (about S$416k savings per half year). Group project. | pandas, statsmodels, pmdarima, scikit-learn, Matplotlib | ARIMA, SARIMA |
| 05 | [Capstone: Short Video Hook Lab](05_Capstone-Hook-Lab) | Predicts whether a YouTube Short will beat its channel's typical views, using 2,740 Shorts from 11 health and longevity creators (random forest, test ROC-AUC 0.60), and rewrites weak titles with Gemini. | YouTube Data API, Google Gemini (google-genai), scikit-learn, pandas, joblib | Logistic Regression, Random Forest |

Each folder has its own README with the problem statement, approach, findings and charts, and notebooks that can be re-run from the included data (see each README for any data that has to be downloaded separately).
