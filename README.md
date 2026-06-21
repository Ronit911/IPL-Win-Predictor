# IPL Win Probability Predictor + Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![MySQL](https://img.shields.io/badge/MySQL-10.4-blue)

## Project Overview
This end-to-end data science project analyzes the Kaggle IPL Complete Dataset (2008-2020) and builds a machine learning model to predict the ball-by-ball win probability of a chasing team. It features an interactive Streamlit dashboard allowing users to simulate live match situations and explore team and player analytics directly from a MySQL database.

## Screenshots
<img width="2449" height="1245" alt="Screenshot 2026-06-21 141549" src="https://github.com/user-attachments/assets/465e992f-a5a4-4bc3-b236-92c6854dc139" />
<img width="2458" height="1308" alt="Screenshot 2026-06-21 141427" src="https://github.com/user-attachments/assets/82e4ff73-a2b6-4b54-9fd8-ba4f6f492bf2" />
<img width="2472" height="1259" alt="Screenshot 2026-06-21 141409" src="https://github.com/user-attachments/assets/d84ecd3a-ea12-414d-ac5c-385c39b568b8" />



- Simulator View
- Analytics View

## How to Run Locally

1. **Prerequisites:** 
   - Install Python 3.8+
   - Install XAMPP or an independent MySQL server.
   
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup:**
   Ensure MySQL is running. Update `config.py` with your MySQL credentials, then run:
   ```bash
   python db_setup.py
   ```

4. **Model Training & EDA (Optional):**
   *(The dataset and models are generated via the below scripts)*
   ```bash
   python run_eda.py
   python model.py
   ```

5. **Start Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## SQL Queries Summary
The `sql/queries.sql` script answers 10 analytical questions:
1. Win percentage by toss decision.
2. Top 10 venues by average 1st innings score.
3. Venues that favor chasing teams.
4. Top 10 death-overs bowlers by economy rate.
5. Team win percentage at each venue.
6. Average powerplay score by team.
7. Matches won by runs vs wickets per season.
8. Top 5 highest run scorers per season.
9. Most matches played at each venue.
10. Win percentage of batting first vs chasing overall.

## Model Performance Metrics

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| Logistic Regression (Baseline) | 78.71% | 0.4380 |
| XGBoost Classifier (Final) | 93.56% | 0.2109 |

## Key Insights from EDA
- **Toss Impact:** A Chi-square test confirmed that winning the toss significantly affects the match outcome (p < 0.05).
- **Chasing Preference:** Most venues strongly favor teams chasing a target, with distinct visual correlations observed.
- **Powerplay Importance:** A clear positive correlation exists between strong powerplay scores (first 6 overs) and the final total.
- **Death Overs:** The wicket fall frequency peaks significantly during the death overs (16-20) as batsmen take more risks.
- **Run Rate Progression:** Average run rates remain stable in middle overs but spike drastically towards the end of the innings.
