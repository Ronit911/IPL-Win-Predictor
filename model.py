import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib
import matplotlib.pyplot as plt

print("Loading data...")
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

print("Feature Engineering...")
# Get total runs scored in 1st innings to set the target for 2nd innings
first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
first_innings.rename(columns={'total_runs': 'target'}, inplace=True)
first_innings['target'] = first_innings['target'] + 1

# Merge target with 2nd innings deliveries
second_innings = deliveries[deliveries['inning'] == 2].merge(first_innings, on='match_id')
second_innings = second_innings.merge(matches[['id', 'venue', 'winner']], left_on='match_id', right_on='id')

# Calculate current score, runs left, balls left
second_innings['current_score'] = second_innings.groupby('match_id')['total_runs'].cumsum()
second_innings['runs_left'] = second_innings['target'] - second_innings['current_score']
second_innings['runs_left'] = second_innings['runs_left'].apply(lambda x: 0 if x < 0 else x)

second_innings['balls_bowled'] = (second_innings['over'] - 1) * 6 + second_innings['ball']
second_innings['balls_left'] = 120 - second_innings['balls_bowled']
second_innings['balls_left'] = second_innings['balls_left'].apply(lambda x: 0 if x < 0 else x)

# Wickets in hand
second_innings['player_dismissed'] = second_innings['player_dismissed'].fillna('')
second_innings['is_wicket'] = (second_innings['player_dismissed'] != '').astype(int)
second_innings['wickets_fallen'] = second_innings.groupby('match_id')['is_wicket'].cumsum()
second_innings['wickets_in_hand'] = 10 - second_innings['wickets_fallen']

# Current Run Rate (CRR)
second_innings['current_run_rate'] = (second_innings['current_score'] * 6) / second_innings['balls_bowled']

# Required Run Rate (RRR)
second_innings['required_run_rate'] = np.where(
    second_innings['balls_left'] > 0,
    (second_innings['runs_left'] * 6) / second_innings['balls_left'],
    0
)

# Powerplay score (1st innings score in first 6 overs)
pp_first = deliveries[(deliveries['inning'] == 1) & (deliveries['over'] <= 6)].groupby('match_id')['total_runs'].sum().reset_index()
pp_first.rename(columns={'total_runs': 'powerplay_score'}, inplace=True)
second_innings = second_innings.merge(pp_first, on='match_id', how='left')

# Target variable: 1 if batting_team won, 0 otherwise
second_innings['result'] = (second_innings['batting_team'] == second_innings['winner']).astype(int)

# Filter missing values
df = second_innings.dropna(subset=['winner', 'venue'])
features = ['batting_team', 'bowling_team', 'venue', 'runs_left', 'balls_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'powerplay_score', 'result']
df = df[features].dropna()

print("Encoding categorical features...")
le_batting = LabelEncoder()
le_bowling = LabelEncoder()
le_venue = LabelEncoder()

# Fit transforms
df['batting_team'] = le_batting.fit_transform(df['batting_team'])
df['bowling_team'] = le_bowling.fit_transform(df['bowling_team'])
df['venue'] = le_venue.fit_transform(df['venue'])

X = df.drop('result', axis=1)
y = df['result']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression (baseline)...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"Logistic Regression Log Loss: {log_loss(y_test, lr_probs):.4f}")

print("\nTraining XGBoost Classifier (final model)...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"XGBoost Log Loss: {log_loss(y_test, xgb_probs):.4f}")

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_preds))

print("Saving models and encoders...")
joblib.dump({
    'model': xgb,
    'le_batting': le_batting,
    'le_bowling': le_bowling,
    'le_venue': le_venue
}, 'models/model.pkl')
print("Model saved to models/model.pkl")

print("Plotting Feature Importance...")
feature_importances = xgb.feature_importances_
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Feature Importance (XGBoost)')
plt.savefig('plots/feature_importance.png')
plt.close()
