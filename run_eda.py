import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import json
import os

matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

# Distribution of first innings totals by venue
first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
first_innings = first_innings.merge(matches[['id', 'venue']], left_on='match_id', right_on='id')
plt.figure(figsize=(12, 6))
sns.histplot(data=first_innings, x='total_runs', bins=30, kde=True)
plt.title('Distribution of First Innings Totals')
plt.xlabel('Total Runs')
plt.ylabel('Frequency')
plt.savefig('plots/first_innings_distribution.png')
plt.close()

# Run rate progression curves by innings
overs = deliveries.groupby(['match_id', 'inning', 'over'])['total_runs'].sum().reset_index()
overs['cumulative_runs'] = overs.groupby(['match_id', 'inning'])['total_runs'].cumsum()
overs['run_rate'] = overs['cumulative_runs'] / overs['over']
plt.figure(figsize=(12, 6))
sns.lineplot(data=overs, x='over', y='run_rate', hue='inning', estimator='mean', errorbar=None)
plt.title('Run Rate Progression by Innings')
plt.xlabel('Over')
plt.ylabel('Average Run Rate')
plt.legend(title='Inning')
plt.savefig('plots/run_rate_progression.png')
plt.close()

# Powerplay score vs final total correlation
pp = deliveries[deliveries['over'] <= 6].groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
pp.rename(columns={'total_runs': 'powerplay_score'}, inplace=True)
final = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
final.rename(columns={'total_runs': 'final_total'}, inplace=True)
merged = pd.merge(pp, final, on=['match_id', 'inning'])
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='powerplay_score', y='final_total', alpha=0.5)
plt.title('Powerplay Score vs Final Total')
plt.xlabel('Powerplay Score (Overs 1-6)')
plt.ylabel('Final Total')
plt.savefig('plots/powerplay_vs_final.png')
plt.close()

# Wickets fallen per over across all matches
wickets = deliveries[deliveries['player_dismissed'].notna() & (deliveries['player_dismissed'] != '')]
wickets_by_over = wickets.groupby('over')['player_dismissed'].count().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=wickets_by_over, x='over', y='player_dismissed', color='salmon')
plt.title('Wickets Fallen per Over Across All Matches')
plt.xlabel('Over')
plt.ylabel('Total Wickets')
plt.savefig('plots/wickets_per_over.png')
plt.close()

# Chi-square hypothesis test: does winning toss significantly affect match outcome?
print("Null Hypothesis (H0): Winning the toss does NOT significantly affect the match outcome.")
matches['toss_winner_won'] = np.where(matches['toss_winner'] == matches['winner'], 'Yes', 'No')
contingency_table = pd.crosstab(matches['toss_decision'], matches['toss_winner_won'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion: Reject H0. Winning the toss significantly affects the match outcome.")
else:
    print("Conclusion: Fail to reject H0. Winning the toss does not significantly affect the match outcome.")

# Top 10 batsmen by strike rate (min 200 balls faced)
batsman_stats = deliveries.groupby('batsman').agg(runs=('batsman_runs', 'sum'), balls=('ball', 'count')).reset_index()
batsman_stats = batsman_stats[batsman_stats['balls'] >= 200]
batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls']) * 100
top_10_sr = batsman_stats.sort_values(by='strike_rate', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_sr, x='strike_rate', y='batsman', hue='batsman', palette='viridis', legend=False)
plt.title('Top 10 Batsmen by Strike Rate (Min 200 balls)')
plt.xlabel('Strike Rate')
plt.ylabel('Batsman')
plt.savefig('plots/top10_strike_rate.png')
plt.close()

# Heatmap of team vs team win/loss record
matches_valid = matches[matches['result'] != 'no result'].copy()
teams = sorted(list(set(matches_valid['team1'].unique()) | set(matches_valid['team2'].unique())))
heatmap_data = pd.DataFrame(np.nan, index=teams, columns=teams)
for t1 in teams:
    for t2 in teams:
        if t1 == t2:
            continue
        subset = matches_valid[((matches_valid['team1']==t1) & (matches_valid['team2']==t2)) | ((matches_valid['team1']==t2) & (matches_valid['team2']==t1))]
        if len(subset) > 0:
            wins = len(subset[subset['winner'] == t1])
            heatmap_data.loc[t1, t2] = wins / len(subset)
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar_kws={'label': 'Win Percentage for Row Team'})
plt.title('Team vs Team Win/Loss Record (Win %)')
plt.savefig('plots/team_vs_team_heatmap.png')
plt.close()

# Generate Jupyter Notebook
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# IPL EDA and Statistics\n", "Exploratory Data Analysis and Statistics for IPL Matches (2008-2020)."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy.stats import chi2_contingency\n\n",
            "matches = pd.read_csv('../data/matches.csv')\n",
            "deliveries = pd.read_csv('../data/deliveries.csv')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Distribution of first innings totals by venue"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()\n",
            "first_innings = first_innings.merge(matches[['id', 'venue']], left_on='match_id', right_on='id')\n",
            "plt.figure(figsize=(12, 6))\n",
            "sns.histplot(data=first_innings, x='total_runs', bins=30, kde=True)\n",
            "plt.title('Distribution of First Innings Totals')\n",
            "plt.xlabel('Total Runs')\n",
            "plt.ylabel('Frequency')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Run rate progression curves by innings"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "overs = deliveries.groupby(['match_id', 'inning', 'over'])['total_runs'].sum().reset_index()\n",
            "overs['cumulative_runs'] = overs.groupby(['match_id', 'inning'])['total_runs'].cumsum()\n",
            "overs['run_rate'] = overs['cumulative_runs'] / overs['over']\n",
            "plt.figure(figsize=(12, 6))\n",
            "sns.lineplot(data=overs, x='over', y='run_rate', hue='inning', estimator='mean', errorbar=None)\n",
            "plt.title('Run Rate Progression by Innings')\n",
            "plt.xlabel('Over')\n",
            "plt.ylabel('Average Run Rate')\n",
            "plt.legend(title='Inning')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Powerplay score vs final total correlation"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pp = deliveries[deliveries['over'] <= 6].groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()\n",
            "pp.rename(columns={'total_runs': 'powerplay_score'}, inplace=True)\n",
            "final = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()\n",
            "final.rename(columns={'total_runs': 'final_total'}, inplace=True)\n",
            "merged = pd.merge(pp, final, on=['match_id', 'inning'])\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.scatterplot(data=merged, x='powerplay_score', y='final_total', alpha=0.5)\n",
            "plt.title('Powerplay Score vs Final Total')\n",
            "plt.xlabel('Powerplay Score (Overs 1-6)')\n",
            "plt.ylabel('Final Total')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Wickets fallen per over across all matches"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "wickets = deliveries[deliveries['player_dismissed'].notna() & (deliveries['player_dismissed'] != '')]\n",
            "wickets_by_over = wickets.groupby('over')['player_dismissed'].count().reset_index()\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(data=wickets_by_over, x='over', y='player_dismissed', color='salmon')\n",
            "plt.title('Wickets Fallen per Over Across All Matches')\n",
            "plt.xlabel('Over')\n",
            "plt.ylabel('Total Wickets')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Chi-square hypothesis test (Toss Decision)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Null Hypothesis (H0): Winning the toss does NOT significantly affect the match outcome.\")\n",
            "matches['toss_winner_won'] = np.where(matches['toss_winner'] == matches['winner'], 'Yes', 'No')\n",
            "contingency_table = pd.crosstab(matches['toss_decision'], matches['toss_winner_won'])\n",
            "chi2, p_value, dof, expected = chi2_contingency(contingency_table)\n",
            "print(f\"Chi-square statistic: {chi2:.4f}\")\n",
            "print(f\"P-value: {p_value:.4f}\")\n",
            "if p_value < 0.05:\n",
            "    print(\"Conclusion: Reject H0. Winning the toss significantly affects the match outcome.\")\n",
            "else:\n",
            "    print(\"Conclusion: Fail to reject H0. Winning the toss does not significantly affect the match outcome.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Top 10 batsmen by strike rate (min 200 balls faced)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "batsman_stats = deliveries.groupby('batsman').agg(runs=('batsman_runs', 'sum'), balls=('ball', 'count')).reset_index()\n",
            "batsman_stats = batsman_stats[batsman_stats['balls'] >= 200]\n",
            "batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls']) * 100\n",
            "top_10_sr = batsman_stats.sort_values(by='strike_rate', ascending=False).head(10)\n",
            "plt.figure(figsize=(12, 6))\n",
            "sns.barplot(data=top_10_sr, x='strike_rate', y='batsman', hue='batsman', palette='viridis', legend=False)\n",
            "plt.title('Top 10 Batsmen by Strike Rate (Min 200 balls)')\n",
            "plt.xlabel('Strike Rate')\n",
            "plt.ylabel('Batsman')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Heatmap of team vs team win/loss record"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "matches_valid = matches[matches['result'] != 'no result'].copy()\n",
            "teams = sorted(list(set(matches_valid['team1'].unique()) | set(matches_valid['team2'].unique())))\n",
            "heatmap_data = pd.DataFrame(np.nan, index=teams, columns=teams)\n",
            "for t1 in teams:\n",
            "    for t2 in teams:\n",
            "        if t1 == t2:\n",
            "            continue\n",
            "        subset = matches_valid[((matches_valid['team1']==t1) & (matches_valid['team2']==t2)) | ((matches_valid['team1']==t2) & (matches_valid['team2']==t1))]\n",
            "        if len(subset) > 0:\n",
            "            wins = len(subset[subset['winner'] == t1])\n",
            "            heatmap_data.loc[t1, t2] = wins / len(subset)\n",
            "plt.figure(figsize=(12, 10))\n",
            "sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar_kws={'label': 'Win Percentage for Row Team'})\n",
            "plt.title('Team vs Team Win/Loss Record (Win %)')\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/eda.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("EDA script completed successfully.")
