-- 1. Win % by toss decision (bat/field) per season
SELECT season, toss_decision, 
       SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) / COUNT(*) * 100 AS win_percentage
FROM matches 
GROUP BY season, toss_decision;

-- 2. Top 10 venues by average first innings score
SELECT m.venue, AVG(d.total_runs) as avg_first_innings_score
FROM matches m
JOIN (SELECT match_id, SUM(total_runs) as total_runs FROM deliveries WHERE inning = 1 GROUP BY match_id) d
ON m.id = d.match_id
GROUP BY m.venue
ORDER BY avg_first_innings_score DESC
LIMIT 10;

-- 3. Which venues favor chasing teams (win % when chasing)
SELECT venue, 
       SUM(CASE WHEN win_by_wickets > 0 THEN 1 ELSE 0 END) / COUNT(*) * 100 AS win_pct_chasing
FROM matches
GROUP BY venue
ORDER BY win_pct_chasing DESC;

-- 4. Top 10 bowlers in death overs (overs 16-20) by economy rate
SELECT bowler, 
       SUM(total_runs - legbye_runs - bye_runs) / (COUNT(CASE WHEN wide_runs = 0 AND noball_runs = 0 THEN 1 END) / 6.0) AS economy_rate
FROM deliveries
WHERE over_num > 15
GROUP BY bowler
HAVING COUNT(CASE WHEN wide_runs = 0 AND noball_runs = 0 THEN 1 END) > 60
ORDER BY economy_rate ASC
LIMIT 10;

-- 5. Team win % at each venue
SELECT venue, winner as team, 
       COUNT(*) as wins,
       (COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY venue)) * 100 as win_percentage
FROM matches
WHERE winner != 'No Result' AND winner IS NOT NULL
GROUP BY venue, winner
ORDER BY venue, win_percentage DESC;

-- 6. Average powerplay score (overs 1-6) by team
SELECT batting_team, AVG(pp_score) as avg_powerplay_score
FROM (
    SELECT match_id, batting_team, SUM(total_runs) as pp_score
    FROM deliveries
    WHERE over_num <= 6
    GROUP BY match_id, batting_team
) pp
GROUP BY batting_team
ORDER BY avg_powerplay_score DESC;

-- 7. Matches won by runs vs wickets per season
SELECT season,
       SUM(CASE WHEN win_by_runs > 0 THEN 1 ELSE 0 END) AS won_by_runs,
       SUM(CASE WHEN win_by_wickets > 0 THEN 1 ELSE 0 END) AS won_by_wickets
FROM matches
GROUP BY season
ORDER BY season;

-- 8. Top 5 highest run scorers per season
WITH PlayerScores AS (
    SELECT m.season, d.batsman, SUM(d.batsman_runs) as total_runs
    FROM deliveries d
    JOIN matches m ON d.match_id = m.id
    GROUP BY m.season, d.batsman
),
RankedScores AS (
    SELECT season, batsman, total_runs,
           DENSE_RANK() OVER(PARTITION BY season ORDER BY total_runs DESC) as rnk
    FROM PlayerScores
)
SELECT season, batsman, total_runs 
FROM RankedScores 
WHERE rnk <= 5
ORDER BY season, rnk;

-- 9. Most matches played at each venue
SELECT venue, COUNT(*) as total_matches
FROM matches
GROUP BY venue
ORDER BY total_matches DESC;

-- 10. Win % of teams batting first vs second overall
SELECT 
    SUM(CASE WHEN win_by_runs > 0 THEN 1 ELSE 0 END) / COUNT(*) * 100 AS win_pct_batting_first,
    SUM(CASE WHEN win_by_wickets > 0 THEN 1 ELSE 0 END) / COUNT(*) * 100 AS win_pct_batting_second
FROM matches
WHERE result != 'no result';
