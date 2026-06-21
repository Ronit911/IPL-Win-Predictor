import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import mysql.connector
from config import DB_CONFIG

st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# Connect to database
@st.cache_resource
def init_connection():
    return mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database='ipl_db'
    )

try:
    conn = init_connection()
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# Load Data for basic lists
@st.cache_data
def load_data():
    matches = pd.read_csv('data/matches.csv')
    return matches

@st.cache_data
def load_players():
    d = pd.read_csv('data/deliveries.csv')
    return sorted(list(set(d['batsman'].dropna().unique()) | set(d['bowler'].dropna().unique())))

matches = load_data()
players_list = load_players()

# Load Model
@st.cache_resource
def load_model():
    data = joblib.load('models/model.pkl')
    return data['model'], data['le_batting'], data['le_bowling'], data['le_venue']

model, le_batting, le_bowling, le_venue = load_model()

teams = sorted(list(set(matches['team1'].dropna().unique()) | set(matches['team2'].dropna().unique())))
venues = sorted(matches['venue'].dropna().unique().tolist())

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1200px-Indian_Premier_League_Official_Logo.svg.png", width=150)
page = st.sidebar.radio("Navigation", ["Live Match Simulator", "Team Analytics", "Player Stats"])

if page == "Live Match Simulator":
    st.title("🏏 Live Match Win Probability Simulator")
    
    col1, col2, col3 = st.columns(3)
    batting_team = col1.selectbox("Batting Team (Chasing)", teams, index=0)
    bowling_team = col2.selectbox("Bowling Team (Defending)", teams, index=1)
    venue = col3.selectbox("Venue", venues)
    
    target_score = st.number_input("Target Score (1st Innings Total + 1)", min_value=1, max_value=300, value=150)
    
    st.markdown("### Current Match Situation")
    c1, c2, c3 = st.columns(3)
    current_over = c1.slider("Current Over", 0, 20, 5)
    current_score = c2.slider("Runs Scored", 0, 300, 45)
    wickets_fallen = c3.slider("Wickets Fallen", 0, 10, 1)
    
    # Auto-calculate metrics
    runs_left = max(0, target_score - current_score)
    balls_left = max(0, 120 - (current_over * 6))
    wickets_in_hand = 10 - wickets_fallen
    crr = (current_score / current_over) if current_over > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0
    
    st.markdown(f"**Runs Left:** {runs_left} | **Balls Left:** {balls_left} | **CRR:** {crr:.2f} | **RRR:** {rrr:.2f}")
    
    if st.button("Predict Probability"):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams must be different!")
        else:
            try:
                # We mock powerplay_score as 45 for generic inputs if not fully available
                powerplay_score = 45 
                
                b_t = le_batting.transform([batting_team])[0]
                bo_t = le_bowling.transform([bowling_team])[0]
                v = le_venue.transform([venue])[0]
                
                input_data = pd.DataFrame({
                    'batting_team': [b_t],
                    'bowling_team': [bo_t],
                    'venue': [v],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_in_hand': [wickets_in_hand],
                    'current_run_rate': [crr],
                    'required_run_rate': [rrr],
                    'powerplay_score': [powerplay_score]
                })
                
                prob = model.predict_proba(input_data)[0]
                win_prob = prob[1]
                
                st.markdown("### Win Probability")
                colA, colB = st.columns(2)
                colA.metric(label=f"{batting_team} (Chasing)", value=f"{win_prob * 100:.1f}%")
                colB.metric(label=f"{bowling_team} (Defending)", value=f"{(1 - win_prob) * 100:.1f}%")
                
                st.progress(float(win_prob))
                
                # Maintain history for Win Probability Shift over Time
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                
                # Check if this over already exists, update it, else append
                st.session_state['history'] = [h for h in st.session_state['history'] if h['over'] != current_over]
                st.session_state['history'].append({'over': current_over, 'prob': win_prob * 100})
                
                hist_df = pd.DataFrame(st.session_state['history']).sort_values('over')
                
                st.markdown("### Win Probability Shift Over Time")
                if len(hist_df) > 1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=hist_df['over'], y=hist_df['prob'], mode='lines+markers', name='Win Prob', line=dict(color='royalblue', width=3)))
                    fig_hist.update_layout(xaxis_title="Over", yaxis_title="Win Probability (%)", yaxis_range=[0, 100])
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("💡 Change the sliders (e.g. increase Current Over and Runs Scored) and click 'Predict Probability' again to build a trend line!")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=hist_df['over'], y=hist_df['prob'], mode='markers', marker=dict(size=12, color='royalblue')))
                    fig_hist.update_layout(xaxis_title="Over", yaxis_title="Win Probability (%)", yaxis_range=[0, 100])
                    st.plotly_chart(fig_hist, use_container_width=True)
                
            except ValueError as e:
                st.error(f"Prediction error: One of the teams or venues is not recognized by the model. Try a different combination.")

elif page == "Team Analytics":
    st.title("📊 Team Analytics")
    team = st.selectbox("Select Team", teams)
    
    if st.button("Get Analytics"):
        cursor = conn.cursor(dictionary=True)
        
        # Win % by Venue
        cursor.execute(f"SELECT venue, COUNT(*) as wins FROM matches WHERE winner = '{team}' GROUP BY venue ORDER BY wins DESC LIMIT 10")
        venues_df = pd.DataFrame(cursor.fetchall())
        if not venues_df.empty:
            st.subheader("Top 10 Venues by Wins")
            st.bar_chart(venues_df.set_index('venue')['wins'])
            
        # Performance trend by season
        cursor.execute(f"SELECT season, COUNT(*) as wins FROM matches WHERE winner = '{team}' GROUP BY season ORDER BY season")
        season_df = pd.DataFrame(cursor.fetchall())
        if not season_df.empty:
            st.subheader("Wins per Season")
            st.line_chart(season_df.set_index('season')['wins'])
            
        # Head to Head Record (Heatmap style dataframe)
        cursor.execute(f"""
            SELECT team1, team2, winner 
            FROM matches 
            WHERE (team1 = '{team}' OR team2 = '{team}') AND winner != 'No Result'
        """)
        h2h_df = pd.DataFrame(cursor.fetchall())
        if not h2h_df.empty:
            h2h_df['Opponent'] = np.where(h2h_df['team1'] == team, h2h_df['team2'], h2h_df['team1'])
            h2h_df['Won'] = np.where(h2h_df['winner'] == team, 1, 0)
            h2h_summary = h2h_df.groupby('Opponent')['Won'].agg(Matches='count', Wins='sum').reset_index()
            h2h_summary['Win %'] = (h2h_summary['Wins'] / h2h_summary['Matches']) * 100
            
            st.subheader(f"Head-to-Head Win % Heatmap for {team}")
            st.dataframe(h2h_summary.style.background_gradient(cmap='coolwarm', subset=['Win %']), use_container_width=True)
            
        cursor.close()

elif page == "Player Stats":
    st.title("🏏 Player Stats")
    player = st.selectbox("Select Player Name", players_list)
    
    if st.button("Search") and player:
        cursor = conn.cursor(dictionary=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏏 Batting Stats")
            cursor.execute(f"""
                SELECT m.season, 
                       SUM(d.batsman_runs) as runs, 
                       COUNT(d.ball) as balls 
                FROM deliveries d 
                JOIN matches m ON d.match_id = m.id 
                WHERE d.batsman LIKE '%{player}%' 
                GROUP BY m.season 
                ORDER BY m.season
            """)
            batting_df = pd.DataFrame(cursor.fetchall())
            if not batting_df.empty:
                total_runs = int(batting_df['runs'].sum())
                total_balls = int(batting_df['balls'].sum())
                overall_sr = (total_runs / total_balls) * 100 if total_balls > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.metric("Total Runs", f"{total_runs}")
                c2.metric("Career Strike Rate", f"{overall_sr:.2f}")
                
                batting_df['Strike Rate'] = (batting_df['runs'] / batting_df['balls']) * 100
                st.markdown("**Season Breakdown**")
                st.dataframe(batting_df[['season', 'runs', 'Strike Rate']].set_index('season').style.format({'Strike Rate': '{:.2f}'}), use_container_width=True)
            else:
                st.info("No batting records found.")
                
        with col2:
            st.subheader("🎾 Bowling Stats")
            cursor.execute(f"""
                SELECT m.season, 
                       SUM(d.total_runs - d.legbye_runs - d.bye_runs) as runs_conceded, 
                       COUNT(CASE WHEN d.wide_runs = 0 AND d.noball_runs = 0 THEN 1 END) as legal_balls,
                       SUM(CASE WHEN d.player_dismissed != '' AND d.dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END) as wickets
                FROM deliveries d 
                JOIN matches m ON d.match_id = m.id 
                WHERE d.bowler LIKE '%{player}%' 
                GROUP BY m.season 
                ORDER BY m.season
            """)
            bowling_df = pd.DataFrame(cursor.fetchall())
            if not bowling_df.empty:
                total_wickets = int(bowling_df['wickets'].sum())
                total_runs_c = float(bowling_df['runs_conceded'].sum())
                total_legal_balls = float(bowling_df['legal_balls'].sum())
                overall_econ = total_runs_c / (total_legal_balls / 6.0) if total_legal_balls > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.metric("Total Wickets", f"{total_wickets}")
                c2.metric("Career Economy", f"{overall_econ:.2f}")
                
                bowling_df['Economy'] = bowling_df['runs_conceded'].astype(float) / (bowling_df['legal_balls'].astype(float) / 6.0)
                st.markdown("**Season Breakdown**")
                st.dataframe(bowling_df[['season', 'wickets', 'Economy']].set_index('season').style.format({'Economy': '{:.2f}'}), use_container_width=True)
            else:
                st.info("No bowling records found.")
                
        cursor.close()
