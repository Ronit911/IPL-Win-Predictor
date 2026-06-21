import pandas as pd
import numpy as np
import mysql.connector
from config import DB_CONFIG

def create_database():
    """Connect to MySQL server and create the ipl_db database."""
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS ipl_db")
    cursor.close()
    conn.close()

def clean_data(matches_path, deliveries_path):
    """Load and clean datasets, handling missing values and data types."""
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    # Handle missing values in matches
    matches['city'] = matches['city'].fillna('Unknown')
    matches['winner'] = matches['winner'].fillna('No Result')
    matches['player_of_match'] = matches['player_of_match'].fillna('None')
    for col in ['umpire1', 'umpire2', 'umpire3']:
        matches[col] = matches[col].fillna('Unknown')
    
    # Handle missing values in deliveries
    deliveries['player_dismissed'] = deliveries['player_dismissed'].fillna('')
    deliveries['dismissal_kind'] = deliveries['dismissal_kind'].fillna('')
    deliveries['fielder'] = deliveries['fielder'].fillna('')
    
    # Replace Pandas NaN with None so MySQL connector interprets them as NULL
    matches = matches.replace({np.nan: None})
    deliveries = deliveries.replace({np.nan: None})
    
    return matches, deliveries

def setup_tables(conn):
    """Create the matches and deliveries tables in the database."""
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS deliveries")
    cursor.execute("DROP TABLE IF EXISTS matches")
    
    # Create matches table
    matches_table = """
    CREATE TABLE matches (
        id INT PRIMARY KEY,
        season INT,
        city VARCHAR(100),
        date VARCHAR(50),
        team1 VARCHAR(100),
        team2 VARCHAR(100),
        toss_winner VARCHAR(100),
        toss_decision VARCHAR(50),
        result VARCHAR(50),
        dl_applied INT,
        winner VARCHAR(100),
        win_by_runs INT,
        win_by_wickets INT,
        player_of_match VARCHAR(100),
        venue VARCHAR(100),
        umpire1 VARCHAR(100),
        umpire2 VARCHAR(100),
        umpire3 VARCHAR(100)
    )
    """
    cursor.execute(matches_table)
    
    # Create deliveries table
    deliveries_table = """
    CREATE TABLE deliveries (
        match_id INT,
        inning INT,
        batting_team VARCHAR(100),
        bowling_team VARCHAR(100),
        over_num INT,
        ball INT,
        batsman VARCHAR(100),
        non_striker VARCHAR(100),
        bowler VARCHAR(100),
        is_super_over INT,
        wide_runs INT,
        bye_runs INT,
        legbye_runs INT,
        noball_runs INT,
        penalty_runs INT,
        batsman_runs INT,
        extra_runs INT,
        total_runs INT,
        player_dismissed VARCHAR(100),
        dismissal_kind VARCHAR(50),
        fielder VARCHAR(100),
        FOREIGN KEY (match_id) REFERENCES matches(id)
    )
    """
    cursor.execute(deliveries_table)
    conn.commit()
    cursor.close()

def insert_data(conn, matches, deliveries):
    """Insert data into the tables."""
    cursor = conn.cursor()
    
    # Insert matches
    print("Inserting matches...")
    match_query = "INSERT INTO matches VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    match_data = [tuple(x) for x in matches.to_numpy()]
    cursor.executemany(match_query, match_data)
    
    # Insert deliveries
    print("Inserting deliveries...")
    deliv_query = "INSERT INTO deliveries VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    deliv_data = [tuple(x) for x in deliveries.to_numpy()]
    
    chunk_size = 1000
    for i in range(0, len(deliv_data), chunk_size):
        cursor.executemany(deliv_query, deliv_data[i:i+chunk_size])
        conn.commit()
        print(f"Inserted {min(i+chunk_size, len(deliv_data))}/{len(deliv_data)} deliveries")
        
    cursor.close()

if __name__ == '__main__':
    print("Starting database setup...")
    create_database()
    
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database='ipl_db'
    )
    
    print("Cleaning data...")
    matches, deliveries = clean_data('data/matches.csv', 'data/deliveries.csv')
    
    print("Setting up tables...")
    setup_tables(conn)
    
    insert_data(conn, matches, deliveries)
    conn.close()
    print("Database setup complete.")
