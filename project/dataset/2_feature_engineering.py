import numpy as np
import pandas as pd


def compute_features(match_data: pd.DataFrame, n_matches=5):
    """
    Compute a set of features for each match
    """

    # Initialize feature columns
    features = {
        'ht_form_points': [],
        'ht_form_goals_scored': [],
        'ht_form_goals_conceded': [],
        'ht_form_wins': [],
        'at_form_points': [],
        'at_form_goals_scored': [],
        'at_form_goals_conceded': [],
        'at_form_wins': [],
        'ht_home_form_points': [],
        'ht_home_form_goals_scored': [],
        'at_away_form_points': [],
        'at_away_form_goals_scored': [],
        'h2h_home_wins': [],
        'h2h_draws': [],
        'h2h_away_wins': [],
        'h2h_total_matches': [],
    }

    for idx, row in match_data.iterrows():
        match_date = row['date']
        home_team = row['ht']
        away_team = row['at']

        past = match_data.iloc[:idx]

        ht_past = past[(past['ht'] == home_team) | (past['at'] == home_team)].tail(n_matches)
        at_past = past[(past['ht'] == away_team) | (past['at'] == away_team)].tail(n_matches)

        features['ht_form_points'].append(calc_team_points(ht_past, home_team))
        features['ht_form_goals_scored'].append(calc_team_goals_scored(ht_past, home_team))
        features['ht_form_goals_conceded'].append(calc_team_goals_conceded(ht_past, home_team))
        features['ht_form_wins'].append(calc_team_wins(ht_past, home_team))
        features['at_form_points'].append(calc_team_points(at_past, away_team))
        features['at_form_goals_scored'].append(calc_team_goals_scored(at_past, away_team))
        features['at_form_goals_conceded'].append(calc_team_goals_conceded(at_past, away_team))
        features['at_form_wins'].append(calc_team_wins(at_past, away_team))

        ht_home_past = past[past['ht'] == home_team].tail(n_matches)
        at_away_past = past[past['at'] == away_team].tail(n_matches)

        features['ht_home_form_points'].append(calc_home_points(ht_home_past))
        features['ht_home_form_goals_scored'].append(ht_home_past['hg'].sum() if len(ht_home_past) > 0 else np.nan)

        features['at_away_form_points'].append(calc_away_points(at_away_past))
        features['at_away_form_goals_scored'].append(at_away_past['ag'].sum() if len(at_away_past) > 0 else np.nan)

        h2h = past[((past['ht'] == home_team) & (past['at'] == away_team)) |
                   ((past['ht'] == away_team) & (past['at'] == home_team))]

        h2h_hw, h2h_d, h2h_aw = calc_h2h(h2h, home_team, away_team)

        features['h2h_home_wins'].append(h2h_hw)
        features['h2h_draws'].append(h2h_d)
        features['h2h_away_wins'].append(h2h_aw)
        features['h2h_total_matches'].append(len(h2h))

    for col, values in features.items():
        match_data[col] = values

    return match_data


def calc_team_points(df, team):
    """Calculate total points for a team in given matches."""
    if len(df) == 0:
        return np.nan

    points = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            if row['result'] == 'H':
                points += 3
            elif row['result'] == 'D':
                points += 1
        else:
            if row['result'] == 'A':
                points += 3
            elif row['result'] == 'D':
                points += 1
    return points


def calc_team_wins(df, team):
    """Calculate number of wins for a team."""
    if len(df) == 0:
        return np.nan
    wins = 0
    for _, row in df.iterrows():
        if row['ht'] == team and row['result'] == 'H':
            wins += 1
        elif row['at'] == team and row['result'] == 'A':
            wins += 1
    return wins


def calc_team_goals_scored(df, team):
    """Calculate goals scored by team."""
    if len(df) == 0:
        return np.nan
    goals = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            goals += row['hg']
        else:
            goals += row['ag']
    return goals


def calc_team_goals_conceded(df, team):
    """Calculate goals conceded by team."""
    if len(df) == 0:
        return np.nan
    goals = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            goals += row['ag']
        else:
            goals += row['hg']
    return goals


def calc_home_points(df):
    """Points from home matches only."""
    if len(df) == 0:
        return np.nan
    return (df['result'] == 'H').sum() * 3 + (df['result'] == 'D').sum()


def calc_away_points(df):
    """Points from away matches only."""
    if len(df) == 0:
        return np.nan
    return (df['result'] == 'A').sum() * 3 + (df['result'] == 'D').sum()


def calc_h2h(df, home_team, away_team):
    """Calculate head-to-head record (from perspective of home_team)."""
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    home_wins = 0
    draws = 0
    away_wins = 0
    for _, row in df.iterrows():
        if row['result'] == 'D':
            draws += 1
        elif row['ht'] == home_team:
            if row['result'] == 'H':
                home_wins += 1
            else:
                away_wins += 1
        else:  # home_team was away in this h2h match
            if row['result'] == 'A':
                home_wins += 1
            else:
                away_wins += 1
    return home_wins, draws, away_wins


def discretize_features(df):
    """Discretize numeric features."""

    df['ht_form'] = pd.cut(
        df['ht_form_points'],
        bins=[-1, 3, 6, 9, 12, 15],
        labels=['poor', 'below avg', 'average', 'good', 'excellent']
    )
    df['at_form'] = pd.cut(
        df['at_form_points'],
        bins=[-1, 3, 6, 9, 12, 15],
        labels=['poor', 'below avg', 'average', 'good', 'excellent']
    )

    df['ht_attack'] = pd.cut(
        df['ht_form_goals_scored'],
        bins=[-1, 3, 7, 10, 100],
        labels=['weak', 'average', 'above avg', 'strong']
    )
    df['at_attack'] = pd.cut(
        df['at_form_goals_scored'],
        bins=[-1, 3, 7, 10, 100],
        labels=['weak', 'average', 'above avg', 'strong']
    )

    df['joint_attack'] = df['ht_attack'].astype(str) + " x " + df['at_attack'].astype(str)

    df['h2h_dominance'] = df.apply(
        lambda r: 'home_dominant' if r['h2h_home_wins'] > r['h2h_away_wins'] + 3
        else ('away_dominant' if r['h2h_away_wins'] > r['h2h_home_wins'] + 3
              else 'balanced'), axis=1
    )

    df['odds_favorite'] = df.apply(
        lambda r: 'home' if r['h_odds'] < r['a_odds'] - 0.5
        else ('away' if r['a_odds'] < r['h_odds'] - 0.5 else 'balanced'), axis=1
    )

    return df


#
matches = pd.read_csv('project/dataset/matches.csv', parse_dates=['date'])

matches = compute_features(matches, n_matches=5)
matches = matches.loc[matches['date'].dt.year > 2005, :].reset_index(drop=True)
matches = matches.fillna(0)
matches = matches.drop(columns=['h2h_total_matches'])

matches = discretize_features(matches)

matches['ht_form'].value_counts()
matches['at_form'].value_counts()
matches['ht_attack'].value_counts()
matches['joint_attack'].value_counts()
matches['at_attack'].value_counts()
matches['h2h_dominance'].value_counts()
matches['odds_favorite'].value_counts()

matches.to_csv('project/dataset/dataset.csv', index=False)

# import pandas as pd
# matches = pd.read_csv('project/dataset/dataset.csv', parse_dates=['date'])
# matches.iloc[0]
