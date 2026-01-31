import os
import pandas as pd

# Download files from https://www.football-data.co.uk/portugalm.php
# and insert the corresponding directory below
DIRECTORY = '/Users/vcerq/Downloads/prob_ai/'
file_list = os.listdir(DIRECTORY)

COLUMNS = {
    'Date': 'date',
    'HomeTeam': 'ht',
    'AwayTeam': 'at',
    'FTHG': 'hg',
    'FTAG': 'ag',
    'FTR': 'result',
    'B365H': 'h_odds',
    'B365D': 'd_odds',
    'B365A': 'a_odds',
}

BAD_TEAM_NAMES = {
    'sp lisbon': 'sporting',
    'estrela': 'est amadora',
    'feirense ': 'feirense',
}

match_data_l = []
for file_name in file_list:
    print(file_name)
    df = pd.read_csv(DIRECTORY + file_name, on_bad_lines='warn')
    df = df[[*COLUMNS]]
    df = df.rename(columns=COLUMNS)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    df['ht'] = df['ht'].str.lower()
    df['at'] = df['at'].str.lower()

    df['ht'] = df['ht'].replace(BAD_TEAM_NAMES)
    df['at'] = df['at'].replace(BAD_TEAM_NAMES)
    df = df.dropna()

    df['hg'] = df['hg'].astype(int)
    df['ag'] = df['ag'].astype(int)

    match_data_l.append(df)

matches = pd.concat(match_data_l).sort_values('date').reset_index(drop=True)
matches.to_csv('project/dataset/matches.csv', index=False)
