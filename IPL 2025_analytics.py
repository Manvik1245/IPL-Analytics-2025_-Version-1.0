import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
base_dir = os.path.dirname(os.path.abspath(__file__))
data_bat= os.path.join(base_dir,'ipl_batsman.csv')
data_matches= os.path.join(base_dir, 'matches.csv')
data_bowl = os.path.join(base_dir, 'ipl_bowler.csv')

def display_values(num):
    pd.set_option('display.max_rows', num)
    
try:
    df_bat= pd.read_csv(data_bat)
    df_matches= pd.read_csv(data_matches)
    df_bowler= pd.read_csv(data_bowl)
except FileNotFoundError:
    print("Error: file cannot be found! ")

def strike_rate():
    strike_sum= df_bat.loc[(df_bat.strike_rate>= 150.000)].groupby("striker").strike_rate.mean().sort_values(ascending= False).count()
    print("Total number of people with a 150+  average strike rate are:- "+  str(strike_sum))
    strike= df_bat.loc[(df_bat.strike_rate>= 150.000)].groupby("striker").strike_rate.mean().sort_values(ascending= False)

def accumalators():
    accumalate= df_bat.groupby("striker").runs_scored.sum().sort_values(ascending=False)
    print(accumalate.head(5))

def dismissals(name):
    total_dismissals= df_bat.loc[(df_bat.striker == name) & (df_bat.dismissal_type != "not out")].groupby("striker").dismissal_type.value_counts().sum()
    print("The total dismissals for "+ name + " are "+ str(total_dismissals) + " dismissals ")
    print("Dismissals type for "+ name)
    dismissal_type= df_bat.loc[(df_bat.striker == name) & (df_bat.dismissal_type != "not out")].groupby("striker").dismissal_type.value_counts()
    print(dismissal_type)
    most_common= df_bat.loc[(df_bat.striker == name) & (df_bat.dismissal_type != "not out")].groupby("striker").dismissal_type.value_counts().idxmax()
    print("Most common dismissal type for " + name + " is " + most_common)

def most_boundaries():
    fours_scored= df_bat.groupby("striker")['fours'].sum()
    sixex_scored= df_bat.groupby("striker")['sixes'].sum()
    total_runs= fours_scored+ sixex_scored
    print("Batsmen who scored the most boundaries are :-")
    print((total_runs).sort_values(ascending= False).head(5))

def most_runs_boundaries():
    fours_scored= df_bat.groupby("striker")['fours'].sum()
    sixex_scored= df_bat.groupby("striker")['sixes'].sum()
    total_runs= (fours_scored*4)+ (sixex_scored*6)
    print("Batsmen who scored the most runs in boundaries are :-")
    print((total_runs).sort_values(ascending= False).head(5))

def conversion_to_fifty():
    matches_played= df_bat.groupby("striker")['match_id'].count()
    fifty= df_bat.loc[df_bat.runs_scored>=50.00].groupby('striker')['match_id'].count()
    conversion= (((fifty/matches_played)*100).fillna(0))
    display= conversion.sort_values(ascending= False)  
    print("Batsmen with the highest conversion rate are:- ")        
    print(display.head(5))

def most_runs_scored():
    most= df_bat.groupby("striker")['runs_scored'].max().sort_values(ascending=False)
    print("The highest scores in thi IPL are:- ")
    print(most.head(5))
                                         
def batting_impact():
    total_runs= df_bat.groupby("striker").runs_scored.sum()
    count= df_bat.groupby("striker").runs_scored.count()
    average_runs= total_runs/count
    balls= df_bat.groupby("striker")["balls_faced"].sum()
    strike= ((total_runs/balls)*100)
    impact= (average_runs+ (average_runs*((strike-100)/100))).sort_values(ascending=False)
    print("The batters with the highest batting impact are:- ")
    print(impact.head(5))

def BPD():
    total_runs = df_bat.groupby("striker").fillna(0).runs_scored.sum()
    count = df_bat.groupby("striker").fillna(1).runs_scored.count()
    average_runs = total_runs / count
    strike = df_bat.groupby("striker").strike_rate.mean()
    df_bat['strike_rate'] = df_bat['strike_rate'].replace([np.inf, -np.inf], 0)
    bpd = (((100 * average_runs) / strike)).replace([np.inf, -np.inf], 0).sort_values(ascending=False)
    return bpd

def effective_scoring():
    total_runs= df_bat.groupby("striker").fillna(1).runs_scored.sum()
    count= df_bat.groupby("striker").fillna(0).runs_scored.count()
    average_runs= total_runs/count
    bpd= BPD()
    print("The batsmen with the most effective scoring are:- ")
    r= ((average_runs/(bpd-1))).sort_values(ascending= False)
    print(r.head(5))

def impact_in_wins():
    batsmen = df_bat.set_index('match_id')
    matches = df_matches.set_index('match_id')
    impact=batsmen.join(matches, lsuffix= '_bat', rsuffix='_match')
    accumalate= impact[impact['player_team'] == impact['match_winner']]
    win_person= accumalate.groupby(['striker', 'player_team'])['runs_scored'].sum()
    team= accumalate.groupby('player_team')['runs_scored'].sum()
    print("The batsmen with the most  percentage impact in their team wins are:- ")
    most= (win_person/win_person.index.get_level_values('player_team').map(team)*100).sort_values(ascending= False)
    print(most)

def most_impactful_bat(team):
    batsmen = df_bat.set_index('match_id')
    matches = df_matches.set_index('match_id')
    impact = batsmen.join(matches, lsuffix='_bat', rsuffix='_match')
    accumalate = impact[(impact['player_team'] == impact['match_winner']) & (impact['player_team'] == team)]
    win_person = accumalate.groupby(['striker', 'player_team'])['runs_scored'].sum()
    team_win = accumalate.groupby('player_team')['runs_scored'].sum()
    most = (win_person / win_person.index.get_level_values('player_team').map(team_win) * 100)
    values = most.values.tolist()
    names = [str(name) for name in most.index]
    fig = plt.figure(figsize=(10, 10))
    plt.pie(values, labels=names)
    plt.title("Batsman with most impact for " + team)
    plt.show()

def allrounder():
  accumalate= df_bat.groupby("striker").runs_scored.sum()
  wicket= df_bowler.groupby("bowler").wickets_taken.sum()
  bat_df = accumalate.to_frame(name='runs')
  bowl_df = wicket.to_frame(name='wickets')
  joined= bat_df.join(bowl_df, how='inner')
  true=(joined[(joined["runs"]>200) & (joined['wickets']>5)]).sort_values(by='runs' , ascending= False)
  return(true)

def all_rounder_impact():
    bat_stats = df_bat[df_bat['dismissal_type'] != "not out"].groupby("striker").agg(
    runs_scored=('runs_scored', 'sum'),
    dismissals=('dismissal_type', 'count'))
    
    bowl_stats = df_bowler.groupby("bowler").agg(
    wickets_taken=('wickets_taken', 'sum'),runs= ('runs_conceded', 'sum'))
   
    allrounder_stats = bat_stats.join(bowl_stats, how='inner')
    
    allrounder_stats['BatAV'] = allrounder_stats['runs_scored'] / allrounder_stats['dismissals']
    allrounder_stats["BOWLAV"]= allrounder_stats['runs'] / allrounder_stats['wickets_taken']
    allrounder_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    allrounder_stats.dropna(subset=['BatAV', 'BOWLAV'], inplace=True)   
    def classify(row):
        difference= row["BatAV"]- row['BOWLAV']
        if difference>=2:
            return("Batting All-rounder")
        elif difference<= -2:
            return("Bowling All-rounder")
        else:
            return("Genuine All-rounder")
    print("All-rounder classification for this IPL are:- ")
    allrounder_stats["Type"]= allrounder_stats.apply(classify,axis=1)
    print(allrounder_stats.sort_values('BOWLAV', ascending= False))
    
def impact_points():
    runs= df_bat.groupby("striker")['runs_scored'].sum().to_frame(name='total_runs')
    wickets= df_bowler.groupby("bowler")['wickets_taken'].sum().to_frame(name= 'total_wickets')
    stat= runs.join(wickets , how= 'inner')
    stat["mvp_score"]= (stat['total_runs']) +(stat['total_wickets']*25)
    print("Batters with the highest impact points are:- ")
    print(stat.sort_values(by= 'mvp_score', ascending= False))

def most_num():
    total_wickets= df_bowler.groupby("bowler")['wickets_taken'].sum().sort_values(ascending= False)
    print("The highest wicket-takers in IPL 2025 are:- ")
    print(total_wickets.head(5))

def most_runs_conceeded():
    total_runs= df_bowler.groupby("bowler")['runs_conceded'].sum().sort_values(ascending=False)
    print("The bowlers who gave the most runs are:- ")
    print(total_runs.head(5))

def most_boundaries_bowling():
    total_four= df_bowler.groupby("bowler")['fours_conceded'].sum()
    total_six= df_bowler.groupby("bowler")['sixes_conceded'].sum()
    total_boundary= (total_four) + (total_six)
    print("Bowlers who gave the most boundaries are:-")
    print((total_boundary).sort_values(ascending= False).head(5))

def boundaries_runs():
    total_four= df_bowler.groupby("bowler")['fours_conceded'].sum()
    total_six= df_bowler.groupby("bowler")['sixes_conceded'].sum()
    total_boundary= (total_four*4) + (total_six*6)
    print("Bowlers who gave most runs in boundaries are:- ")
    print((total_boundary).sort_values(ascending= False).head(5))

def lowest_economy():
    average_economy= df_bowler.groupby("bowler")['economy_rate'].mean().sort_values(ascending= True)
    print("The bowler with the lowest average economy rate are:-")
    print(average_economy.head(5))

def strike_rate_bowling():
    total_overs= df_bowler.groupby("bowler")["overs_bowled"].sum()
    total_balls= total_overs*6
    total_wickets= df_bowler.groupby("bowler")['wickets_taken'].sum()
    strike= (total_balls/total_wickets).replace([np.inf, -np.inf], 0)
    print("Bowlers with the highest strike rates are:- ")
    print((strike).sort_values(ascending= True).head(5))

def impact_in_bowling():
    bowling = df_bowler.set_index('match_id')
    matches = df_matches.set_index('match_id')
    impact = bowling.join(matches, lsuffix='_bow', rsuffix='_match')
    accumalate= impact[impact['player_team'] == impact['match_winner']]
    win_person= accumalate.groupby(['bowler', 'player_team'])['wickets_taken'].sum()
    team= accumalate.groupby('player_team')['wickets_taken'].sum()
    most = (win_person / win_person.index.get_level_values('player_team').map(team) * 100).sort_values(ascending= False)
    print("The bowlers who had the most impact in wins are:-")
    print(most)
    
def most_impactful_bowl(team):
    bowling = df_bowler.set_index('match_id')
    matches = df_matches.set_index('match_id')
    impact = bowling.join(matches, lsuffix='_bowl', rsuffix='_match')
    accumalate = impact[(impact['player_team'] == impact['match_winner']) & (impact['player_team'] == team)]
    win_person = accumalate.groupby(['bowler', 'player_team'])['wickets_taken'].sum()
    team_win = accumalate.groupby('player_team')['wickets_taken'].sum()
    most = (win_person / win_person.index.get_level_values('player_team').map(team_win) * 100)
    values = most.values.tolist()
    names = [str(name) for name in most.index]
    fig = plt.figure(figsize=(20,10))
    plt.pie(values, labels=names)
    plt.title("Bowlers with the most impact for " + team)
    plt.show()
    
def most_three_wickets():
    count_wickets= df_bowler.loc[df_bowler.wickets_taken>=3].groupby("bowler")['wickets_taken'].count().sort_values(ascending= False)
    print("The bowlers with the most 3 or more wicket hauls in an innings are:- ")
    print(count_wickets.head(5))

def most_runs_given():
    most= df_bowler.groupby("bowler")['runs_conceded'].max().sort_values(ascending=False)
    print("Bowlers who gave the most runs in an innings in this IPL are:- ")
    print(most.head(5))

def most_dot_balls():
    most_balls= df_bowler.groupby("bowler")["dots_bowled"].sum().sort_values(ascending= False)
    print("The bowlers who bowled the most dotballs in this IPL are:- ")
    print(most_balls.head(5))

def bowling_impact():
    stats= df_bowler.groupby("bowler")[['wickets_taken', 'dots_bowled','runs_conceded']].sum()
    stats["impact_score"]= ((stats['wickets_taken']*20)+ (stats['dots_bowled']*2)) - (stats['runs_conceded'])
    print("The bowlers with the highest bowling impact in this IPL are:- ")
    print(stats['impact_score'].sort_values(ascending= False).head(5))
    
# For predictions 
def highest_run_getters():
    y= df_bat.groupby('striker').runs_scored.sum()
    features= ['batting_position', 'strike_rate', 'fours', 'sixes','singles', 'doubles']
    X= df_bat.groupby('striker')[features].sum()
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    forest_model= RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    print("The highest run scorers in the next IPL are:- ")
    predict= forest_model.predict(val_X)
    result= pd.DataFrame({'striker': val_X.index, 'runs_scored' : predict})
    result= result.round().sort_values(by='runs_scored', ascending=False)
    print(result.head())
 
highest_run_getters()
    
def highest_wicket_takers():
     y= df_bowler.groupby('bowler').wickets_taken.sum()
     features= ["runs_conceded", 'dots_bowled', 'fours_conceded', 'sixes_conceded', 'economy_rate']
     X= df_bowler.groupby('bowler')[features].sum()
     train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
     forest= RandomForestRegressor(random_state=1)
     forest.fit(train_X,train_y)
     print("The highest wicket takers in next year's IPL are:-")
     predict= forest.predict(val_X)
     result= pd.DataFrame({'bowler': val_X.index, 'wickets_taken' : predict})
     result= result.round().sort_values(by='wickets_taken', ascending=False)
     print(result.head())
highest_wicket_takers()


def main_menu():
    print("      🏏 IPL 2025 ANALYTICS ENGINE v1.0      ")
    print("="*50)
    print("--- BATTING ANALYSIS ---")
    print("1.  Top Strikers (150+ SR)")
    print("2.  Top Accumulators (Runs)")
    print("3.  Dismissal Analysis (Search Player)")
    print("4.  Most Boundaries")
    print("5.  Most Runs in Boundaries")
    print("6.  50+ Conversion Rates")
    print("7.  Highest Scores (Single Innings)")
    print("8.  Batting Impact Score")
    print("9.  Effective Scoring Index")
    print("10. Impact in Wins (Team Share %)")
    print("11. Team Batting Distribution (Chart)")
    print(("27. Balls Per Dismisal"))
    
    print("\n--- ALL-ROUNDER ANALYSIS ---")
    print("12. Elite All-Rounders List")
    print("13. All-Rounder Classification (Bat/Bowl/Genuine)")
    print("14. MVP Leaderboard (Impact Points)")
    
    print("\n--- BOWLING ANALYSIS ---")
    print("15. Top Wicket Takers")
    print("16. Most Runs Conceded (Total)")
    print("17. Most Boundaries Conceded")
    print("18. Most Runs Conceded via Boundaries")
    print("19. Best Economy Rates")
    print("20. Best Bowling Strike Rates")
    print("21. Impact in Wins (Bowling Share %)")
    print("22. Team Bowling Distribution (Chart)")
    print("23. Most 3-Wicket Hauls")
    print("24. Most Expensive Spells (Innings)")
    print("25. Dot Ball Kings")
    print("26. Bowling Impact Score")
    
    print("\n--- MACHINE LEARNING PREDICTIONS ---")
    print("28. Predict Highest Run Scorers (ML)")
    print("29. Predict Highest Wicket Takers (ML)")
    
    print("\n0.  Exit")
    print("="*50)
    
    while True:
        choice = input("👉 Enter Option Number: ")

        if choice == '1': strike_rate()
        elif choice == '2': accumalators()
        elif choice == '3': 
            name = input("Enter player name (e.g. Kohli, Will Jacks): ")
            dismissals(name)
        elif choice == '4': most_boundaries()
        elif choice == '5': most_runs_boundaries()
        elif choice == '6': conversion_to_fifty()
        elif choice == '7': most_runs_scored()
        elif choice == '8': batting_impact()
        elif choice == '9': effective_scoring()
        elif choice == '10': impact_in_wins()
        elif choice == '11': 
            team = input("Enter Team Code (e.g. RCB, MI, CSK): ")
            most_impactful_bat(team)
        elif choice == "27": 
            print("Batsmen with the highest balls per dismisal are:- ")
            print(BPD())
        elif choice == '12': allrounder()
        elif choice == '13': all_rounder_impact()
        elif choice == '14': impact_points()
        elif choice == '15': most_num()
        elif choice == '16': most_runs_conceeded()
        elif choice == '17': most_boundaries_bowling()
        elif choice == '18': boundaries_runs()
        elif choice == '19': lowest_economy()
        elif choice == '20': strike_rate_bowling()
        elif choice == '21': impact_in_bowling()
        elif choice == '22': 
            team = input("Enter Team Code (e.g. RCB, MI, CSK): ")
            most_impactful_bowl(team)
        elif choice == '23': most_three_wickets()
        elif choice == '24': most_runs_given()
        elif choice == '25': most_dot_balls()
        elif choice == '26': bowling_impact()
        elif choice == '28': highest_run_getters()
        elif choice == '29': highest_wicket_takers()
        elif choice == '0':
            print("Exiting... Have a great day!")
            break
        else:
            print("❌ Invalid Selection. Please try again.")

main_menu()

