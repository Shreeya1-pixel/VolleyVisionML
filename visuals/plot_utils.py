import matplotlib.pyplot as plt
import pandas as pd

def plot_player_trends(df, player_id):
    player_data = df[df['player_id'] == player_id]
    player_data.set_index('matches_played')[['spikes', 'blocks', 'serves', 'digs']].plot()
    plt.title(f'Performance Trends for Player {player_id}')
    plt.xlabel("Matches")
    plt.ylabel("Count")
    plt.savefig('visuals/player_trend.png')
    plt.close()

