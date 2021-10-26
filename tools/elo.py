
import numpy as np
import pickle

class elo_league:
    def __init__(self, load_path = "model/saved_models/league_data"):
        self.league_path = load_path
        try:
            save_file = open(self.league_path + ".pkl", "rb")
            self.player_list = pickle.load(save_file)
            self.league_size = len(list(self.player_list.keys()))-1
            print("Old league loaded")

        except:
            print("No league found, making new leauge")
            self.player_list = {}  # Dictionary holding elo of all players
            self.league_size = 0
            self.add_player("model" + str(self.league_size), 0)
            self.league_size += 1

    def calculate_elo(self, known_player, new_player, wining_prob):
        # known_player: The name of the player with known elo
        # new_player: Name of the new player to add with calculated elo
        # wining_prob: The calculated wining probability of new_player
        player_a_elo = self.player_list[known_player]
        elo = player_a_elo - 400*np.log10(1/wining_prob-1)
        self.player_list[new_player] = elo
        return elo

    def common_duel_elo(self, wining_prob):
        # Get the elo of the newest model and add a new model
        known_player = self.get_latest_player()
        new_name = "model" + str(self.league_size)
        new_elo = self.calculate_elo(known_player, new_name, wining_prob)
        self.league_size += 1
        return new_elo, self.league_size

    def add_player(self, known_player, elo):
        # For adding players with known elo
        self.player_list[known_player] = elo

    def save_league(self):
        save_file = open(self.league_path + ".pkl", "wb+")
        pickle.dump(self.player_list, save_file)
        save_file.close()

    def get_latest_player(self):
        return list(self.player_list.keys())[-1]




