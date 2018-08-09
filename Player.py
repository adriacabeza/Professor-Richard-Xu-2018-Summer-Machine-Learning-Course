from Team import Team


class Player:
    """A class for keeping info about the players"""
    def __init__(self, player):
        self.team = Team(player[0])
        self.id = player[1]
        self.x = round(player[2],1)
        self.y = round(player[3],1)
        self.color = self.team.color
    def get_info(self):
        return self.team.id, self.id, self.x, self.y
    # def show_pos(self):
    #     return self.id, self.x, self.y