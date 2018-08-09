class Ball:
    """A class for keeping info about the balls"""
    def __init__(self, ball):
        self.x = round(ball[2],1)
        self.y = round(ball[3],1)
        self.radius = ball[4]
        self.color = '#ff8c00'  # Hardcoded orange
    def get_info(self):
        return self.x, self.y, self.radius
