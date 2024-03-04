import pygame
from logic import Logic

class GUI:

    def __init__(self):
        pygame.init()
        self.logic = Logic()
        self.window = None

    def run(self):
        DISPLAY_WIDTH = 800
        DISPLAY_HEIGHT = 800
        self.window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))

        while True:
            self.paint()

            for event in pygame.event.get():
                if(event.type == pygame.QUIT):
                    quit()


    def paint(self):
        # white background
        self.window.fill((255, 255, 255))
        self.draw_tiles()
        pygame.display.update()

    def draw_tiles(self):
        TILE_X_OFFSET = 100
        TILE_Y_OFFSET = 100
        TILE_MARGIN = 25

        for row_i, row in enumerate(self.logic.board):
            for col_i, val in enumerate(row):
                self.draw_tile(self.logic.board[row_i][col_i], TILE_X_OFFSET + row_i * (TILE_MARGIN + TILE_X_OFFSET), TILE_Y_OFFSET + col_i * (TILE_MARGIN + TILE_Y_OFFSET))

    def draw_tile(self, val, x, y, length=100):
        pygame.draw.rect(self.window, (0,0,0), [x, y, length, length], 3)