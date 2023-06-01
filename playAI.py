# Written by Anmol Gulati

from pong import Game
import pygame
import neat
import os
import time
import pickle
import sys


class playAI:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.right_paddle = self.game.right_paddle
        self.left_paddle = self.game.left_paddle
        self.show_start_screen()  # COMMENT THIS OUT when running run_neat()

    # This function creates a simple design for the starting screen
    def show_start_screen(self):
        font_title = pygame.font.Font(None, 48)
        font_info = pygame.font.Font(None, 24)

        welcome_text = font_title.render("Welcome to Pong AI", True, (255, 255, 255))
        start_text = font_info.render("Click Space to start", True, (255, 255, 255))
        how_to_play_text = font_info.render("How to Play: W to move up, S to move down", True, (255, 255, 255))

        text_rect1 = welcome_text.get_rect(
            center=(self.game.window.get_width() // 2, self.game.window.get_height() // 2 - 50))
        text_rect2 = start_text.get_rect(
            center=(self.game.window.get_width() // 2, self.game.window.get_height() // 2 + 10))
        text_rect3 = how_to_play_text.get_rect(
            center=(self.game.window.get_width() // 2, self.game.window.get_height() // 2 + 50))

        background_color = (0, 0, 0)  # Set background color
        text_color = (255, 255, 255)  # Set text color
        highlight_color = (0, 255, 0)  # Set highlight color

        while True:
            self.game.window.fill(background_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return

            animation_color = text_color if pygame.time.get_ticks() % 1000 < 500 else highlight_color
            animated_welcome_text = font_title.render("Welcome to Pong AI", True, animation_color)
            animated_start_text = font_info.render("Click Space to start", True, animation_color)
            animated_how_to_play_text = font_info.render("How to Play: W to move up, S to move down", True,
                                                         animation_color)

            self.game.window.blit(animated_welcome_text, text_rect1)
            self.game.window.blit(animated_start_text, text_rect2)
            self.game.window.blit(animated_how_to_play_text, text_rect3)
            pygame.display.flip()

    # this function tests the AI against a human player by passing a NEAT neural network
    def try_ai(self, net):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            game_stats = self.game.loop()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            result = net.activate((self.right_paddle.y, abs(
                self.right_paddle.x - self.ball.x), self.ball.y))
            res = result.index(max(result))

            if res == 1:  # AI moves up
                self.game.move_paddle(left=False, up=True)
            elif res == 2:  # AI moves down
                self.game.move_paddle(left=False, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)  # moves left paddle
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)  # moves right paddle

            self.game.draw(draw_score=True)  # shows score not hits
            pygame.display.update()

    # this function trains the AI by passing two NEAT neural networks and NEAT config object
    # both AI's play against each other to determine their fitness
    def develop_ai(self, genome_1, genome_2, config, draw=False):
        run = True
        start_time = time.time()

        network1 = neat.nn.FeedForwardNetwork.create(genome_1, config)
        network2 = neat.nn.FeedForwardNetwork.create(genome_2, config)
        self.genome_1 = genome_1
        self.genome_2 = genome_2
        maxHits = 50

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_stats = self.game.loop()
            self.move_ai_paddles(network1, network2)
            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            if game_stats.left_score == 1 or game_stats.right_score == 1 or game_stats.left_hits >= maxHits:
                self.calculate_fitness(game_stats, duration)
                break

        return False

    # this function determines if we should move the left/right paddle based on two neural networks that control them
    def move_ai_paddles(self, network1, network2):
        players = [(self.genome_1, network1, self.left_paddle, True),
                   (self.genome_2, network2, self.right_paddle, False)]
        for (genome, net, paddle, left) in players:
            output = net.activate(
                (paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:  # Don't move
                genome.fitness -= 0.01  # we want to discourage this
            elif decision == 1:  # Move up
                valid = self.game.move_paddle(left=left, up=True)
            else:  # Move down
                valid = self.game.move_paddle(left=left, up=False)

            if not valid:  # If the movement makes the paddle go off the screen punish the AI
                genome.fitness -= 1

    def calculate_fitness(self, game_stats, duration):
        self.genome_1.fitness += game_stats.left_hits + duration
        self.genome_2.fitness += game_stats.right_hits + duration


# this function runs each genome against each other one time to determine the fitness
def evaluate_genome(genomes, config):
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    for i, (genome_id1, genome_1) in enumerate(genomes):
        print(round(i / len(genomes) * 100), end=" ")
        genome_1.fitness = 0
        for genome_id2, genome_2 in genomes[min(i + 1, len(genomes) - 1):]:
            genome_2.fitness = 0 if genome_2.fitness is None else genome_2.fitness
            pong = playAI(win, width, height)

            force_quit = pong.develop_ai(genome_1, genome_2, config, draw=True)
            if force_quit:
                quit()


# this function runs NEAT alg using a restored checkpoint, evolves population for a specified no. of generations,
# saves the best genome to a file for later use
def run_neat_algorithm(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7') # remove comment and change 7 to whichever saved
    # checkpoint you want to run from
    p = neat.Population(config)  # comment this if you want to run from saved checkpoint
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(evaluate_genome, 50)  # change 50 to 1 if you want to test next generation's performance right away
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


# this function loads the best neural network, creates a game window, and tests performance
def test_best(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    best_network = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = playAI(win, width, height)
    pong.try_ai(best_network)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # run_neat_algorithm(config)
    test_best(config)
