import pygame
import random
import pickle
import os

# -------------------- Load trained policy --------------------
with open("policy.pkl", "rb") as f:
    policy = pickle.load(f)

# -------------------- Pygame setup --------------------
pygame.init()
WIDTH, HEIGHT = 1000, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blackjack - Agent vs Dealer")
font = pygame.font.SysFont('arial', 20)
big_font = pygame.font.SysFont('arial', 36)
clock = pygame.time.Clock()

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,200,0)
YELLOW = (255,255,0)
TABLE_GREEN = (0,120,0)

CARD_WIDTH, CARD_HEIGHT = 60, 90
LOG_WIDTH, LOG_HEIGHT = 350, 200

# -------------------- Card setup --------------------
SUITS = ['hearts','diamonds','clubs','spades']
RANKS = ['ace','2','3','4','5','6','7','8','9','10','jack','queen','king']
CARD_IMAGES = {}

# Load all card images
for suit in SUITS:
    for rank in RANKS:
        filename = f"cards/{rank}_of_{suit}.png"
        if os.path.exists(filename):
            image = pygame.image.load(filename)
            image = pygame.transform.scale(image, (CARD_WIDTH, CARD_HEIGHT))
            CARD_IMAGES[(rank, suit)] = image

# Map ranks to Blackjack values
def card_value(rank):
    if rank in ['jack','queen','king']:
        return 10
    elif rank == 'ace':
        return 1
    else:
        return int(rank)

def draw_card():
    suit = random.choice(SUITS)
    rank = random.choice(RANKS)
    return (rank, suit)

# -------------------- Player and Dealer classes --------------------
class Player:
    def __init__(self, cards, dealer_card):
        self.cards = cards
        self.dealer_card = dealer_card
        self.update_sum()

    def add_card(self, card):
        self.cards.append(card)
        self.update_sum()

    def update_sum(self):
        total = sum([card_value(r) for r,s in self.cards])
        ace_count = sum(1 for r,s in self.cards if r=='ace')
        self.usable_ace = False
        if ace_count > 0 and total + 10 <= 21:
            self.usable_ace = True
            total += 10
        self.current_sum = total

    def get_state(self):
        return (self.current_sum, self.usable_ace, card_value(self.dealer_card[0]))

class Dealer:
    def __init__(self, cards):
        self.cards = cards
        self.update_sum()

    def add_card(self, card):
        self.cards.append(card)
        self.update_sum()

    def update_sum(self):
        total = sum([card_value(r) for r,s in self.cards])
        ace_count = sum(1 for r,s in self.cards if r=='ace')
        self.usable_ace = False
        if ace_count > 0 and total + 10 <= 21:
            self.usable_ace = True
            total += 10
        self.current_sum = total

    def should_hit(self):
        return self.current_sum < 17

# -------------------- Game State --------------------
player = None
dealer = None
dealer_hole = None
revealed = False
action_log = []
game_started = False
round_over = False
wins = {"Player":0, "Dealer":0, "Draw":0}

# -------------------- Buttons --------------------
start_btn = pygame.Rect(50, HEIGHT-80, 115, 50)
new_btn = pygame.Rect(200, HEIGHT-80, 135, 50)

def draw_buttons():
    pygame.draw.rect(win, GREEN, start_btn)
    win.blit(font.render("START", True, BLACK), (start_btn.x+20,start_btn.y+15))
    pygame.draw.rect(win, YELLOW, new_btn)
    win.blit(font.render("NEW GAME", True, BLACK), (new_btn.x+10,new_btn.y+15))

# -------------------- Display Hands & Log --------------------
def display_hands():
    win.fill(TABLE_GREEN)

    # Dealer cards
    win.blit(font.render(f"Dealer: {dealer.current_sum}", True, WHITE), (50, 20))
    for i, card in enumerate(dealer.cards):
        row = i // 10
        col = i % 10
        x = 150 + col*(CARD_WIDTH+5)
        y = 20 + row*(CARD_HEIGHT+5)
        image = CARD_IMAGES.get(card)
        if image:
            pygame.draw.rect(win, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
            win.blit(image, (x,y))

    # Player cards
    win.blit(font.render(f"Player: {player.current_sum}", True, WHITE), (50, 150))
    for i, card in enumerate(player.cards):
        row = i // 10
        col = i % 10
        x = 150 + col*(CARD_WIDTH+5)
        y = 150 + row*(CARD_HEIGHT+5)
        image = CARD_IMAGES.get(card)
        if image:
            pygame.draw.rect(win, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
            win.blit(image, (x,y))

    # Action log
    log_x, log_y = 50, 300
    pygame.draw.rect(win, BLACK, (log_x-10, log_y-10, LOG_WIDTH, LOG_HEIGHT))
    for i, act in enumerate(action_log[-8:]):
        win.blit(font.render(act, True, WHITE), (log_x, log_y + i*20))

    # Scoreboard top-right
    board_width = 200
    board_height = 90
    board_x = WIDTH - board_width - 20
    board_y = 20
    pygame.draw.rect(win, BLACK, (board_x, board_y, board_width, board_height))
    pygame.draw.rect(win, WHITE, (board_x, board_y, board_width, board_height), 3)
    win.blit(font.render(f"Player Wins: {wins['Player']}", True, WHITE), (board_x+10, board_y+10))
    win.blit(font.render(f"Dealer Wins: {wins['Dealer']}", True, WHITE), (board_x+10, board_y+35))
    win.blit(font.render(f"Draws: {wins['Draw']}", True, WHITE), (board_x+10, board_y+60))

    draw_buttons()
    pygame.display.update()

# -------------------- Check for Blackjack --------------------
def check_for_blackjack():
    global round_over, revealed
    if player.current_sum == 21 and len(player.cards) == 2:
        action_log.append("Player has Blackjack!")
        revealed = True
        hole_rank, hole_suit = dealer_hole
        action_log.append(f"Dealer reveals hole: {hole_rank} of {hole_suit}")
        dealer.add_card(dealer_hole)
        display_hands()
        pygame.time.delay(800)
        if dealer.current_sum == 21:
            message = "Draw!"
            wins['Draw'] += 1
        else:
            message = "Player Wins!"
            wins['Player'] += 1
        show_end_message(message)
        round_over = True
        return True
    return False

# -------------------- New Game --------------------
def new_game():
    global player, dealer, dealer_hole, revealed, action_log, game_started, round_over
    dealer_up = draw_card()
    dealer_hole = draw_card()
    dealer = Dealer([dealer_up])
    player_cards = [draw_card(), draw_card()]
    player = Player(player_cards, dealer_up)
    action_log = ["Game Ready! Click Start!"]
    revealed = False
    game_started = False
    round_over = False
    display_hands()
    check_for_blackjack()

# -------------------- Show End Message --------------------
def show_end_message(message):
    box_width = 300
    box_height = 80
    box_x = WIDTH - box_width - 20
    box_y = HEIGHT - box_height - 20
    s = pygame.Surface((box_width, box_height))
    s.set_alpha(200)
    s.fill(BLACK)
    win.blit(s, (box_x, box_y))
    pygame.draw.rect(win, WHITE, (box_x, box_y, box_width, box_height),3)
    text_surface = big_font.render(message, True, RED)
    text_rect = text_surface.get_rect(center=(box_x+box_width//2, box_y+box_height//2))
    win.blit(text_surface, text_rect)
    pygame.display.update()
    pygame.time.delay(3000)

# -------------------- Agent Plays --------------------
def agent_play():
    global round_over
    while True:
        state = player.get_state()
        action = True if player.current_sum < 12 else policy.get(state, False)  # Force hit if sum < 12
        if action:
            card = draw_card()
            player.add_card(card)
            action_log.append(f"Agent hits: {card[0]} of {card[1]} >>> {player.current_sum}")
            display_hands()
            pygame.time.delay(800)
            if player.current_sum > 21:
                action_log.append("Agent busts!")
                display_hands()
                round_over = True
                return
        else:
            action_log.append("Agent sticks")
            display_hands()
            pygame.time.delay(800)
            break

# -------------------- Dealer Plays --------------------
def dealer_play():
    global round_over, revealed
    revealed = True
    hole_rank, hole_suit = dealer_hole
    action_log.append(f"Dealer reveals: {hole_rank} of {hole_suit}")
    dealer.add_card(dealer_hole)
    display_hands()
    pygame.time.delay(800)
    while dealer.should_hit():
        card = draw_card()
        dealer.add_card(card)
        action_log.append(f"Dealer hits: {card[0]} of {card[1]} -> {dealer.current_sum}")
        display_hands()
        pygame.time.delay(800)

# -------------------- Determine Winner --------------------
def determine_winner():
    global wins, round_over
    if player.current_sum > 21:
        message = "Dealer Wins!"
        wins['Dealer'] += 1
    elif dealer.current_sum > 21:
        message = "Player Wins!"
        wins['Player'] += 1
    elif player.current_sum > dealer.current_sum:
        message = "Player Wins!"
        wins['Player'] += 1
    elif dealer.current_sum > player.current_sum:
        message = "Dealer Wins!"
        wins['Dealer'] += 1
    else:
        message = "Draw!"
        wins['Draw'] += 1
    round_over = True
    display_hands()
    show_end_message(message)

# -------------------- Main Loop --------------------
new_game()
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_btn.collidepoint(event.pos) and not game_started and not round_over:
                game_started = True
                action_log.append("Agent turn starts")
                display_hands()
                agent_play()
                if not round_over:
                    dealer_play()
                    determine_winner()
            elif new_btn.collidepoint(event.pos):
                new_game()

pygame.quit()