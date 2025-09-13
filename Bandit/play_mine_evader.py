import pygame
import pickle
import numpy as np
import random

# ================= Load trained Q-values =================
with open("trained_mine_values.pkl", "rb") as f:
    q_values = pickle.load(f)

# ================= Settings =================
actions = ["Up", "Down", "Left", "Right"]
agent_color = (0, 100, 255)
grid_color = (60, 60, 60)
button_color = (100, 160, 250)
button_hover = (150, 200, 255)
button_pressed = (60, 110, 200)
feedback_color_safe = (0, 200, 0)
bg_color = (230, 240, 255)
cell_size = 120
agent_radius = 25
move_speed = 10  # faster movement for smooth animation

pygame.init()
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Mine Evader Game")
font = pygame.font.SysFont(None, 32)
feedback_font = pygame.font.SysFont(None, 50)
clock = pygame.time.Clock()

center_x, center_y = screen_width // 2, screen_height // 2
agent_pos = [center_x, center_y]
target_pos = agent_pos.copy()
agent_moving = False
feedback_text = ""

# ================= Load images =================
mine_img = pygame.image.load("./img/mine.png").convert_alpha()  # supports transparency
mine_img = pygame.transform.scale(mine_img, (80, 80))  # bigger mine

# Buttons
button_width, button_height = 200, 50
new_episode_pos = (screen_width // 4 - button_width // 2, screen_height - 80)
run_agent_pos = (3 * screen_width // 4 - button_width // 2, screen_height - 80)

# Game state
current_state = [0, 0, 0, 0]
mines_pos = []

# ================= Functions =================
def state_to_index(state):
    return state[0]*8 + state[1]*4 + state[2]*2 + state[3]

def generate_random_state():
    n_mines = random.randint(1, 3)
    state = [0, 0, 0, 0]
    positions = random.sample([0, 1, 2, 3], n_mines)
    for pos in positions:
        state[pos] = 1
    return state

def state_to_positions(state):
    offsets = [(-cell_size, 0), (cell_size, 0), (0, -cell_size), (0, cell_size)]
    positions = []
    for i, val in enumerate(state):
        if val == 1:
            positions.append(offsets[i])
    return positions

# ================= Main loop =================
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]

    screen.fill(bg_color)

    # --- Draw grid lines ---
    for i in range(-1, 2):
        pygame.draw.line(screen, grid_color,
                         (center_x - cell_size*1.5, center_y + i*cell_size),
                         (center_x + cell_size*1.5, center_y + i*cell_size), 3)
        pygame.draw.line(screen, grid_color,
                         (center_x + i*cell_size, center_y - cell_size*1.5),
                         (center_x + i*cell_size, center_y + cell_size*1.5), 3)

    # --- Draw mines ---
    for dx, dy in mines_pos:
        screen.blit(mine_img, (center_x + dx - 40, center_y + dy - 40))  # center 80x80 image

    # --- Draw agent ---
    pygame.draw.circle(screen, agent_color, agent_pos, agent_radius)

    # --- Draw buttons ---
    buttons = [("New Episode", new_episode_pos), ("Run Agent", run_agent_pos)]
    for text, pos in buttons:
        button_rect = pygame.Rect(*pos, button_width, button_height)
        if button_rect.collidepoint(mouse_pos):
            color = button_pressed if mouse_pressed else button_hover
        else:
            color = button_color
        pygame.draw.rect(screen, color, button_rect, border_radius=12)
        pygame.draw.rect(screen, (0, 0, 0), button_rect, 2, border_radius=12)
        txt_surf = font.render(text, True, (0, 0, 0))
        txt_rect = txt_surf.get_rect(center=button_rect.center)
        screen.blit(txt_surf, txt_rect)

    # --- Feedback on top ---
    if feedback_text:
        feedback_render = feedback_font.render(feedback_text, True, feedback_color_safe)
        feedback_x = screen_width // 2 - feedback_render.get_width() // 2
        feedback_y = 20
        screen.blit(feedback_render, (feedback_x, feedback_y))

    # --- Move agent smoothly with easing ---
    if agent_moving:
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        dist = np.hypot(dx, dy)
        if dist < move_speed:
            agent_pos = target_pos.copy()
            agent_moving = False
            feedback_text = "Safe Move!"
        else:
            # Smooth movement with easing
            agent_pos[0] += move_speed * dx / dist
            agent_pos[1] += move_speed * dy / dist

    pygame.display.flip()
    clock.tick(60)

    # --- Event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            new_rect = pygame.Rect(*new_episode_pos, button_width, button_height)
            run_rect = pygame.Rect(*run_agent_pos, button_width, button_height)
            if new_rect.collidepoint((x, y)):
                current_state = generate_random_state()
                mines_pos = state_to_positions(current_state)
                agent_pos = [center_x, center_y]
                agent_moving = False
                feedback_text = ""
            elif run_rect.collidepoint((x, y)):
                state_idx = state_to_index(current_state)
                action = np.argmax(q_values[state_idx])
                offsets = [(-cell_size, 0), (cell_size, 0), (0, -cell_size), (0, cell_size)]
                target_pos = [center_x + offsets[action][0], center_y + offsets[action][1]]
                agent_moving = True

pygame.quit()
