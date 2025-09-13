import pygame
import pickle
import numpy as np

# ================= Load Trained Q-values =================
with open("trained_q_values.pkl", "rb") as f:
    q_values = pickle.load(f)

# ================= Object Types and Actions =================
object_types = ["Fragile_Light", "Medium_Rigid", "Heavy_Rigid"]
actions = ["Grip Low", "Grip Medium", "Grip High"]
best_grips = [0, 1, 2]  # true optimal grips

# Colors for grip zones
grip_colors = [(0, 102, 204), (255, 204, 0), (204, 0, 0)]  # Blue, Yellow, Red

# ================= Pygame Setup =================
pygame.init()
screen_width, screen_height = 900, 650
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Robotic Gripper Game")
font = pygame.font.SysFont(None, 32)
feedback_font = pygame.font.SysFont(None, 64)
clock = pygame.time.Clock()

# ================= Load Images =================
robot_img = pygame.image.load("./img/robot.png")
robot_img = pygame.transform.scale(robot_img, (100, 100)) 

object_imgs = [
    pygame.transform.scale(pygame.image.load("./img/fragile_light.png"), (100, 100)),
    pygame.transform.scale(pygame.image.load("./img/medium_rigid.png"), (100, 100)),
    pygame.transform.scale(pygame.image.load("./img/heavy_rigid.png"), (100, 100))
]

# ================= Positions =================
grip_positions = [
    (screen_width // 4, 130),
    (screen_width // 2, 130),
    (3 * screen_width // 4, 130)
]

robot_start_pos = [screen_width // 2 - 50, screen_height // 2 - 50]  
robot_pos = robot_start_pos.copy()

button_width, button_height = 200, 60
button_y = screen_height - 120
object_button_positions = [
    (screen_width // 4 - button_width//2, button_y, button_width, button_height),
    (screen_width // 2 - button_width//2, button_y, button_width, button_height),
    (3*screen_width//4 - button_width//2, button_y, button_width, button_height)
]

object_positions = [
    (screen_width // 4 - 50, button_y - 120),
    (screen_width // 2 - 50, button_y - 120),
    (3*screen_width//4 - 50, button_y - 120)
]

# ================= Game State =================
selected_obj = None
robot_action = None
moving = False
target_pos = robot_start_pos.copy()
move_speed = 6
feedback_text = ""
feedback_color = (0,0,0)

# Grabbing animation
grabbing = False
grab_offset = 0
grab_direction = -1
grab_max_offset = 5     # smaller movement for subtle effect
grab_speed = 1.5        # slower grabbing motion

# ================= Main Loop =================
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]

    # --- Draw gradient background ---
    for y in range(screen_height):
        color = (
            240 - y//10, 240 - y//10, 255  # Soft gradient: light blue to white
        )
        pygame.draw.line(screen, color, (0, y), (screen_width, y))

    # --- Draw grip zones with rounded corners and shadow ---
    for i, (gx, gy) in enumerate(grip_positions):
        rect = pygame.Rect(gx-60, gy-60, 120, 120)
        shadow_rect = rect.copy()
        shadow_rect.x += 5
        shadow_rect.y += 5
        pygame.draw.rect(screen, (200, 200, 200), shadow_rect, border_radius=15)  # shadow
        pygame.draw.rect(screen, grip_colors[i], rect, border_radius=15)
        
        # Text centered under the box
        text = font.render(actions[i], True, (0,0,0))
        text_rect = text.get_rect(center=(rect.centerx, rect.bottom + 20))
        screen.blit(text, text_rect)

    # --- Draw object images above buttons ---
    for i, pos in enumerate(object_positions):
        screen.blit(object_imgs[i], pos)

    # --- Draw object selection buttons (interactive) ---
    for i, (x, y, w, h) in enumerate(object_button_positions):
        rect = pygame.Rect(x, y, w, h)
        # Hover and click effects
        if rect.collidepoint(mouse_pos):
            if mouse_pressed:
                color = (60, 110, 200)  # pressed
            else:
                color = (150, 200, 255)  # hover
        else:
            color = (100, 160, 250)  # normal
        pygame.draw.rect(screen, color, rect, border_radius=12)
        pygame.draw.rect(screen, (0,0,0), rect, 2, border_radius=12)
        text = font.render(object_types[i], True, (0,0,0))
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)

    # --- Draw robot ---
    display_robot_pos = [robot_pos[0], robot_pos[1] + grab_offset]
    screen.blit(robot_img, display_robot_pos)

    # --- Robot movement and feedback ---
    if selected_obj is not None:
        if robot_action is None:
            robot_action = np.argmax(q_values[selected_obj])
            target_pos = [grip_positions[robot_action][0]-50, grip_positions[robot_action][1]-50]  
            moving = True
            grabbing = False
            grab_offset = 0

        if moving:
            dx = target_pos[0] - robot_pos[0]
            dy = target_pos[1] - robot_pos[1]
            distance = np.hypot(dx, dy)
            if distance < move_speed:
                robot_pos = target_pos.copy()
                moving = False
                grabbing = True
                feedback_text = "Right Gripper!" if robot_action == best_grips[selected_obj] else "Try Again!"
                feedback_color = (0,255,0) if robot_action == best_grips[selected_obj] else (255,0,0)
            else:
                robot_pos[0] += move_speed * dx / distance
                robot_pos[1] += move_speed * dy / distance

        if grabbing:
            grab_offset += grab_direction * grab_speed
            if abs(grab_offset) >= grab_max_offset:
                grab_direction *= -1

        # Feedback text with shadow
        shadow_render = feedback_font.render(feedback_text, True, (50,50,50))
        shadow_x = screen_width // 2 - shadow_render.get_width() // 2 + 3
        shadow_y = screen_height // 2 - shadow_render.get_height() // 2 + 3
        screen.blit(shadow_render, (shadow_x, shadow_y))
        feedback_render = feedback_font.render(feedback_text, True, feedback_color)
        feedback_x = screen_width // 2 - feedback_render.get_width() // 2
        feedback_y = screen_height // 2 - feedback_render.get_height() // 2
        screen.blit(feedback_render, (feedback_x, feedback_y))

    pygame.display.flip()
    clock.tick(60)

    # --- Event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            for i, (bx, by, bw, bh) in enumerate(object_button_positions):
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_obj = i
                    robot_action = None
                    robot_pos = robot_start_pos.copy()
                    moving = False
                    grabbing = False
                    grab_offset = 0
                    feedback_text = ""

pygame.quit()
