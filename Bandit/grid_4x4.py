"""4x4 GridWorld with Policy Iteration and Pygame Visualization
    - A simple 4x4 GridWorld environment
    - Agent learns the optimal policy using policy iteration and solving linear Bellman equations
    - The environment:
        * Goal state
        * Pit states
        * Step Rewards for non-terminal states
    - The agent can move in four directions: up, down, left, right
    - If the agent attempts to move out of bounds, it stays in the same position
    - The policy is stochastic, allowing multiple optimal actions in a state
    - The optimal policy and state values are visualized using Matplotlib, and the agent's
      movement is animated in a Pygame window.
"""
import numpy as np
import pygame
import matplotlib.pyplot as plt
import sys
import copy

class GridWorld:
    """4x4 GridWorld Environment with Policy Iteration
    States:
        - 4x4 grid positions (row, col)
        Actions:
        - UP, DOWN, LEFT, RIGHT
        Rewards:
        - Step reward: -1.0
        - Goal state: +10.0
        - Pit states: -10.0
        Terminal States:
        - Goal state and pit states
    """
    def __init__(self):
        self.ROWS, self.COLS = 4, 4
        self.GAMMA = 0.9
        self.STEP_REWARD = -1.0
        self.GOAL = (3, 3)
        self.PITS = [(0, 2), (2, 0), (2, 2)]
        self.TERMINALS = {self.GOAL: +10.0}
        for p in self.PITS:
            self.TERMINALS[p] = -10.0
        self.ACTIONS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        self.policy = None
        self.V = None
        self.policy_history = []
        self.value_history = []

    def in_bounds(self, r, c):
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def next_state_from(self, r, c, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc):
            return (r, c)
        return (nr, nc)

    def run_policy_iteration(self):
        self.policy = np.full((self.ROWS, self.COLS), None, dtype=object)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r, c) not in self.TERMINALS:
                    self.policy[r, c] = ['RIGHT']

        stable = False
        while not stable:
            # Policy evaluation
            nonterm_states = [(r, c) for r in range(self.ROWS) for c in range(self.COLS) if (r, c) not in self.TERMINALS]
            idx_map = {s: i for i, s in enumerate(nonterm_states)}
            n = len(nonterm_states)

            A = np.zeros((n, n))
            b = np.zeros(n)
            for s, i in idx_map.items():
                r, c = s
                A[i, i] = 1.0
                a = self.policy[r, c][0]  # Use first action for evaluation
                ns_r, ns_c = self.next_state_from(r, c, a)
                if (ns_r, ns_c) in self.TERMINALS:
                    b[i] = self.TERMINALS[(ns_r, ns_c)]
                else:
                    j = idx_map[(ns_r, ns_c)]
                    A[i, j] -= self.GAMMA
                    b[i] = self.STEP_REWARD

            v_nonterm = np.linalg.solve(A, b)
            self.V = np.zeros((self.ROWS, self.COLS))
            for s, i in idx_map.items():
                self.V[s] = v_nonterm[i]
            for t, val in self.TERMINALS.items():
                self.V[t] = val

            # Store policy and values before improvement
            self.policy_history.append(copy.deepcopy(self.policy))
            self.value_history.append(np.copy(self.V))

            # Policy improvement
            policy_stable = True
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    if (r, c) in self.TERMINALS:
                        continue
                    best_val = -1e9
                    best_actions = []
                    for a in self.ACTIONS:
                        ns_r, ns_c = self.next_state_from(r, c, a)
                        immediate = self.TERMINALS.get((ns_r, ns_c), self.STEP_REWARD)
                        val = immediate + self.GAMMA * self.V[ns_r, ns_c]
                        if val > best_val:
                            best_val = val
                            best_actions = [a]
                        elif val == best_val:
                            best_actions.append(a)
                    if set(best_actions) != set(self.policy[r, c]):
                        policy_stable = False
                        self.policy[r, c] = best_actions
            stable = policy_stable

    def _plot_policy_on_ax(self, V, policy, ax, single_action=False):
        ax.set_xlim(0, self.COLS)
        ax.set_ylim(0, self.ROWS)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), self.COLS, self.ROWS, color=(0.1, 0.1, 0.1)))

        # Draw goal and pits
        for r, c in [self.GOAL]:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=(0.3, 0.8, 0.3, 0.7)))
        for r, c in self.PITS:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=(0.8, 0.3, 0.3, 0.7)))

        # Draw grid lines
        for r in range(self.ROWS + 1):
            ax.plot([0, self.COLS], [r, r], color='white', linewidth=2)
        for c in range(self.COLS + 1):
            ax.plot([c, c], [0, self.ROWS], color='white', linewidth=2)

        # Draw state values
        for r in range(self.ROWS):
            for c in range(self.COLS):
                ax.text(c + 0.5, r + 0.5, f"{V[r, c]:.1f}", ha='center', va='center', fontsize=14, color='cyan')

        # Draw arrows (single action for intermediate, all for final)
        arrow_size = 0.2
        corner_offset = 0.15
        for r in range(self.ROWS):
            for c in range(self.COLS):
                acts = policy[r, c]
                if acts is None:
                    continue
                actions_to_plot = [acts[0]] if single_action and len(acts) > 0 else acts
                for a in actions_to_plot:
                    dr, dc = self.ACTIONS[a]
                    start_x = c + 0.5 + (dc * arrow_size * 0.5)
                    start_y = r + 0.5 + (dr * arrow_size * 0.5)
                    start_x += corner_offset if dc >= 0 else -corner_offset
                    start_y += -corner_offset if dr <= 0 else corner_offset
                    ax.arrow(start_x, start_y, dc * arrow_size, dr * arrow_size,
                             head_width=0.08, head_length=0.08,
                             fc='yellow', ec='yellow', length_includes_head=True)

    def _plot_rewards_on_ax(self, ax):
        ax.set_xlim(0, self.COLS)
        ax.set_ylim(0, self.ROWS)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), self.COLS, self.ROWS, color=(0.1, 0.1, 0.1)))

        # Draw goal and pits
        for r, c in [self.GOAL]:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=(0.3, 0.8, 0.3, 0.7)))
        for r, c in self.PITS:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=(0.8, 0.3, 0.3, 0.7)))

        # Draw grid lines
        for r in range(self.ROWS + 1):
            ax.plot([0, self.COLS], [r, r], color='white', linewidth=2)
        for c in range(self.COLS + 1):
            ax.plot([c, c], [0, self.ROWS], color='white', linewidth=2)

        # Draw arrows and rewards for all actions in non-terminal states
        arrow_size = 0.15
        text_offset = 0.05
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r, c) in self.TERMINALS:
                    continue
                for a in self.ACTIONS:
                    ns_r, ns_c = self.next_state_from(r, c, a)
                    reward = self.TERMINALS.get((ns_r, ns_c), self.STEP_REWARD)
                    dr, dc = self.ACTIONS[a]
                    start_x = c + 0.5
                    start_y = r + 0.5
                    end_x = start_x + dc * arrow_size
                    end_y = start_y + dr * arrow_size
                    ax.arrow(start_x, start_y, dc * arrow_size, dr * arrow_size,
                             head_width=0.05, head_length=0.05,
                             fc='orange', ec='orange', length_includes_head=True)
                    # Place reward text at or near the arrowhead
                    text_x = end_x
                    text_y = end_y
                    if a == 'RIGHT':
                        text_x = min(end_x + text_offset, c + 0.95)
                        ha, va = 'left', 'center'
                    elif a == 'LEFT':
                        text_x = max(end_x - text_offset, c + 0.05)
                        ha, va = 'right', 'center'
                    elif a == 'DOWN':
                        text_y = min(end_y, r + 0.95)
                        ha, va = 'center', 'top'
                    elif a == 'UP':
                        text_y = max(end_y, r + 0.05)
                        ha, va = 'center', 'bottom'
                    ax.text(text_x, text_y, f"{reward:.1f}", ha=ha, va=va, fontsize=8, color='white')

    def plot_final_policy(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        self._plot_rewards_on_ax(axs[0])
        axs[0].set_title("State-Action Immediate Rewards", fontsize=16, color='black')
        
        self._plot_policy_on_ax(self.V, self.policy, axs[1], single_action=False)
        axs[1].set_title("Optimal Policy & State Values", fontsize=16, color='black')
        
        plt.tight_layout()
        plt.show(block=False)

    def plot_policy_updates(self):
        n = len(self.policy_history)
        if n == 0:
            return
        fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
        axs = axs if n > 1 else [axs]
        for i in range(n):
            self._plot_policy_on_ax(self.value_history[i], self.policy_history[i], axs[i], single_action=False)
            title = "Initial Policy" if i == 0 else f"Policy After {i} Update{'s' if i > 1 else ''}"
            axs[i].set_title(title, fontsize=16, color='black')
        plt.tight_layout()
        plt.show(block=False)

    def run_simulation(self):
        pygame.init()
        TITLE_H = 64
        WIDTH, HEIGHT = 700, 700
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("4x4 GridWorld - Stochastic Policy")

        BG = (20, 20, 40)
        TITLE_BG = (30, 30, 30)
        TITLE_FG = (255, 255, 255)
        CELL_BG = (50, 50, 80)
        GRID_LINE = (0, 0, 0)
        GOAL_COLOR = (80, 200, 120)
        PIT_COLOR = (220, 80, 80)
        AGENT_COLOR = (40, 110, 230)
        TEXT_COLOR = (255, 255, 255)
        BUTTON_COLOR = (50, 150, 200)
        BUTTON_TEXT = (255, 255, 255)
        SLOT_COLOR = (100, 100, 120)
        SLOT_HOVER_COLOR = (120, 120, 140)

        title_font = pygame.font.SysFont(None, 34)
        val_font = pygame.font.SysFont(None, 20)
        button_font = pygame.font.SysFont(None, 28)
        prompt_font = pygame.font.SysFont(None, 24)

        agent_pos = None
        running = True
        clock = pygame.time.Clock()
        STEP_DELAY_MS = 500
        moving = False
        last_move = pygame.time.get_ticks()
        button_rect = pygame.Rect(WIDTH // 2 - 60, HEIGHT - 45, 120, 40)
        hovered_slot = None

        def get_cell_and_slot_rects():
            w, h = screen.get_size()
            cell_w = w / self.COLS
            cell_h = (h - TITLE_H - 80) / self.ROWS
            cell_rects = {}
            slot_rects = {}
            
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    x = c * cell_w
                    y = TITLE_H + r * cell_h + 30
                    cell_rects[(r, c)] = pygame.Rect(x, y, cell_w, cell_h)
                    
                    # Add slot circles for non-terminal states
                    if (r, c) not in self.TERMINALS:
                        slot_center = (x + cell_w / 2, y + cell_h / 2)  # Center of cell
                        slot_radius = min(cell_w, cell_h) * 0.08
                        slot_rects[(r, c)] = (slot_center, slot_radius)
            
            return cell_rects, slot_rects

        def draw_grid():
            nonlocal hovered_slot
            screen.fill(BG)
            w, h = screen.get_size()
            cell_w = w / self.COLS
            cell_h = (h - TITLE_H - 80) / self.ROWS

            # title
            pygame.draw.rect(screen, TITLE_BG, (0, 0, w, TITLE_H))
            title = title_font.render("4x4 GridWorld", True, TITLE_FG)
            screen.blit(title, (12, TITLE_H // 2 - title.get_height() // 2))

            # prompt
            prompt = prompt_font.render("Click a circle slot to plce the agent, then click Start", True, (200, 200, 50))
            screen.blit(prompt, (12, TITLE_H + 5))

            # Check for hovered slot
            mx, my = pygame.mouse.get_pos()
            hovered_slot = None
            _, slot_rects = get_cell_and_slot_rects()
            
            for (r, c), (center, radius) in slot_rects.items():
                dx, dy = mx - center[0], my - center[1]
                if dx*dx + dy*dy <= radius*radius:
                    hovered_slot = (r, c)
                    break

            # draw cells
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    x = c * cell_w
                    y = TITLE_H + r * cell_h + 30
                    rect = pygame.Rect(x, y, cell_w, cell_h)
                    if (r, c) == self.GOAL:
                        col = GOAL_COLOR
                    elif (r, c) in self.PITS:
                        col = PIT_COLOR
                    else:
                        col = CELL_BG
                    pygame.draw.rect(screen, col, rect)
                    pygame.draw.rect(screen, GRID_LINE, rect, 2)

                    # Draw state value in top-left corner
                    val_text = f"{self.V[r, c]:.1f}"
                    val_surf = val_font.render(val_text, True, TEXT_COLOR)
                    val_rect = (x + 5, y + 5)
                    screen.blit(val_surf, val_rect)
                    
                    # Draw slot circles for non-terminal states
                    if (r, c) not in self.TERMINALS:
                        slot_center = (x + cell_w / 2, y + cell_h / 2)  # Center of cell
                        slot_radius = int(min(cell_w, cell_h) * 0.08)
                        slot_color = SLOT_HOVER_COLOR if hovered_slot == (r, c) else SLOT_COLOR
                        pygame.draw.circle(screen, slot_color, (int(slot_center[0]), int(slot_center[1])), slot_radius)
                        pygame.draw.circle(screen, GRID_LINE, (int(slot_center[0]), int(slot_center[1])), slot_radius, 2)

            # start button
            pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
            btn_text = button_font.render("Start", True, BUTTON_TEXT)
            btn_rect = btn_text.get_rect(center=button_rect.center)
            screen.blit(btn_text, btn_rect)

        def move_agent(pos):
            r, c = pos
            if pos in self.TERMINALS:
                return pos
            acts = self.policy[r, c]
            a = np.random.choice(acts)
            ns_r, ns_c = self.next_state_from(r, c, a)
            return (ns_r, ns_c)

        draw_grid()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    button_rect = pygame.Rect(event.w // 2 - 60, event.h - 45, 120, 40)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    
                    # Check if clicked on start button
                    if button_rect.collidepoint(mx, my) and agent_pos is not None:
                        moving = True
                    else:
                        # Check if clicked on a slot circle
                        _, slot_rects = get_cell_and_slot_rects()
                        for (r, c), (center, radius) in slot_rects.items():
                            dx, dy = mx - center[0], my - center[1]
                            if dx*dx + dy*dy <= radius*radius:
                                agent_pos = (r, c)
                                break

            now = pygame.time.get_ticks()
            if moving and agent_pos is not None and now - last_move >= 500:
                last_move = now
                agent_pos = move_agent(agent_pos)
                if agent_pos in self.TERMINALS:
                    moving = False

            draw_grid()
            if agent_pos is not None:
                w, h = screen.get_size()
                cell_w = w / self.COLS
                cell_h = (h - TITLE_H - 80) / self.ROWS
                r, c = agent_pos
                x = c * cell_w + cell_w / 2
                y = TITLE_H + r * cell_h + 30 + cell_h / 2
                radius = int(min(cell_w, cell_h) * 0.12)
                pygame.draw.circle(screen, AGENT_COLOR, (int(x), int(y)), radius)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    gw = GridWorld()
    gw.run_policy_iteration()
    gw.plot_policy_updates()
    gw.plot_final_policy()
    gw.run_simulation()