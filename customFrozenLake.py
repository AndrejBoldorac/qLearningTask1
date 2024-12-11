import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class CustomGridEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, grid_size=5, num_obstacles=5, render_mode=None):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(grid_size * grid_size)  # Grid states

        self.agent_pos = [0, 0]  # Start at top-left corner
        self.goal_pos = [grid_size - 1, grid_size - 1]  # Goal at bottom-right corner
        self.obstacles = self._generate_obstacles()

        self.window = None
        self.clock = None
        self.cell_size = 50  # Size of each grid cell
        self.render_mode = render_mode

    def reset(self):
        self.agent_pos = [0, 0]
        return self._get_state()

    def step(self, action):
        # Map actions to movements
        move_map = {
            0: [-1, 0],  # Up
            1: [1, 0],  # Down
            2: [0, -1],  # Left
            3: [0, 1]  # Right
        }

        # Calculate new position
        new_pos = [
            self.agent_pos[0] + move_map[action][0],
            self.agent_pos[1] + move_map[action][1]
        ]

        # Check boundaries
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.agent_pos = new_pos

        # Check for terminal states
        done = False
        reward = -1  # Penalty for each step
        if self.agent_pos == self.goal_pos:
            reward = 1000  # Reward for reaching the goal
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -100  # Penalty for hitting an obstacle
            done = True

        return self._get_state(), reward, done, {}

    def render(self, agent=None):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)

        # Clear the screen
        self.window.fill((255, 255, 255))

        # Draw the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

                # Calculate the average Q-value for this state, if the agent is provided
                if agent:
                    state_index = x * self.grid_size + y
                    if state_index < agent.q_table.shape[0]:
                        q_values = agent.q_table[state_index]
                        avg_q_value = np.mean(q_values)

                        # Render the Q-value text
                        text_surface = self.font.render(f"{avg_q_value:.2f}", True, (0, 0, 0))
                        self.window.blit(
                            text_surface,
                            (y * self.cell_size + 5, x * self.cell_size + 5),  # Offset for padding
                        )

        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.window, (255, 0, 0), obs_rect)

        # Draw goal
        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size,
            self.goal_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.window, (0, 255, 0), goal_rect)

        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size,
            self.agent_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.window, (0, 0, 255), agent_rect)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    def _get_state(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def _generate_obstacles(self):
        # All possible moves to check adjacent cells (up, down, left, right)
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Calculate valid neighbors of the goal
        valid_neighbors = []
        for offset in neighbor_offsets:
            neighbor = [self.goal_pos[0] + offset[0], self.goal_pos[1] + offset[1]]
            if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                valid_neighbors.append(neighbor)

        # Generate obstacles excluding goal and its neighbors
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            if pos not in obstacles and pos != self.agent_pos and pos != self.goal_pos:
                if pos not in valid_neighbors:  # Exclude immediate neighbors of the goal
                    obstacles.append(pos)

        # Ensure at least one valid neighbor of the goal remains free
        # Remove one random obstacle if necessary
        if all(neighbor not in obstacles for neighbor in valid_neighbors):
            # Randomly pick one valid neighbor and make sure it's open
            free_neighbor = valid_neighbors[np.random.randint(len(valid_neighbors))]
            if free_neighbor in obstacles:
                obstacles.remove(free_neighbor)

        return obstacles


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.001):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes=1000, max_steps=200):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                env.render(self)
                total_reward += reward
                state = next_state
                if done:
                    break
            self.decay_epsilon()
            print(f"Episode {episode + 1}: Total Reward = {total_reward} Epsilon: {self.epsilon}")


env = CustomGridEnvironment(grid_size=20, num_obstacles=50)
agent = QLearningAgent(env)

agent.train(episodes=5000, max_steps=500)
