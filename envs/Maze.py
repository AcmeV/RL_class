import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Maze(tk.Tk, object):

    def __init__(self, unit=40, height=8, width=8):

        super(Maze, self).__init__()

        self.unit = unit
        self.height = height
        self.width = width

        self.has_terminal_tag = True
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.step_counter = 0
        self.is_render = False
        self.title('maze')
        self.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.height * self.unit,
                                width=self.width * self.unit)

        # create grids
        for c in range(0, self.width * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.width * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([self.unit * 2, self.unit])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([self.unit, self.unit * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # hell
        hell3_center = origin + np.array([self.unit * 2, self.unit * 6])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        # hell
        hell4_center = origin + np.array([self.unit * 6, self.unit * 2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        # hell
        hell5_center = origin + np.array([self.unit * 4, self.unit * 4])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        # hell
        hell6_center = origin + np.array([self.unit * 4, self.unit * 1])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # hell
        hell7_center = origin + np.array([self.unit * 1, self.unit * 3])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')

        # hell
        hell8_center = origin + np.array([self.unit * 2, self.unit * 4])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')


        # create oval
        oval_center = origin + self.unit * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.step_counter = 0
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.unit:
                base_action[1] -= self.unit
        elif action == 1:  # down
            if s[1] < (self.height - 1) * self.unit:
                base_action[1] += self.unit
        elif action == 2:  # right
            if s[0] < (self.width - 1) * self.unit:
                base_action[0] += self.unit
        elif action == 3:  # left
            if s[0] > self.unit:
                base_action[0] -= self.unit


        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 5
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7), self.canvas.coords(self.hell8)]:
            reward = -5
            done = True
        elif s_ == s:
            reward = -1
            done = False
        else:
            reward = -1
            done = False

        self.step_counter += 1
        # time.sleep(0.1)
        return s_, reward, done, self.step_counter

    def close(self):
        pass

    def render(self):
        if self.is_render:
            time.sleep(0.1)
            self.update()
        else:
            self.update()