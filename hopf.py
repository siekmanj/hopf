import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Hopfield:
  def __init__(self, input_dim):
    self.w = np.zeros((input_dim, input_dim))
    self.input_dim = input_dim
    self.theta = 0
    self.state = None

  def learn(self, pattern):
    self.w += np.outer(pattern, pattern) - np.identity(len(pattern))

  def energy(self, pattern):
    s = 0
    for i in range(self.input_dim):
      for j in range(self.input_dim):
        s += pattern[i] * pattern[j] * self.w[i,j]
    s /= -2

    for i in range(self.input_dim):
      s += self.theta * pattern[i]

    return s

  def step(self, a=None):
    if a is not None:
        self.state = a

    i = np.random.choice(range(self.input_dim))

    s = 0
    for j in range(self.input_dim):
      if i != j:
        s += self.w[i,j] * self.state[j] - self.theta
      
    self.state[i] = int(np.sign(s))
    return self.state, self.energy(self.state)

if __name__ == '__main__':
  x1 = [1, -1, 1, -1, 1, -1, 1, -1]
  x2 = [1, 1, 1, 1, -1, -1, -1, -1]
  x3 = [1, 1, 1, -1, 1, -1, 1, -1]
  x4 = [-1, -1, 1, 1, -1, -1, 1, 1]

  dim = len(x1)
  m = Hopfield(dim)
  m.learn(x1)
  m.learn(x2)

  m.step(x4)
  while True:
    a, energy = m.step()
    print(energy, a)
    input()

