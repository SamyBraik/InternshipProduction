import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the time-evolving probability density function p_t(x)
def p_t(x, t):
    mean = t  # mean shifts from 0 to 1
    std = 0.2 + 0.2 * t  # standard deviation increases from 0.2 to 0.5
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# Prepare the plot
x = np.linspace(-1, 2, 400)
times = np.linspace(0, 1, 20)
fig, ax = plt.subplots(figsize=(8, 4))
lines, = ax.plot([], [], lw=2)
ax.set_xlim(-1, 2)
ax.set_ylim(0, 2.5)
ax.set_title("Probability Path $p_t(x)$ in $\mathbb{R}^1$")
ax.set_xlabel("x")
ax.set_ylabel("Density")

# Animation function
def animate(i):
    t = times[i]
    y = p_t(x, t)
    lines.set_data(x, y)
    ax.set_title(f"Probability Path $p_t(x)$ at $t = {t:.2f}$")
    return lines,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(times), interval=300, blit=True)
plt.show()
