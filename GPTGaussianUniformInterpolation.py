import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Gaussian and uniform components
def gaussian(x, mean=0.0, std=0.2):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def uniform(x, a=0.0, b=1.0):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0.0)

# Interpolated density p_t(x)
def p_t(x, t):
    g = gaussian(x)
    u = uniform(x)
    return (1 - t) * g + t * u

# Prepare the plot
x = np.linspace(-1, 2, 400)
times = np.linspace(0, 1, 20)
fig, ax = plt.subplots(figsize=(8, 4))
lines, = ax.plot([], [], lw=2)
ax.set_xlim(-1, 2)
ax.set_ylim(0, 2.5)
ax.set_title("Interpolated Density $p_t(x)$ from Gaussian to Uniform")
ax.set_xlabel("x")
ax.set_ylabel("$p_t(x)$")

# Animation function
def animate(i):
    t = times[i]
    y = p_t(x, t)
    lines.set_data(x, y)
    ax.set_title(f"Interpolated Density $p_t(x)$ at each time t")
    return lines,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(times), interval=300, blit=True)
plt.show()

# Save the animation as a GIF
# output_path = "/home/admin-sbraik/Documents/InternshipProduction/FlowMatching/interpolated_density.gif"
# ani.save(output_path, writer='imagemagick', fps=10)

# Extract 6 frames at regular intervals
num_frames = 6
selected_times = np.linspace(0, 1, num_frames)
output_frames_dir = "/home/admin-sbraik/Documents/InternshipProduction/FlowMatching/frames"

# Create directory if it doesn't exist
os.makedirs(output_frames_dir, exist_ok=True)

# Save the selected frames
for idx, t in enumerate(selected_times):
    plt.figure(figsize=(8, 4))
    y = p_t(x, t)
    plt.plot(x, y, lw=2)
    plt.xlim(-1, 2)
    plt.ylim(0, 2.5)
    plt.title(f"Interpolated Density $p_t(x)$ at t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("$p_t(x)$")
    frame_path = os.path.join(output_frames_dir, f"frame_{idx + 1}.png")
    plt.savefig(frame_path)
    plt.close()