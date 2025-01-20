import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import matplotlib.colors as mcolors
import lab_02

# Definitions of cell states
HEALTHY = 0             # Healthy cell
INFECTED_STAGE_1 = 1    # Early stage of infection
INFECTED_STAGE_2 = 2    # Advanced stage of infection
RECOVERED = 3           # Recovered cell
DEAD = 4                # Dead cell
WATER = -1              # Water area

def load_image_as_grid(filename):
    """
    Load an image and convert it into a grid representation.
    Water areas are marked with -1, and healthy cells are marked with 0.
    """
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('The image file could not be loaded.')
    
    threshold_value, thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    grid = np.where(thresh == 0, WATER, HEALTHY)
    return grid

def generate_temperature_grid(grid, low=0, high=30):
    """
    Generate a temperature grid based on the input grid.
    The grid is divided into three horizontal sections with different temperature ranges.
    """
    rows, cols = grid.shape
    one_third = rows // 3

    temperature_grid = np.zeros_like(grid, dtype=float)
    temperature_grid[:one_third, :] = low  # Top section
    temperature_grid[one_third:2*one_third, :] = (low + high) / 2  # Middle section
    temperature_grid[2*one_third:, :] = high  # Bottom section

    return temperature_grid

def apply_rules(grid, temperature_grid):
    """
    Apply the infection rules to the grid based on the current state and temperature grid.
    """
    new_grid = np.copy(grid)
    infected_neighbors = np.zeros(grid.shape, dtype=int)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (dx, dy) != (0, 0):
                shifted_grid = np.roll(grid, (dx, dy), axis=(0, 1))
                infected_neighbors += (shifted_grid == INFECTED_STAGE_1) | (shifted_grid == INFECTED_STAGE_2)

    infection_chance = np.where(temperature_grid < 10, 0.05, 
                                np.where(temperature_grid < 25, 0.15, 0.1))
    recovery_chance = np.where(temperature_grid > 25, 0.4, 0.2)
    reinfection_chance = np.where(temperature_grid > 20, 0.05, 0.01)

    water_cells = (grid == WATER)
    new_grid[water_cells & (infected_neighbors > 0) & 
             (np.random.rand(*grid.shape) < infection_chance * infected_neighbors)] = INFECTED_STAGE_1

    new_grid[grid == INFECTED_STAGE_1] = INFECTED_STAGE_2
    new_grid[(grid == INFECTED_STAGE_2) & (np.random.rand(*grid.shape) < 0.2)] = DEAD
    new_grid[(grid == INFECTED_STAGE_2) & (np.random.rand(*grid.shape) < recovery_chance)] = RECOVERED
    new_grid[(grid == RECOVERED) & (infected_neighbors > 2) & 
             (np.random.rand(*grid.shape) < reinfection_chance)] = INFECTED_STAGE_1

    return new_grid

def initialize_infection(grid):
    """
    Initialize the infection in random water cells.
    """
    water_cells = np.argwhere(grid == WATER)
    if len(water_cells) > 0:
        initial_infected = np.random.choice(len(water_cells), size=min(10, len(water_cells)), replace=False)
        for idx in initial_infected:
            x, y = water_cells[idx]
            grid[x, y] = INFECTED_STAGE_1

def drop_medicine(grid):
    """
    Drop medicine, healing 90% of infected cells.
    """
    infected_cells = np.argwhere((grid == INFECTED_STAGE_1) | (grid == INFECTED_STAGE_2))
    if len(infected_cells) > 0:
        cells_to_heal = np.random.choice(len(infected_cells), size=int(0.9 * len(infected_cells)), replace=False)
        for idx in cells_to_heal:
            x, y = infected_cells[idx]
            grid[x, y] = RECOVERED
    print("Medicine dropped: 90% of infected cells healed.")

def animate_infection(filename, steps=50):
    """
    Animate the spread of infection based on an input image file.
    """
    grid = load_image_as_grid(filename)
    initialize_infection(grid)
    temperature_grid = generate_temperature_grid(grid)

    # Color map for the states
    cmap = mcolors.ListedColormap(['#ADD8E6', 'green', 'orange', 'red', '#00FFFF', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Set up the animation
    fig, ax = plt.subplots()
    mat = ax.matshow(grid, cmap=cmap, norm=norm)

    # Create legend for the states
    labels = ['Water', 'Land', 'Infected (Stage 1)', 'Infected (Stage 2)', 'Recovered', 'Dead']
    colors = [cmap(norm(i)) for i in range(-1, 5)]
    patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i],
                        label="{:s}".format(labels[i]))[0] for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    def update(frame):
        nonlocal grid
        grid = apply_rules(grid, temperature_grid)
        mat.set_data(grid)
        return [mat]

    def on_key(event):
        if event.key == 'm':  # Drop medicine
            print("Pressed 'm' - Dropping medicine.")
            drop_medicine(grid)

    fig.canvas.mpl_connect('key_press_event', on_key)
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
    plt.show()

# Execute animation based on the provided image file
animate_infection(r'D:\Studia\modelowanie_dyskretne\lab_05\dilated_image2.bmp', steps=50)
