# Lake Infection Simulator

The **Lake Infection Simulator** is a simulation of infection spread in a lake environment using cellular automata principles combined with morphological operations and local environmental conditions. The project applies discrete modeling techniques to visualize the dynamics of infection over time.

## Features

- **Infection Simulation**:
  - The infection spreads based on local environmental factors such as temperature.
  - Cellular states include: Healthy, Early Stage Infection, Advanced Infection, Recovered, Dead, and Water.

- **Morphological Image Operations**:
  - Load an image as a grid representation of the environment.
  - Perform dilation and other morphological operations to preprocess input data.

- **Interactivity**:
  - Drop medicine during the simulation to heal a large portion of infected cells (`90%`).

- **Visualization**:
  - Animated visualization of infection progression over time.
  - Color-coded map for various states in the environment.

- **Cellular Automata Rules**:
  - Automata-based rules drive the state transitions of cells based on neighbor states and probabilistic conditions influenced by environmental factors.

## How to Use

1. Prepare an input image representing the lake environment, where:
   - White pixels represent land.
   - Black pixels represent water.

2. Run the infection simulation:
   ```python
   animate_infection('path_to_your_image.bmp', steps=50)
3. Use the M key during the animation to activate the medicine drop feature.

4. Optionally, preprocess the input image with morphological operations such as dilation:

  from morphological_operations import dilate
  processed_image = dilate(image, radius=5)
  save_and_show_image(processed_image, 'output_path.bmp')

## Requirements
  - Python 3.8+
  - Libraries: numpy, matplotlib, opencv-python, Pillow

## Inspiration
This project was developed as part of discrete modeling coursework to explore phenomena in natural environments and experiment with computational techniques such as cellular automata and morphological operations.

## How It Works
The simulation leverages cellular automata rules to model infection spread, incorporating:

  - Neighbor cell analysis to determine infection probability.
  - Temperature-dependent probabilities for infection, recovery, and reinfection.
  - Morphological preprocessing to adjust environmental constraints such as water boundaries.
The cellular automaton evolves based on probabilistic state transitions, visualized frame by frame in an animatio
