# Generative Energy Based Model (GEBM) and Joint Energy Based Model (JEBM)

This project explores the training and analysis of Generative Energy Based Models (GEBM) and Joint Energy Based Models (JEBM). The primary focus is on investigating the deep connection between generative and classifier models, with an emphasis on training generative classifiers by learning JEBM and unlearning GEBM. Additionally, the project quantifies the learned information content by computing thermodynamic quantities associated with the models, providing insights into the thermodynamics of GEBM and JEBM.

## Project Structure

The project is organized into several modules, each serving a specific purpose:

- `data_setup.py`: Initializes data loaders for the classifier and sets up the dataset.
- `visualization.py`: Contains functions for visualizing thermodynamic quantities associated with the models.
- `losses.py`: Defines classes for computing various optimization objectives, including generative classifier learning and EBM training.
- `LangevianMC.py`: Implements the Langevin Monte Carlo sampler for sampling from the models.
- `model.py`: Defines the neural network structures for the GEBM and JEBM.
- `main.py`: The main script that orchestrates the training process, sets hyperparameters, and visualizes results.

## Getting Started

1. Install the required dependencies:

    ```bash
    pip install torch torchvision matplotlib
    ```

2. Run the main script:

    ```bash
    python main.py
    ```

## Hyperparameters and Configurations

- `cl_type`: Type of optimization objective, including 'gcl', 'dcl', 'm_ebm', and 'j_ebm'.
- `noise`: Level of noise added to the original images during training.
- `model_size`: Size of the neural network models.
- `model_type`: Type of the model architecture, such as 'shallow' or 'cnn'.
- `gamma`: Trade-off parameter between joint and marginal loss.
- `epochs`: Number of training epochs.
- `learning_rate`: Learning rate for the optimizer.
- `from_scratch`: Percentage of new images generated from scratch.
- `optimizer_type`: Type of optimizer, either 'sgd' or 'adam'.
- `MC_steps`: Number of Monte Carlo steps for sampling.
- `sample_size`: Batch size for sampling.

## Results

The project generates and analyzes thermodynamic quantities, providing insights into the behavior and information content of the trained models. Visualizations and plots are available to help interpret the results.

