# An Interpretable Slope Displacement Decomposition Model Based on Physics-Informed Neural Networks

The  Physics - Informed Neural Network (PINN) methods were employed to decompose slope displacement. This decomposition was carried out considering factors such as rainfall, blasting, creep, and excavation.

## Features

- **Hybrid Modeling**: Combines a predefined physical equation with a data-driven optimization approach.
- **Constraint Handling**: Integrates specific physical constraints into the loss function to ensure the learned parameters are physically meaningful.
- **Parameter Control**: Uses `nn.Softplus` to enforce positivity for certain parameters while allowing others to be negative.
- **Reproducibility**: Sets random seeds for PyTorch and NumPy to ensure consistent results.
- **Visualization**: Generates comprehensive plots to evaluate model performance, including:
    - True vs. Predicted values.
    - Prediction error distribution.
    - Loss history over epochs.
    - Contribution of each component to the final prediction.
- **Organized Output**: Saves all generated plots and the final results DataFrame into a dedicated `results` directory.

## Project Structure

```
.
├── .gitignore
├── data/
│   └── example_data.xlsx   # Input data file
├── model.py                # Main training script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
The `results/` directory will be created automatically upon running the script.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Simply run the main script from the project's root directory:

```bash
python model.py
```

The script will:
1.  Load the data from `data/example_data.xlsx`.
2.  Train the parameter network, printing the loss every 100 epochs.
3.  Display the final fitted parameters and how well they satisfy the constraints.
4.  Show and save the result plots (`results_summary.png`, `Decomposition.png`) in the `results` folder.
5.  Save the detailed prediction results and parameters to `results/results.xlsx`.

## Model Equation

The model aims to fit the parameters of the following equation:

$Y = a_1 \cdot x_1 + b_1 \cdot x_2 + (c_1 \cdot x_3 + c_2 \cdot (1 - e^{-c_3 \cdot x_3})) + (c_4 \cdot x_4 + c_5 \cdot (1 - e^{-c_6 \cdot x_4})) + (d_1 \cdot \sqrt{x_5} + d_2 \cdot x_5) + (d_3 \cdot \sqrt{x_6} + d_4 \cdot x_6)$

With the following constraints:
1.  $d_1 \cdot \sqrt{16} + d_2 \cdot 16 = 9.6$
2.  $d_3 \cdot \sqrt{15} + d_4 \cdot 15 = 15.2$
