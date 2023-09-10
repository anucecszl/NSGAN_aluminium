import pandas as pd
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PM
from joblib import dump, load
import alaGAN
import torch

# Load the dataset and extract element and process names.
element_names = pd.read_excel('Al_mechanical_dataset.xlsx').columns.values[5:30]
process_names = pd.read_excel('Al_mechanical_dataset.xlsx').columns.values[30:40]
dataset_df = pd.read_excel('Al_mechanical_dataset.xlsx')
data_np = dataset_df.to_numpy()

# Normalize features of the alloys.
comp_data = data_np[:, 5:40].astype(float)
comp_min = np.min(comp_data, axis=0)
comp_max = np.max(comp_data, axis=0)

feature_names = dataset_df.columns.values[5:40]

# Load pretrained GAN generator.
generator = alaGAN.Generator()
generator.load_state_dict(torch.load('generator_net_aluminium_gp.pt'))
generator.eval()

# Load pretrained regression models for mechanical properties.
yield_regressor = load('yield_regressor.joblib')
tensile_regressor = load('tensile_regressor.joblib')
elongation_regressor = load('elongation_regressor.joblib')


class AlloyOptimizationProblem(Problem):
    def __init__(self):
        # Define bounds and number of objectives.
        super().__init__(n_var=10, n_obj=2, xl=-3, xu=3)

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert numpy input to PyTorch tensor.
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            # Use GAN to generate alloy compositions.
            fake_alloys = generator(x_tensor).numpy()
        # Denormalize the generated alloy features.
        fake_alloys = fake_alloys * comp_max + comp_min
        # Calculate the objectives.
        f1 = - tensile_regressor.predict(fake_alloys)
        f2 = - elongation_regressor.predict(fake_alloys)
        out["F"] = np.column_stack([f1, f2])


# Set up the optimization problem and algorithm.
problem = AlloyOptimizationProblem()
algorithm = NSGA2(pop_size=250, mutation=PM(prob=0.1, eta=20))
termination = get_termination("n_gen", 100)

# Run the optimization.
res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(), save_history=True,
               seed=1,
               verbose=False)

# Convert the optimal solutions to actual alloy compositions using the GAN.
result_tensor = torch.tensor(res.X, dtype=torch.float32)
with torch.no_grad():
    optimal_alloys = generator(result_tensor).numpy()

optimal_alloys = optimal_alloys * comp_max + comp_min
optimal_alloys[:, :25] = optimal_alloys[:, :25] / np.sum(optimal_alloys[:, :25], axis=1).reshape((-1, 1))

# Extract names and processing methods for the optimal alloys.
alloy_names = []
for i in range(optimal_alloys.shape[0]):
    composition = optimal_alloys[i, :25]
    comp_string = ''
    for j in range(len(composition)):
        if composition[j] > 0.0001:
            comp_string += element_names[j]
            comp_string += str(round(composition[j], 4))
            composition[j] = 0
    alloy_names.append(comp_string)

# Get optimal processing methods.
process_indices = np.argmax(optimal_alloys[:, 25:], axis=1)
process_one_hot = np.zeros_like(optimal_alloys[:, 25:])
for i in range(len(process_one_hot)):
    process_one_hot[i][process_indices[i]] = 1
optimal_alloys[:, 25:] = process_one_hot

# Calculate the properties of optimal alloys using regression models.
property_array = np.zeros((optimal_alloys.shape[0], 3))
property_array[:, 0] = elongation_regressor.predict(optimal_alloys)
property_array[:, 1] = tensile_regressor.predict(optimal_alloys)
property_array[:, 2] = yield_regressor.predict(optimal_alloys)

process_name_list = [process_names[index] for index in process_indices]

# Create DataFrames for the output.
formula_df = pd.DataFrame(zip(alloy_names, process_name_list),
                          columns=['Composition', 'Processing method'])
property_df = pd.DataFrame(property_array, columns=['Elongation', 'Tensile', 'Yield'])
feature_df = pd.DataFrame(optimal_alloys, columns=list(feature_names))
output_df = pd.concat([formula_df, property_df, feature_df], axis=1)

# Save the results to an Excel file.
output_df.to_excel('NSGAN_result.xlsx', index=False)
