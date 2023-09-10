# NSGAN Aluminium Alloy Optimization

This repository presents an innovative generative design framework: Non-Dominant Sorting Generative Adversarial Networks (NSGAN). This framework utilizes the capabilities of Generative Adversarial Networks (GAN) for data generation and integrates the NSGA-II algorithm for multi-objective optimization of aluminium alloy properties.

## Repository Contents

- **Al_mechanical_dataset.xlsx**: A comprehensive dataset containing mechanical properties of diverse aluminium alloys.
- **NSGAN.py**: A Python script that implements the NSGAN algorithm, tailored for multi-objective optimization in metallurgical contexts.
- **alaGAN.py**: A GAN model designed to generate synthetic data pertinent to aluminium alloys.
- **generator_net_aluminium_gp.pt**: Trained model parameters for the GAN used in the synthetic generation of aluminium alloy data.
- **elongation_regressor.joblib, tensile_regressor.joblib, yield_regressor.joblib**: Pretrained regression models that efficiently predict specific alloy properties.
- **optimised_aluminium_samples.xlsx**: Resultant file showcasing optimized aluminium alloy samples derived from the implemented framework.
- **requirements.txt**: List of project dependencies. Installation is achieved via `pip install -r requirements.txt`.

The intent of this repository is to provide insights and tools that merge GANs and multi-objective optimization, specifically for aluminium alloy research and development.

