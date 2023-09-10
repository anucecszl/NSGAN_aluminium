# NSGAN Aluminium Alloy Optimization

This repository showcases the **Non-Dominant Sorting Generative Adversarial Networks (NSGAN)**. NSGAN is an innovative generative design framework, strategically combining the data generation capabilities of Generative Adversarial Networks (GAN) with the multi-objective optimization prowess of the NSGA-II algorithm. The practical implementation of this framework is demonstrated in the context of aluminium alloy generation and optimization. This repository provides a snapshot of the proposed framework, focusing on certain components of our code, mainly the GAN and NSGAN models. Alongside the code, we've included the training dataset, a database containing optimized aluminium alloy, as well as the parameters of machine learning models saved throughout this endeavor.

![image](https://github.com/anucecszl/NSGAN_aluminium/assets/51730485/831a1fb6-5967-4404-9a19-f4971d79d931)


## Repository Contents

- **Al_mechanical_dataset.xlsx**: A comprehensive dataset containing mechanical properties of diverse aluminium alloys.
- **NSGAN.py**: A Python script that implements the NSGAN algorithm, tailored for multi-objective optimization in metallurgical contexts.
- **alaGAN.py**: A GAN model designed to generate synthetic data pertinent to aluminium alloys.
- **generator_net_aluminium_gp.pt**: Trained model parameters for the GAN used in the synthetic generation of aluminium alloy data.
- **elongation_regressor.joblib, tensile_regressor.joblib, yield_regressor.joblib**: Pretrained regression models that efficiently predict specific alloy properties.
- **optimised_aluminium_samples.xlsx**: Resultant file showcasing optimized aluminium alloy samples derived from the implemented framework.
- **requirements.txt**: List of project dependencies. Installation is achieved via `pip install -r requirements.txt`.

The intent of this repository is to provide insights and tools that merge GANs and multi-objective optimization, specifically for aluminium alloy research and development.

