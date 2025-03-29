# Multi-Objective Optimization Anonymization Model (MO-OBAM) using Particle Swarm Optimization 

**Overview**

This repository contains the implementation of a multi-objective optimization anonymization model (MO-OBAM) using the Particle Swarm Optimization (PSO) algorithm. The model is designed to optimize the trade-offs between privacy preservation and machine learning performance in structured data.

**Usage**

To implement the MO-OBAM model:
1. Open `constants.py` and define:
   - Numeric quasi-identifiers
   - Categorical quasi-identifiers
   - Sensitive attributes in your dataset

2. Open `MO-OBAM.ipynb`, which is the main file to run the model.
   - Define all required parameters in `parameters_dic`.
   - In the `run_particle_swarm_experiment()` function, set:
     - `n_population`: Number of particles
     - `maxIter`: Number of iterations (for PSO algorithm)

3. Running the notebook will:
   - Save the anonymized dataset to your PC.
   - Store results tracking the values of each component in the MO-OBAM model (e.g., information loss, entropy, k-anonymity violation) through iterations.
   - Allow you to visualize how PSO finds global solutions over iterations.

**Dependencies**

Ensure you have the following Python packages installed:
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib

Install them using:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

**Citation**

If you use this repository for your research, please cite:
> Yusi Wei, Hande Y Benson, Joseph K Agor, and Muge Capan. 2025. Multi-Objective Optimization-Based Anonymization of Structured Data for
Machine Learning. _arXiv preprint arXiv:2501.01002 (2025)_.

**Contact**

For questions or collaborations, please contact [yw825@drexel.edu].

