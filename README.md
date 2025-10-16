# Revolutionizing Distance Indicators with AI-Driven Symbolic Regression

This project aims to revolutionise astronomical distance measurement techniques by applying AI-driven symbolic regression to discover more accurate and interpretable alternatives to traditional methods like the Tully-Fisher relation. By analysing simulated datasets that reproduce known galactic properties and distances, we seek to uncover mathematical expressions that more precisely describe these relationships, potentially reducing the current **20% error margin** in distance measurements.

## Project Structure

```
TF_symbolic_regression/
├── input/          # Input data files and simulated datasets
├── src/            # Python source code and scripts
├── models/         # Trained models and saved results
├── notebooks/      # Jupyter notebooks for analysis and exploration
└── README.md       # Project documentation
```

## Overview

The Tully-Fisher relation is a fundamental scaling relation in astronomy that connects the luminosity of spiral galaxies to their rotation velocity. However, current distance indicators suffer from significant uncertainties that limit their application to higher redshifts. This project leverages advanced machine learning algorithms to explore the vast combinatorial space of possible equations, with the goal of developing distance indicators that can be extended to higher redshifts with improved accuracy.

## Key Objectives

- **Apply symbolic regression** to simulated datasets that reproduce the Tully-Fisher relation
- **Discover new mathematical expressions** that may outperform current distance indicator techniques
- **Reduce the intrinsic scatter** in distance measurements, enabling extension to higher redshifts
- **Compare AI-discovered relations** to human-derived formulas, including the original Tully-Fisher relation
- **Develop more interpretable alternatives** to black-box neural networks for distance estimation

## Methodology

1. **Generate simulated datasets** incorporating known galactic properties and distances with realistic observational effects
2. **Implement symbolic regression algorithms** to explore the vast combinatorial space of possible mathematical expressions
3. **Evaluate discovered equations** based on accuracy, complexity, and interpretability
4. **Compare AI-generated formulas** with traditional distance indicator relations
5. **Visualize results** in complexity vs. performance plots similar to recent breakthroughs in AI-driven physics discovery

## Expected Outcomes

- **A set of novel mathematical expressions** for estimating extragalactic distances
- **Potential improvement** over the 20% error margin of current distance indicators  
- **Insights into the effectiveness** of human-derived vs. AI-discovered astronomical relations
- **A complexity-performance visualization** plotting discovered equations by increasing complexity (x-axis) against performance score (y-axis), similar to Figure 5 in [Lemos et al. 2022](https://arxiv.org/abs/2202.02306)

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn gplearn sympy
```

### Quick Start

1. **Generate simulated data:**
   ```bash
   cd src/
   python tf_simulation.py
   ```
   This creates a simulated Tully-Fisher dataset in the `input/` folder with realistic observational effects and hidden systematic biases.

2. **Run symbolic regression analysis:**
   ```bash
   cd notebooks/
   jupyter notebook
   ```
   Open and run the analysis notebooks to discover new distance indicator expressions.

## Files Description

### src/
- `tf_simulation.py` - Main simulation script generating realistic Tully-Fisher datasets with observational effects
- `symbolic_regression.py` - (Planned) Implementation of symbolic regression algorithms
- `evaluation_metrics.py` - (Planned) Tools for evaluating discovered expressions

### input/
- `tully_fisher_simulated_dataset.csv` - Generated simulated galaxy dataset with 20 physical and observational properties
- `real_tf_data.csv` - (Future) Real observational data for validation

### models/
- (Future) Saved symbolic regression models and discovered mathematical expressions

### notebooks/
- `project_structure_setup.ipynb` - Project setup and data generation verification
- `data_exploration.ipynb` - (Planned) Exploratory data analysis of simulated datasets
- `symbolic_regression_analysis.ipynb` - (Planned) Main symbolic regression implementation and results
- `performance_comparison.ipynb` - (Planned) Comparison of AI-discovered vs. traditional relations

## Technical Approach

This project implements state-of-the-art symbolic regression techniques inspired by:

- **AI Feynman methodology** for discovering interpretable physical laws
- **Genetic programming approaches** for exploring mathematical expression space
- **Multi-objective optimization** balancing accuracy, complexity, and interpretability
- **Cross-validation techniques** ensuring robust performance on unseen data

## Features

- **Realistic galaxy simulation** with proper mass-velocity-luminosity relationships
- **Observational effects** including photometric errors, velocity uncertainties, and selection biases
- **Hidden systematic effects** that can be discovered by AI/ML techniques
- **Comprehensive visualization** of the simulated datasets and discovered relations
- **Performance benchmarking** against traditional distance indicators

## Impact and Applications

The successful completion of this project will:

- **Advance astronomical distance measurements** with potentially transformative accuracy improvements
- **Demonstrate AI interpretability** in scientific discovery contexts
- **Provide insights** into the fundamental physics governing galaxy scaling relations
- **Enable cosmological studies** at higher redshifts with improved precision
- **Establish a framework** for AI-driven discovery in other areas of astrophysics

## References

This project builds upon cutting-edge research in AI-driven scientific discovery:

- **AI Feynman**: [A Physics-Inspired Method for Symbolic Regression](https://arxiv.org/abs/2006.10782) - Foundational methodology for discovering interpretable physical laws
- **Rediscovering orbital mechanics with machine learning**: [Lemos et al. 2022](https://arxiv.org/abs/2202.02306) - Inspiration for our complexity-performance visualization approach (targeting Figure 5 style results)
- **Tully-Fisher relation in MaNGA and IllustrisTNG**: [Pelliciari et al. 2023](https://arxiv.org/abs/2302.05029) - Modern observational and simulation constraints
- **Tully-Fisher relation in the SIMBA simulation**: [Glowacki et al. 2020](https://arxiv.org/abs/2003.03402) - Simulation-based validation framework
- **Tully-Fisher relation in the TNG50 cosmological simulation**: [Pulsoni et al. 2025](https://arxiv.org/abs/2503.00194) - Latest high-resolution simulation results

## Contributing

This project is part of ongoing research into AI-enhanced astronomical distance indicators. Contributions are welcome in the following areas:

- **Algorithm development** for symbolic regression in astrophysical contexts
- **Observational data integration** for validation and benchmarking
- **Performance optimization** of symbolic regression algorithms
- **Visualization improvements** for scientific interpretation

## Future Work

- Integration with real observational datasets (MaNGA, GAIA, etc.)
- Extension to other distance indicators (Surface Brightness Fluctuations, Type Ia Supernovae)
- Application to higher-redshift galaxy populations
- Development of uncertainty quantification frameworks for AI-discovered relations

## License

Academic use only. Please cite this work if used in research publications.

## Contact

For questions, collaborations, or access to preliminary results, please open an issue or contact the research team.

---

*"The goal is not just to find better distance indicators, but to understand why they work and what fundamental physics they reveal about the universe."*
