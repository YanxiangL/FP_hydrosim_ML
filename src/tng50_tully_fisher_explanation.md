# Chapter X: A Computational Framework for Generating Tully-Fisher Relation Data from the TNG50 Simulation

## Abstract

This chapter details a Python-based computational framework designed to generate synthetic galaxy catalogs for the study of the Tully-Fisher Relation (TFR). The framework leverages the IllustrisTNG project's public data API to access and process galaxy properties from the TNG50 cosmological magnetohydrodynamical simulation. It implements a comprehensive pipeline that includes: (1) retrieving galaxy subhalo data from the TNG50 snapshot at redshift $z=0$; (2) deriving key physical properties such as stellar, gas, and halo masses; (3) calculating galaxy kinematics to estimate rotational velocities; and (4) implementing two distinct methods for assigning multi-wavelength absolute magnitudes. The first photometric method uses empirical stellar mass-to-light ratios, while the second, more physically motivated approach simulates the process by performing stellar population synthesis (SPS) and calculating dust attenuation based on the properties of individual stellar and gas particles. The final output is a rich dataset that enables detailed investigation into the TFR, its scatter, and its dependence on various galaxy properties. This document provides a thorough explanation of the underlying astrophysical concepts, the data processing pipeline, and the implementation details of the code.

---

## 1. Introduction

The Tully-Fisher Relation (TFR), first established by Tully and Fisher (1977), is a fundamental empirical correlation in extragalactic astronomy. It links the intrinsic luminosity of a spiral galaxy to its maximum rotation velocity. The relation is typically expressed as:

$L \propto V_{\text{rot}}^{\alpha}$

where $L$ is the galaxy's luminosity, $V_{\text{rot}}$ is its rotation velocity, and $\alpha$ is a slope that depends on the photometric band in which the luminosity is measured. Because luminosity can be used to determine distance, the TFR serves as a crucial extragalactic distance indicator, allowing for the mapping of the local Universe.

Modern cosmological simulations provide a powerful theoretical laboratory for studying the formation and evolution of galaxies in a cosmological context. The IllustrisTNG project (Nelson et al., 2019; Pillepich et al., 2018) is a suite of state-of-the-art simulations that model the evolution of dark matter, gas, stars, and supermassive black holes, successfully reproducing a wide range of observed galaxy properties. The TNG50 simulation (Pillepich et al., 2019; Nelson et al., 2019b), with its high resolution, offers a particularly valuable dataset for studying the detailed properties of individual galaxies.

This chapter describes a Python script, `tng50_tully_fisher.py`, developed to harness the TNG50 simulation data for a detailed study of the TFR. The primary goal of this framework is to generate a synthetic TFR dataset by post-processing simulation outputs. This allows for a controlled investigation into the physical drivers of the TFR and its scatter, which is not always possible with observational data alone.

The framework accesses TNG50 data via its web-based API, bypassing the need to download terabytes of raw simulation data. It processes this data to derive the two key components of the TFR: rotation velocity and absolute magnitude, implementing physically motivated models for each.

## 2. Methodology: Data Acquisition and Processing

The core of the framework is the `TNG50TullyFisherGenerator` class, which encapsulates all functionality. The methodology can be broken down into several key stages.

### 2.1. The IllustrisTNG Simulation and API Access

The framework is configured to use the **TNG50-1** simulation at **snapshot 99**, which corresponds to redshift $z=0$. All data is retrieved using the official TNG web API (`https://www.tng-project.org/api/`). This requires a personal API key, which the script reads from an environment variable (`TNG_API_KEY`) or prompts the user for. The script uses the `requests` library to handle all API calls directly.

The cosmological parameters used throughout the script are consistent with the TNG simulation, which is based on the Planck 2018 results (Planck Collaboration, 2020), specifically using `H0 = 67.74 km/s/Mpc` and `Ω_m = 0.3089`.

### 2.2. Galaxy Sample Selection

The first step in the pipeline is to retrieve a catalog of galaxies (or "subhalos" in simulation terminology) from the specified snapshot. This is handled by the `load_tng50_data` method.

1.  **API Call**: The method queries the `/api/TNG50-1/snapshots/99/subhalos/` endpoint. A `limit` parameter is used to specify the maximum number of subhalos to retrieve.
2.  **Filtering**: The script iterates through the retrieved subhalos and filters them based on their total stellar mass (`mass_stars`). The default range is set to select galaxies with stellar masses between $10^9$ and $10^{12} M_{\odot}$, which encompasses the typical mass range of spiral galaxies used in TFR studies.
3.  **Property Extraction**: For each subhalo that passes the filter, a second, more detailed API call is made to its specific endpoint (e.g., `/api/TNG50-1/snapshots/99/subhalos/{subhalo_id}/`). From this detailed record, a comprehensive set of properties is extracted, including:
    *   **Masses**: Stellar mass, gas mass, dark matter mass, and total subhalo mass. All are converted from the simulation's internal units to solar masses.
    *   **Kinematics**: The 3D velocity vector (`vel`), maximum circular velocity (`vmax`), and velocity dispersion (`veldisp`).
    *   **Structural Properties**: The stellar half-mass radius (`halfmassrad`), which serves as a proxy for the effective radius $R_{\text{eff}}$.
    *   **Baryonic Properties**: Star formation rate (`sfr`) and gas/stellar metallicities.

This process results in a `pandas` DataFrame (`self.galaxy_data`) containing the main properties of the selected galaxy sample.

## 3. Methodology: Deriving Physical Properties

With the galaxy catalog loaded, the next step is to derive the two primary quantities of the TFR: rotation velocity and absolute magnitude.

### 3.1. Kinematics: Rotation Velocity ($V_{\text{rot}}$)

The script uses the `SubhaloMaxCircVel` (`vmax`) from the TNG catalog as the primary proxy for a galaxy's rotation velocity. This value represents the peak of the subhalo's circular velocity profile, $V_c(r) = \sqrt{GM(<r)/r}$, and is a robust measure of the gravitational potential.

However, the TFR is observationally defined using the velocity of the flat part of the rotation curve, often measured at a large radius (e.g., 2.2 times the disk scale length). To account for this, the script applies a simple empirical correction factor, assuming $V_{\text{rot}} \approx 0.7 \times V_{\text{max}}$.

For completeness, the script also includes a fallback method, `calculate_rotation_velocity`, which attempts to compute the rotation velocity directly from mock gas particle kinematics. This method:
1.  Calculates the relative positions and velocities of gas particles with respect to the galaxy's center.
2.  Transforms these into a cylindrical coordinate system.
3.  Calculates the tangential velocity component ($v_{\phi}$) for each particle.
4.  Estimates $V_{\text{rot}}$ as the median tangential velocity of particles located near a target radius (2.2 times the half-mass radius).

**Note**: The particle data used in this calculation is **mock data** generated for demonstration purposes, as retrieving full particle cutouts for hundreds of galaxies via the API is computationally expensive. In the final analysis, the more reliable `vmax` from the TNG catalog is used.

### 3.2. Photometry: Absolute Magnitudes

A key feature of this framework is its implementation of two different methods for assigning absolute magnitudes to the simulated galaxies. This allows for a comparison between a simple empirical approach and a more detailed physical model.

#### 3.2.1. Approach 1: Empirical Mass-to-Light Ratios

The `_estimate_absolute_magnitude` method provides a quick, empirical estimate of a galaxy's absolute magnitude based solely on its total stellar mass. It uses a simple log-linear relation calibrated to match observational data (e.g., Bell et al., 2003):

$M_r = -1.8 \times (\log_{10}(M_*) - 10.0) - 20.5$

where $M_r$ is the absolute magnitude in the SDSS r-band and $M_*$ is the stellar mass in solar masses. Corrections for other photometric bands are applied as simple offsets. This method provides a baseline but ignores the detailed effects of star formation history, metallicity, and dust.

#### 3.2.2. Approach 2: Simulated Photometry (SKIRT Framework)

This approach, inspired by the "ExtinctionOnly" mode of radiative transfer codes like SKIRT (Camps & Baes, 2020), provides a much more physically grounded estimate of galaxy photometry. It involves two main steps: stellar population synthesis and dust attenuation.

**A. Stellar Population Synthesis (SPS)**

The `calculate_stellar_sed_skirt` method calculates the intrinsic (dust-free) spectral energy distribution (SED) of each galaxy.

1.  **Stellar Particle Properties**: For each galaxy, the script uses mock stellar particle data. Each particle has a mass, age, and metallicity. The age is derived from its formation time (`StellarFormationTime`), which is a scale factor in the TNG data, by converting it to a lookback time using the simulation's cosmology.
2.  **SSP Models**: The core of SPS is to assign a luminosity to each stellar particle based on its properties. This is done by interpolating from a grid of Simple Stellar Population (SSP) models. The script uses mock SSP templates that are designed to follow realistic trends based on established models like BC03 (Bruzual & Charlot, 2003) or FSPS (Conroy, Gunn, & White, 2009). In these models:
    *   Luminosity per unit mass is a strong function of age (younger populations are much brighter, especially in the UV and blue bands).
    *   Luminosity also depends on metallicity.
3.  **Interpolation**: A 2D interpolation is performed on the SSP grid in (age, metallicity) space to find the luminosity per solar mass for each particle in various broadband filters (from FUV to mid-IR).
4.  **Total Luminosity**: The total intrinsic luminosity of the galaxy in a given band is the sum of the luminosities of all its stellar particles.

**B. Dust Attenuation**

The `calculate_dust_extinction_skirt` method calculates the amount of light absorbed and scattered by dust.

1.  **Dust-to-Gas Ratio**: The model first assumes that dust follows the gas. The dust-to-gas ratio is estimated from the metallicity of the gas particles, following the empirical relation from Rémy-Ruyer et al. (2014), where the ratio scales linearly with metallicity.
2.  **Dust Surface Density**: A simplified dust surface density is calculated by projecting the total dust mass (gas mass × dust-to-gas ratio) onto an effective area derived from the galaxy's size. This is a significant simplification of a full 3D radiative transfer calculation.
3.  **Extinction Law**: The extinction in magnitudes for each photometric band ($A_{\lambda}$) is calculated by multiplying the dust surface density by an extinction coefficient. This coefficient is determined by the **Cardelli, Clayton & Mathis (1989) extinction law**, which describes the wavelength-dependent behavior of interstellar dust extinction.

**C. Final Magnitudes**

Finally, the `generate_absolute_magnitudes_skirt` method combines these results.
*   The **intrinsic (dust-free) absolute magnitude** is calculated from the total luminosity derived from the SPS step using the standard formula: $M_{\text{band}} = M_{\text{sun, band}} - 2.5 \log_{10}(L_{\text{band}}/L_{\odot})$.
*   The **attenuated (dusty) absolute magnitude** is then calculated by adding the extinction: $M_{\text{dusty}} = M_{\text{intrinsic}} + A_{\lambda}$.

This provides two sets of physically motivated magnitudes: one representing the intrinsic stellar light and one representing what an observer might see after the light has been attenuated by dust.

## 4. Analysis and Visualization

Once the final DataFrame is generated with all derived properties, the script provides functions for analysis and visualization.

*   **`generate_tully_fisher_data`**: This is the main driver method that iterates through each galaxy, calls the routines for calculating rotation velocity and magnitudes, and compiles the results into a final CSV file (`tng50_tully_fisher_complete.csv`).
*   **`fit_tully_fisher_relation`**: This function performs a linear least-squares fit to the TFR in log-space ($\log(V_{\text{rot}})$ vs. absolute magnitude or $\log(M_*)$). It returns the slope, intercept, and scatter ($\sigma$) of the relation.
*   **`plot_tully_fisher_relation`**: This function visualizes the TFR, plotting the data points, the best-fit line, and the 1-$\sigma$ scatter.
*   **`create_comprehensive_plots`**: This function generates a suite of diagnostic plots to explore other well-known galaxy scaling relations, such as the stellar mass-halo mass relation and the size-mass relation.
*   **`compare_magnitude_methods_updated`**: This function creates plots directly comparing the absolute magnitudes derived from the simple mass-to-light ratio method versus the more complex SPS+dust method, which is crucial for validating the models.

## 5. Conclusion and Future Work

The `tng50_tully_fisher.py` script provides a powerful and flexible framework for generating synthetic Tully-Fisher data from the TNG50 simulation. By leveraging the TNG API, it allows for detailed astrophysical modeling without the need for local access to the full simulation dataset. The implementation of a physically motivated pipeline for calculating galaxy magnitudes, including stellar population synthesis and dust attenuation, represents a significant step beyond simple empirical relations.

The resulting datasets can be used to:
*   Study the intrinsic TFR and compare it to observational data.
*   Investigate the physical origins of the scatter in the TFR by correlating residuals with other galaxy properties (e.g., gas fraction, star formation rate, morphology).
*   Explore the TFR across 15 different photometric bands, from the FUV to the mid-IR.

Future work could involve replacing the mock particle data with real particle cutouts from the TNG API for a more accurate kinematic analysis and replacing the mock SSP templates with data loaded directly from standard models like BC03 or FSPS. Nonetheless, this framework provides a robust and well-documented foundation for theoretical studies of the Tully-Fisher relation.

## 6. References

*   Bell, E. F., McIntosh, D. H., Katz, N., & Weinberg, M. D. (2003). The Optical and Near-Infrared Properties of Galaxies. I. Luminosity and Stellar Mass Functions. *The Astrophysical Journal Supplement Series*, 149(2), 289.
*   Bruzual, G., & Charlot, S. (2003). Stellar population synthesis at the resolution of 2003. *Monthly Notices of the Royal Astronomical Society*, 344(4), 1000-1028.
*   Camps, P., & Baes, M. (2020). The SKIRT code for dust radiative transfer in galaxy simulations. *Astronomy & Computing*, 31, 100381.
*   Cardelli, J. A., Clayton, G. C., & Mathis, J. S. (1989). The relationship between infrared, optical, and ultraviolet extinction. *The Astrophysical Journal*, 345, 245-256.
*   Conroy, C., Gunn, J. E., & White, M. (2009). The Propagation of Uncertainties in Stellar Population Synthesis Modeling. I. The Age-Metallicity Degeneracy. *The Astrophysical Journal*, 699(1), 486.
*   Nelson, D., Springel, V., Pillepich, A., et al. (2019). The IllustrisTNG simulations: public data release. *Computational Astrophysics and Cosmology*, 6(1), 2.
*   Nelson, D., Pillepich, A., Springel, V., et al. (2019b). First results from the TNG50 simulation: the galaxy population at z=1 and the problem of quenching. *Monthly Notices of the Royal Astronomical Society*, 490(3), 3234-3261.
*   Pillepich, A., Springel, V., Nelson, D., et al. (2018). First results from the IllustrisTNG simulations: the galaxy colour bimodality. *Monthly Notices of the Royal Astronomical Society*, 475(1), 648-675.
*   Pillepich, A., Nelson, D., Springel, V., et al. (2019). First results from the TNG50 simulation: the stellar mass-size relation of star-forming and quiescent galaxies. *Monthly Notices of the Royal Astronomical Society*, 490(3), 3196-3233.
*   Planck Collaboration, Aghanim, N., Akrami, Y., et al. (2020). Planck 2018 results. VI. Cosmological parameters. *Astronomy & Astrophysics*, 641, A6.
*   Rémy-Ruyer, A., Madden, S. C., Galliano, F., et al. (2014). Gas-to-dust mass ratios in local galaxies over a 2 dex metallicity range. *Astronomy & Astrophysics*, 563, A31.
*   Tully, R. B., & Fisher, J. R. (1977). A new method of determining distances to galaxies. *Astronomy and Astrophysics*, 54, 661-673.
