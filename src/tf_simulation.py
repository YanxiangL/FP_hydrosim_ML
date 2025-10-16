import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm
import seaborn as sns

class TullyFisherSimulator:
    def __init__(self, seed=42):
        """
        Initialize the Tully-Fisher relation simulator
        
        Parameters:
        seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Standard Tully-Fisher parameters
        # L = A * v^alpha where L is luminosity, v is velocity width
        self.tf_slope = 4.0  # alpha parameter (typical value ~3.5-4.5)
        self.tf_zero_point = -21.0  # A parameter in magnitude units (more realistic)
        self.intrinsic_scatter = 0.3  # intrinsic scatter in magnitudes
        
        # Physical parameters for more realistic simulation
        self.mass_velocity_slope = 3.5  # M_halo ∝ v^β
        self.mass_light_ratio_mean = 10.0  # M/L ratio
        self.mass_light_ratio_scatter = 0.2
        
    def generate_galaxy_properties(self, n_galaxies=1000):
        """
        Generate fundamental galaxy properties
        
        Parameters:
        n_galaxies (int): Number of galaxies to simulate
        
        Returns:
        dict: Dictionary containing galaxy properties
        """
        # Generate velocity widths (km/s) - log-normal distribution
        log_velocity_mean = np.log10(200)  # ~200 km/s typical
        log_velocity_std = 0.3
        log_velocities = np.random.normal(log_velocity_mean, log_velocity_std, n_galaxies)
        velocities = 10**log_velocities
        
        # Generate halo masses from velocity widths
        log_halo_masses = np.log10(1e12) + self.mass_velocity_slope * (log_velocities - log_velocity_mean)
        halo_masses = 10**log_halo_masses
        
        # Generate stellar masses (with scatter)
        stellar_mass_fraction = 0.1  # typical f_* ~ 0.1
        stellar_mass_scatter = 0.3
        log_stellar_masses = (np.log10(halo_masses) + 
                            np.log10(stellar_mass_fraction) + 
                            np.random.normal(0, stellar_mass_scatter, n_galaxies))
        stellar_masses = 10**log_stellar_masses
        
        # Generate surface brightness (affects observability)
        surface_brightness = np.random.normal(22.0, 1.5, n_galaxies)  # mag/arcsec^2
        
        # Generate inclination angles (affects observed velocity width)
        inclinations = np.arccos(np.random.uniform(0.3, 1.0, n_galaxies))  # avoid edge-on
        
        # Generate distances (Mpc) - more realistic for TF surveys
        distances = np.random.lognormal(np.log(30), 0.6, n_galaxies)
        distances = np.clip(distances, 5, 150)  # typical TF survey range
        
        # Generate galaxy types (affects TF relation)
        galaxy_types = np.random.choice(['Spiral', 'Irregular', 'Dwarf'], 
                                      n_galaxies, p=[0.7, 0.2, 0.1])
        
        return {
            'velocity_width': velocities,
            'halo_mass': halo_masses,
            'stellar_mass': stellar_masses,
            'distance': distances,
            'inclination': inclinations,
            'surface_brightness': surface_brightness,
            'galaxy_type': galaxy_types
        }
    
    def apply_tully_fisher_relation(self, properties):
        """
        Apply the Tully-Fisher relation to generate luminosities
        
        Parameters:
        properties (dict): Galaxy properties from generate_galaxy_properties
        
        Returns:
        dict: Updated properties with luminosities and magnitudes
        """
        velocities = properties['velocity_width']
        
        # Correct for inclination
        corrected_velocities = velocities / np.sin(properties['inclination'])
        
        # Apply basic TF relation: M = A - α * log(v)
        # where M is absolute magnitude, more negative = brighter
        # Adjusted zero point to be more realistic
        absolute_magnitudes = (self.tf_zero_point - 
                             self.tf_slope * np.log10(corrected_velocities/100))  # normalized
        
        # Add intrinsic scatter
        absolute_magnitudes += np.random.normal(0, self.intrinsic_scatter, len(velocities))
        
        # Add galaxy type dependencies
        type_corrections = {'Spiral': 0.0, 'Irregular': 0.5, 'Dwarf': 1.0}
        for i, gtype in enumerate(properties['galaxy_type']):
            absolute_magnitudes[i] += type_corrections[gtype]
        
        # Convert to luminosities (Solar units)
        # Using M_sun = 4.83 in V-band
        M_sun = 4.83
        luminosities = 10**(-0.4 * (absolute_magnitudes - M_sun))
        
        # Calculate apparent magnitudes
        distance_moduli = 5 * np.log10(properties['distance'] * 1e6 / 10)
        apparent_magnitudes = absolute_magnitudes + distance_moduli
        
        properties.update({
            'luminosity': luminosities,
            'absolute_magnitude': absolute_magnitudes,
            'apparent_magnitude': apparent_magnitudes,
            'corrected_velocity': corrected_velocities
        })
        
        return properties
    
    def add_observational_effects(self, properties):
        """
        Add realistic observational uncertainties and selection effects
        
        Parameters:
        properties (dict): Galaxy properties
        
        Returns:
        dict: Properties with observational effects
        """
        n_galaxies = len(properties['velocity_width'])
        
        # Velocity measurement errors (km/s)
        velocity_errors = np.random.normal(0, 10, n_galaxies)  # ~10 km/s typical
        observed_velocities = properties['corrected_velocity'] + velocity_errors
        
        # Photometric errors (magnitudes) - more realistic
        # Error increases with distance and surface brightness
        base_phot_error = 0.02  # reduced base error
        distance_factor = np.sqrt(properties['distance'] / 30.0)  # normalized to 30 Mpc
        sb_factor = np.exp((properties['surface_brightness'] - 22.0) / 4.0)  # reduced factor
        
        photometric_errors = base_phot_error * distance_factor * sb_factor
        photometric_errors = np.clip(photometric_errors, 0.01, 0.3)  # reasonable range
        
        observed_magnitudes = (properties['apparent_magnitude'] + 
                             np.random.normal(0, photometric_errors))
        
        # Selection effects: brightness limit (more realistic for surveys)
        brightness_limit = 22.0  # apparent magnitude limit (more realistic)
        detectable = observed_magnitudes < brightness_limit
        
        properties.update({
            'observed_velocity': observed_velocities,
            'observed_magnitude': observed_magnitudes,
            'velocity_error': velocity_errors,
            'photometric_error': photometric_errors,
            'detectable': detectable
        })
        
        return properties
    
    def add_systematic_effects(self, properties):
        """
        Add systematic effects that could be discovered by AI
        
        Parameters:
        properties (dict): Galaxy properties
        
        Returns:
        dict: Properties with systematic effects
        """
        # Hidden systematic: TF relation depends on stellar mass
        mass_correction = 0.1 * np.log10(properties['stellar_mass'] / 1e10)
        
        # Hidden systematic: Non-linear velocity dependence
        velocity_correction = 0.05 * (np.log10(properties['corrected_velocity'] / 200))**2
        
        # Hidden systematic: Distance-dependent bias (Malmquist-like)
        distance_bias = 0.02 * np.log10(properties['distance'] / 50)
        
        # Apply corrections to absolute magnitude
        corrected_abs_mag = (properties['absolute_magnitude'] - 
                           mass_correction + velocity_correction + distance_bias)
        
        # Recalculate apparent magnitude
        distance_moduli = 5 * np.log10(properties['distance'] * 1e6 / 10)
        corrected_app_mag = corrected_abs_mag + distance_moduli
        
        properties.update({
            'true_absolute_magnitude': corrected_abs_mag,
            'mass_correction': mass_correction,
            'velocity_correction': velocity_correction,
            'distance_bias': distance_bias
        })
        
        return properties
    
    def generate_dataset(self, n_galaxies=1000, add_systematics=True):
        """
        Generate complete simulated dataset
        
        Parameters:
        n_galaxies (int): Number of galaxies to simulate
        add_systematics (bool): Whether to add hidden systematic effects
        
        Returns:
        pandas.DataFrame: Complete simulated dataset
        """
        # Generate base properties
        properties = self.generate_galaxy_properties(n_galaxies)
        
        # Apply TF relation
        properties = self.apply_tully_fisher_relation(properties)
        
        # Add systematic effects if requested
        if add_systematics:
            properties = self.add_systematic_effects(properties)
        
        # Add observational effects
        properties = self.add_observational_effects(properties)
        
        # Convert to DataFrame
        df = pd.DataFrame(properties)
        
        # Only keep detectable galaxies
        df = df[df['detectable']].reset_index(drop=True)
        
        return df
    
    def plot_dataset(self, df, figsize=(15, 10)):
        """
        Create comprehensive plots of the simulated dataset
        
        Parameters:
        df (pandas.DataFrame): Simulated dataset
        figsize (tuple): Figure size
        """
        # Check if dataset is empty
        if len(df) == 0:
            print("Warning: Dataset is empty. Cannot create plots.")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No detectable galaxies in dataset', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Empty Dataset')
            return fig
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Classic Tully-Fisher diagram
        ax = axes[0, 0]
        scatter = ax.scatter(np.log10(df['corrected_velocity']), 
                           df['absolute_magnitude'], 
                           c=df['distance'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('log(Velocity Width) [km/s]')
        ax.set_ylabel('Absolute Magnitude')
        ax.set_title('Tully-Fisher Relation')
        ax.invert_yaxis()
        plt.colorbar(scatter, ax=ax, label='Distance (Mpc)')
        
        # 2. Observed vs True relation
        ax = axes[0, 1]
        ax.scatter(np.log10(df['observed_velocity']), 
                  df['observed_magnitude'] - 5*np.log10(df['distance']*1e6/10), 
                  alpha=0.6, color='red', label='Observed')
        ax.scatter(np.log10(df['corrected_velocity']), 
                  df['absolute_magnitude'], 
                  alpha=0.6, color='blue', label='True')
        ax.set_xlabel('log(Velocity Width) [km/s]')
        ax.set_ylabel('Absolute Magnitude')
        ax.set_title('Observed vs True TF Relation')
        ax.legend()
        ax.invert_yaxis()
        
        # 3. Residuals vs stellar mass
        ax = axes[0, 2]
        # Calculate residuals from simple TF fit
        coeffs = np.polyfit(np.log10(df['corrected_velocity']), 
                          df['absolute_magnitude'], 1)
        predicted = np.polyval(coeffs, np.log10(df['corrected_velocity']))
        residuals = df['absolute_magnitude'] - predicted
        
        ax.scatter(np.log10(df['stellar_mass']), residuals, alpha=0.6)
        ax.set_xlabel('log(Stellar Mass) [M_sun]')
        ax.set_ylabel('TF Residuals')
        ax.set_title('Residuals vs Stellar Mass')
        ax.axhline(y=0, color='red', linestyle='--')
        
        # 4. Distance distribution
        ax = axes[1, 0]
        ax.hist(df['distance'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel('Count')
        ax.set_title('Distance Distribution')
        
        # 5. Velocity width distribution by galaxy type
        ax = axes[1, 1]
        for gtype in df['galaxy_type'].unique():
            subset = df[df['galaxy_type'] == gtype]
            ax.hist(np.log10(subset['velocity_width']), 
                   alpha=0.7, label=gtype, bins=20)
        ax.set_xlabel('log(Velocity Width) [km/s]')
        ax.set_ylabel('Count')
        ax.set_title('Velocity Distribution by Type')
        ax.legend()
        
        # 6. Observational errors
        ax = axes[1, 2]
        ax.scatter(df['distance'], df['photometric_error'], alpha=0.6)
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel('Photometric Error (mag)')
        ax.set_title('Observational Errors vs Distance')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = TullyFisherSimulator(seed=42)
    
    # Generate dataset
    print("Generating simulated Tully-Fisher dataset...")
    df = simulator.generate_dataset(n_galaxies=1000, add_systematics=True)
    
    print(f"Generated {len(df)} detectable galaxies")
    print(f"Dataset shape: {df.shape}")
    
    if len(df) == 0:
        print("\nWarning: No detectable galaxies were generated!")
        print("This might be due to:")
        print("- Brightness limit too restrictive")
        print("- Distance range too large")
        print("- Tully-Fisher parameters producing too faint galaxies")
        exit(1)
    
    print("\nDataset columns:")
    print(df.columns.tolist())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df[['velocity_width', 'distance', 'absolute_magnitude', 
             'stellar_mass', 'observed_velocity']].describe())
    
    # Create plots
    fig = simulator.plot_dataset(df)
    
    # Save dataset to input folder
    output_path = os.path.join('..', 'input', 'tully_fisher_simulated_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved as '{output_path}'")
    
    # Show correlations
    print("\nCorrelation matrix (key variables):")
    key_vars = ['velocity_width', 'absolute_magnitude', 'stellar_mass', 
               'distance', 'luminosity']
    print(df[key_vars].corr().round(3))
