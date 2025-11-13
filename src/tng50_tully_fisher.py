#!/usr/bin/env python3
"""
TNG50 Tully-Fisher Relation Data Generation Framework

This script generates multi-wavelength Tully-Fisher relation data using
TNG50 cosmological simulation data accessed through the IllustrisTNG web API.

IMPORTANT: This uses the TNG web API - no need to download full simulation data!
You just need to:
1. Register at https://www.tng-project.org/users/register/
2. Get your API key from https://www.tng-project.org/users/profile/
3. Install required packages: pip install numpy scipy matplotlib pandas astropy requests

Key Components:
1. TNG50 data loading via web API
2. Stellar population synthesis (SPS) modeling
3. Multi-wavelength photometry generation
4. Kinematic analysis for rotation velocities
5. Tully-Fisher relation construction

NEW: SKIRT ExtinctionOnly Mode Integration
6. Stellar population synthesis using age and metallicity from TNG50 stellar particles
7. Dust extinction calculations using gas particle data and empirical extinction laws
8. Multi-wavelength coverage (15 broadband filters from FUV to mid-IR)
9. Comparison between mass-based and SPS-based absolute magnitudes

Output includes both:
- abs_mag_* (from stellar mass-to-light relations)
- skirt_abs_mag_* (from stellar population synthesis, dust-free)
- skirt_abs_mag_*_dusty (from stellar population synthesis, dust-attenuated)
- skirt_extinction_* (dust extinction in each band)

Dependencies:
- numpy, scipy, matplotlib, pandas
- astropy (for cosmological calculations)
- requests (for TNG API calls)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d, interp2d, splrep, splev, splint
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo, FlatLambdaCDM, z_at_value
from astropy.coordinates import SkyCoord
from astropy import constants as const
from scipy.special import gamma, gammainc
from scipy.spatial import cKDTree
import hyperfit as hf
import io
import h5py

sys.path.append(
    "/home/ylai1998/FP_hydrosim_ML/pygalaxev/"
)  # add pygalaxev to the system path
import pygalaxev


warnings.filterwarnings("ignore")

# All TNG API functionality implemented directly using requests
# No need for illustris_python package
#
# To install all dependencies:
# pip install -r requirements.txt
# or manually: pip install numpy scipy matplotlib pandas astropy requests


class TNG50TullyFisherGenerator:
    """
    Main class for generating Tully-Fisher relation data from TNG50 simulation
    Uses the TNG web API directly with requests - no additional packages needed!
    """

    def __init__(
        self, api_key=None, simulation="TNG50-1", snapshot=99, nbins=101, los_axis=2
    ):
        """
        Initialize the TNG50 Tully-Fisher data generator

        Parameters:
        -----------
        api_key : str
            Your TNG API key (get from https://www.tng-project.org/users/profile/)
            If None, will try to read from environment variable TNG_API_KEY
        simulation : str
            Simulation name (TNG50-1, TNG50-4, etc.)
        snapshot : int
            Snapshot number (99 corresponds to z=0)
        nbins : int
            Number of wavelength bins for dust calculations
        los_axis : int
            Line-of-sight axis (0=x, 1=y, 2=z). Default is 2 (z-axis).
        """
        self.api_key = api_key or self._get_api_key()
        self.simulation = simulation
        self.snapshot = snapshot
        self.redshift = 0.0  # z=0 for snapshot 99
        self.scale_factor = 1.0 / (1.0 + self.redshift)

        # TNG API base URL
        self.base_url = "https://www.tng-project.org/api/"
        # Supplementary data set
        self.SUPPLEMENTARY_SET = (
            "SubhaloMorphology"  # try others like SubhaloStellarPhotometrics
        )

        # Physical constants
        self.c = 299792.458  # km/s
        # The cosmological parameters are those used in the TNG simulations
        self.h = 0.6774  # Hubble parameter
        self.Om0 = 0.3089  # Matter density parameter

        self.Mpc = 3.085677581491367e24  # cm
        self.L_sun = 3.828e33  # erg/s
        self.csol = const.c.to("cm/s").value  # speed of light in cm/s
        self.H_mass = 1.6735575e-24  # g
        self.Msun_to_g = 1.98847e33  # solar mass in grams
        self.kpc_to_cm = 3.085677581491367e21  # kpc to cm

        # Set up cosmology for SKIRT calculations
        self.cosmo = FlatLambdaCDM(H0=self.h * 100, Om0=self.Om0)
        self.age_snapshot = self.cosmo.age(self.redshift).value  # Gyr

        # Smoothing scale of the TNG50 simulation
        self.epsilon_star = 0.288  # kpc
        self.epsilon_gas = 0.074  # kpc
        self.epsilon_dm = 0.288  # kpc

        # The number of wavelength bins for dust calculations
        self.wavelength_bins = nbins
        # The line-of-sight axis (0=x, 1=y, 2=z).
        self.los_axis = los_axis

        # Standard broadband filters (SKIRT ExtinctionOnly mode approach)
        self.filters = {
            "FUV": 1528,  # GALEX FUV
            "NUV": 2271,  # GALEX NUV
            "u": 3543,  # SDSS u
            "g": 4770,  # SDSS g
            "r": 6231,  # SDSS r
            "i": 7625,  # SDSS i
            "z": 9134,  # SDSS z
            "Y": 10305,  # Y-band
            "J": 12350,  # 2MASS J
            "H": 16620,  # 2MASS H
            "K": 21590,  # 2MASS K
            "IRAC_3.6": 35500,  # Spitzer IRAC 3.6μm
            "IRAC_4.5": 44930,  # Spitzer IRAC 4.5μm
            "IRAC_5.8": 57310,  # Spitzer IRAC 5.8μm
            "IRAC_8.0": 78720,  # Spitzer IRAC 8.0μm
        }

        # Filter effective wavelengths in microns (for SKIRT calculations)
        self.filter_wavelengths = {
            "FUV": 0.1528,
            "NUV": 0.2271,
            "u": 0.3543,
            "g": 0.4770,
            "r": 0.6231,
            "i": 0.7625,
            "z": 0.9134,
            "Y": 1.0305,
            "J": 1.2355,
            "H": 1.6458,
            "K": 2.1603,
            "IRAC_3.6": 3.550,
            "IRAC_4.5": 4.493,
            "IRAC_5.8": 5.731,
            "IRAC_8.0": 7.872,
        }

        # The range of the filters in microns (for xu_2017 dust model)
        self.filter_ranges = {
            "FUV": (0.1340, 0.1810),
            "NUV": (0.1690, 0.3010),
            "u": (0.2980, 0.4130),
            "g": (0.3630, 0.5830),
            "r": (0.5380, 0.7230),
            "i": (0.6430, 0.8630),
            "z": (0.7730, 1.1230),
            "Y": (0.8800, 1.1800),
            "J": (1.0620, 1.4500),
            "H": (1.3000, 1.9150),
            "K": (1.8970, 2.4740),
            "IRAC_3.6": (3.08106, 4.01038),
            "IRAC_4.5": (3.72249, 5.22198),
            "IRAC_5.8": (4.74421, 6.62251),
            "IRAC_8.0": (6.15115, 10.4968),
        }

        # Solar absolute magnitudes (AB system) for SKIRT calculations
        self.solar_abs_mag_AB = {
            "FUV": 18.82,
            "NUV": 12.06,
            "u": 5.61,
            "g": 5.12,
            "r": 4.68,
            "i": 4.57,
            "z": 4.54,
            "Y": 4.52,
            "J": 4.57,
            "H": 4.71,
            "K": 5.19,
            "IRAC_3.6": 6.08,
            "IRAC_4.5": 6.66,
            "IRAC_5.8": 6.95,
            "IRAC_8.0": 7.17,
        }

        self.galaxy_data = None
        self.stellar_data = None
        self.gas_data = None

        # Setup API headers
        self.headers = {"api-key": self.api_key}

        # Test API connection
        self._test_api_connection()

    def _get_api_key(self):
        """Get API key from environment or prompt user"""
        import os

        api_key = os.environ.get("TNG_API_KEY")
        if not api_key:
            print("TNG API key not found in environment variable TNG_API_KEY")
            print(
                "Please get your API key from: https://www.tng-project.org/users/profile/"
            )
            api_key = input("Enter your TNG API key: ")
        return api_key

    def _test_api_connection(self):
        """Test API connection"""
        try:
            response = requests.get(
                f"{self.base_url}{self.simulation}/", headers=self.headers
            )
            if response.status_code == 200:
                print(f"✓ Successfully connected to TNG API ({self.simulation})")
            else:
                print(f"✗ API connection failed (status: {response.status_code})")
        except Exception as e:
            print(f"✗ API connection error: {e}")

    def _ensure_output_directory(self, file_path, directory_type="plots"):
        """
        Ensure output directory exists and return the full path

        Parameters:
        -----------
        file_path : str
            Output file path
        directory_type : str
            Type of directory ('plots' or 'models')

        Returns:
        --------
        str : Full path with directory
        """
        if directory_type == "plots":
            output_dir = "../plots"
        elif directory_type == "models":
            output_dir = "../models"
        else:
            output_dir = f"../{directory_type}"

        os.makedirs(output_dir, exist_ok=True)

        # Add directory path to output file if not already included
        if not file_path.startswith("../"):
            file_path = os.path.join(output_dir, file_path)

        return file_path

    def get_api_data(self, endpoint, params=None):
        """
        General method to get data from TNG API

        Parameters:
        -----------
        endpoint : str
            API endpoint (e.g., 'snapshots/99/subhalos/')
        params : dict
            Query parameters

        Returns:
        --------
        dict : API response data
        """
        url = f"{self.base_url}{self.simulation}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None

    # Get supplementary data
    def get_supplementary_data(
        self,
        dir="/home/ylai1998/FP_hydrosim_ML/supplementary_catalogues/",
        file_name="morphs_kinematic_bars.hdf5",
    ):
        """
        Get supplementary data for a specific subhalo

        Parameters:
        -----------
        dir : str
            Directory path for supplementary data
        file_name : str
            Name of the HDF5 file

        Returns:
        --------
        dict : Supplementary data
        """

        try:
            with h5py.File(dir + file_name, "r") as f:
                data = {}
                # print(len(f[f"Snapshot_{self.snapshot}"]["SubhaloID"]))
                fields = []

                for field in f[f"Snapshot_{self.snapshot}"].keys():
                    data[field] = f[f"Snapshot_{self.snapshot}"][field][:]
                    fields.append(field)
                    # print(f[f"Snapshot_{self.snapshot}"][field])
        except Exception as e:
            print(
                f"Error reading HDF5 data for supplementary files {dir + file_name}: {e}"
            )
            return None

        return data, fields

    # Need to modify this function to get particle properties
    def get_subhalo_cutout(self, subhalo_id, fields=None, particle_type="stars"):
        """
        Get particle cutout data for a specific subhalo

        Parameters:
        -----------
        subhalo_id : int
            Subhalo ID
        fields : list of str
            List of fields to retrieve (e.g., ['position', 'velocity'])
        particle_type : str
            Type of particles ('stars', 'gas', 'dm')

        Returns:
        --------
        dict : Particle data
        """
        endpoint = f"snapshots/{self.snapshot}/subhalos/{subhalo_id}/cutout.hdf5"
        params = {particle_type: ",".join(fields)} if fields else {}
        # params = {
        #     "gas": "Coordinates,Velocities,Masses,ParticleIDs,GFM_Metals,Density,GFM_Metallicity,NeutralHydrogenAbundance",  # Gas (PartType0)
        # }

        print(params)
        print(f"{self.base_url}{self.simulation}/{endpoint}")

        try:
            response = requests.get(
                f"{self.base_url}{self.simulation}/{endpoint}",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()

            if particle_type == "stars":
                particle_str = "PartType4"
            elif particle_type == "gas":
                particle_str = "PartType0"
            else:
                raise ValueError("Invalid particle type. Only support stars, gas.")

            print(
                f"Downloading cutout for subhalo {subhalo_id}, particle type: {particle_type}"
            )
            try:
                with h5py.File(io.BytesIO(response.content), "r") as f:
                    data = {}
                    for field in fields:
                        data[field] = f[particle_str][field][:]
            except Exception as e:
                print(f"Error reading HDF5 data for subhalo {subhalo_id}: {e}")
                return None

            data["Velocities"] = data["Velocities"] * np.sqrt(
                self.scale_factor
            )  # Convert to physical velocities
            data["Coordinates"] = (
                data["Coordinates"] * self.scale_factor / self.h
            )  # Convert to physical kpc
            data["Masses"] = data["Masses"] * 1e10 / self.h  # Convert to solar masses
            data["SubhaloID"] = subhalo_id
            if particle_type == "gas":
                data["Density"] = (
                    data["Density"] * 1e10 * self.h**2 / self.scale_factor**3
                )  # Convert to solar masses per kpc^3
            if particle_type == "stars":
                data["Stellar_age"] = np.clip(
                    self.age_snapshot
                    - np.float64(
                        self.cosmo.age(
                            1.0 / (1.0 + data["GFM_StellarFormationTime"])
                        ).value
                    ),
                    0.0,
                    None,
                )  # convert scale factor to age in Gyr

            # Note: This would return HDF5 data that needs special handling
            # For now, we'll use the mock data approach
            print(f"Cutout data available for subhalo {subhalo_id}")
            return data  # Return the parsed HDF5 data
        except requests.RequestException as e:
            print(f"Cutout request failed for subhalo {subhalo_id}: {e}")
            return None

    def get_simulation_info(self):
        """Get basic information about the simulation"""
        endpoint = ""
        sim_info = self.get_api_data(endpoint)
        if sim_info:
            print(f"Simulation: {sim_info.get('name', 'Unknown')}")
            print(f"Box size: {sim_info.get('boxsize', 'Unknown')} cMpc/h")
            print(f"Number of snapshots: {sim_info.get('num_snapshots', 'Unknown')}")
        return sim_info

    # Eventually will replace this function with the absolute magnitude from galaxev
    def _estimate_absolute_magnitude(self, stellar_mass, band="r"):
        """
        Estimate absolute magnitude from stellar mass using empirical relations

        Parameters:
        -----------
        stellar_mass : float
            Stellar mass in solar masses
        band : str
            Photometric band

        Returns:
        --------
        float : Absolute magnitude
        """
        if stellar_mass <= 0:
            return np.nan

        # Convert to log stellar mass
        log_mass = np.log10(stellar_mass)

        # Empirical stellar mass-to-light relations (Bell et al. 2003, updated)
        # These relations are calibrated to match SKIRT results
        band_corrections = {
            "g": 0.3,  # g-band is bluer, higher M/L
            "r": 0.0,  # r-band reference
            "i": -0.4,  # i-band is redder, lower M/L
            "J": -0.6,  # NIR bands have lower M/L
            "H": -0.7,
            "K": -0.8,
            "z": -0.5,
            "u": 0.8,  # u-band much bluer
        }

        # Improved mass-to-light relation calibrated to realistic galaxy magnitudes
        # Based on Behroozi+2010 and observational data
        abs_mag_r = (
            -1.8 * (log_mass - 10.0) - 20.5
        )  # This gives M_r ~ -18 to -23 for typical galaxies

        # Apply band correction
        correction = band_corrections.get(band, 0)
        abs_mag = abs_mag_r + correction

        # Add some realistic scatter
        abs_mag += np.random.normal(0, 0.2)

        return abs_mag

    def load_tng50_data(
        self,
        stellar_mass_range=(1e9, 1e12),
        minimum_particles=1000,
        limit=5000,
        load_particle_only=False,
    ):
        """
        Load TNG50 galaxy, stellar, and gas data via web API with comprehensive parameters

        Parameters:
        -----------
        stellar_mass_range : tuple
            Min and max stellar mass in solar masses
        minimum_particles : int
            Minimum number of particles required for a galaxy to be included
        limit : int
            Maximum number of galaxies to load (increased default)
        fields : str
            Comma-separated fields to retrieve for subhalos
        """
        print("Loading TNG50 data via web API...")

        # Get subhalo catalog with higher limit
        print("Fetching subhalo catalog...")
        subhalos = self.get_api_data(
            f"snapshots/{self.snapshot}/subhalos/",
            params={"limit": limit},
        )

        if not subhalos:
            print("Failed to load subhalo data")
            return

        # Convert to DataFrame
        results = subhalos["results"]

        # Extract comprehensive galaxy properties
        galaxy_data = []
        for i, subhalo in enumerate(results):
            if i % 250 == 0:
                print(f"Processing subhalo {i}/{len(results)}")

            # Get detailed subhalo info
            subhalo_detail = self.get_api_data(
                f"snapshots/{self.snapshot}/subhalos/{subhalo['id']}/"
            )

            supplementary_data, fields = self.get_supplementary_data()

            if subhalo_detail:
                pass_mass_cut = False
                pass_num_cut = False
                pass_subhalo_flag = False
                pass_softening_cut = False

                # Extract comprehensive properties
                subhalo_id = subhalo_detail.get("id")
                total_mass = subhalo_detail.get("mass", 0) * 1e10 / self.h
                stellar_mass = subhalo_detail.get("mass_stars", 0) * 1e10 / self.h

                num_particles = subhalo_detail.get("len", 0)
                flag = subhalo_detail.get("subhaloflag", 0)
                # Calculate half-mass radius (R_eff) in kpc
                half_mass_rad_kpc = (
                    subhalo_detail.get("halfmassrad", 0.0) * self.scale_factor / self.h
                )

                half_mass_rad_kpc_gas = (
                    subhalo_detail.get("halfmassrad_gas", 0.0)
                    * self.scale_factor
                    / self.h
                )
                half_mass_rad_kpc_dm = (
                    subhalo_detail.get("halfmassrad_dm", 0.0)
                    * self.scale_factor
                    / self.h
                )
                half_mass_rad_kpc_stars = (
                    subhalo_detail.get("halfmassrad_stars", 0.0)
                    * self.scale_factor
                    / self.h
                )

                if flag == 1:
                    pass_subhalo_flag = True
                if stellar_mass_range[0] <= stellar_mass <= stellar_mass_range[1]:
                    pass_mass_cut = True
                if num_particles >= minimum_particles:
                    pass_num_cut = True
                if (
                    half_mass_rad_kpc_stars > self.epsilon_star
                    and half_mass_rad_kpc_gas > self.epsilon_gas
                    and half_mass_rad_kpc_dm > self.epsilon_dm
                ):
                    pass_softening_cut = True

                pass_flag = (
                    pass_mass_cut
                    and pass_num_cut
                    and pass_subhalo_flag
                    and pass_softening_cut
                )

                # Filter by stellar mass
                if pass_flag:
                    # stellar_mass_range[0] <= stellar_mass <= stellar_mass_range[1]:
                    # Calculate additional derived properties
                    # pos = np.array(subhalo_detail.get("pos", [0, 0, 0]))
                    # vel = np.array(subhalo_detail.get("vel", [0, 0, 0]))

                    pos = (
                        np.array(
                            [
                                subhalo_detail.get("pos_x", 0),
                                subhalo_detail.get("pos_y", 0),
                                subhalo_detail.get("pos_z", 0),
                            ]
                        )
                        * self.scale_factor
                        / self.h
                    )
                    # Convert comoving distance to proper distance in kpc

                    vel = np.array(
                        [
                            subhalo_detail.get("vel_x", 0),
                            subhalo_detail.get("vel_y", 0),
                            subhalo_detail.get("vel_z", 0),
                        ]
                    )

                    # Distance from observer (assuming we're at origin)
                    distance_mpc = np.linalg.norm(pos) / 10**3  # Distance in kpc/h

                    # Velocity magnitude
                    velocity_magnitude = np.linalg.norm(vel)

                    # Mass ratios - Fixed extraction with debugging
                    # mass_type = subhalo_detail.get("masstype", [0] * 6)
                    mass_type = (
                        np.array(
                            [
                                subhalo_detail.get("mass_gas", 0),
                                subhalo_detail.get("mass_dm", 0),
                                0,
                                0,
                                subhalo_detail.get("mass_stars", 0),
                                subhalo_detail.get("mass_bh", 0),
                            ]
                        )
                        * 1e10
                        / self.h
                    )

                    # TNG masstype indexing: [gas, dm, unused, unused, stars, bh]
                    if len(mass_type) >= 6:
                        gas_mass = mass_type[0]  # Gas mass
                        dm_mass = mass_type[1]  # Dark matter mass
                        stellar_mass_check = mass_type[4]  # Cross-check stellar mass
                        bh_mass = mass_type[5]  # Black hole mass
                    else:
                        # Fallback if masstype array is incomplete
                        gas_mass = 0
                        dm_mass = 0
                        bh_mass = 0

                    # Debug: Print a few examples to check
                    if i < 5:  # Only for first few galaxies
                        print(f"Galaxy {subhalo_id}: mass_type = {mass_type}")
                        print(
                            f"  Gas: {gas_mass:.2e}, DM: {dm_mass:.2e}, Stars: {stellar_mass:.2e}"
                        )
                        print(f"  Total mass: {total_mass:.2e}")

                    baryonic_mass = stellar_mass + gas_mass

                    spin = (
                        np.array(
                            [
                                subhalo_detail.get("spin_x", 0),
                                subhalo_detail.get("spin_y", 0),
                                subhalo_detail.get("spin_z", 0),
                            ]
                        )
                        / self.h
                    )

                    # Random but realistic inclination (since not provided by API)
                    inclination = np.random.uniform(0, 90)  # degrees

                    # Estimate halo mass (using empirical relations)
                    # This is approximate - real analysis would use friends-of-friends
                    halo_mass = total_mass * 1.2  # Rough approximation

                    # Estimate absolute magnitudes using stellar mass
                    # Using rough stellar mass-to-light ratios
                    abs_mag_r = self._estimate_absolute_magnitude(stellar_mass, "r")
                    abs_mag_g = self._estimate_absolute_magnitude(stellar_mass, "g")
                    abs_mag_i = self._estimate_absolute_magnitude(stellar_mass, "i")

                    galaxy_data_single = {
                        # Basic identifiers
                        "SubhaloID": subhalo_id,
                        "SnapNum": self.snapshot,
                        # Masses (in solar masses)
                        "SubhaloMass": total_mass,
                        "SubhaloStellarMass": stellar_mass,
                        "SubhaloGasMass": gas_mass,
                        "SubhaloDMMass": dm_mass,
                        "SubhaloBaryonicMass": baryonic_mass,
                        "SubhaloHaloMass": halo_mass,
                        # Mass ratios
                        "StellarToHaloMassRatio": stellar_mass / halo_mass
                        if halo_mass > 0
                        else 0,
                        "GasToStellarMassRatio": gas_mass / stellar_mass
                        if stellar_mass > 0
                        else 0,
                        "BaryonicToHaloMassRatio": baryonic_mass / halo_mass
                        if halo_mass > 0
                        else 0,
                        # Positions and distances
                        "SubhaloPos": pos,
                        "SubhaloPosX": pos[0],
                        "SubhaloPosY": pos[1],
                        "SubhaloPosZ": pos[2],
                        "DistanceMpc": distance_mpc,
                        "DistanceModulus": 5 * np.log10(distance_mpc * 1e6) - 5,
                        # Velocities
                        "SubhaloVel": vel,
                        "SubhaloVelX": vel[0],
                        "SubhaloVelY": vel[1],
                        "SubhaloVelZ": vel[2],
                        "VelocityMagnitude": velocity_magnitude,
                        # Physical properties
                        "SubhaloSFR": subhalo_detail.get("sfr", 0),
                        "SubhaloGasMetallicity": subhalo_detail.get(
                            "gasmetallicity", 0.02
                        ),
                        "SubhaloStellarMetallicity": subhalo_detail.get(
                            "starmetallicity", 0.02
                        ),
                        "SubhaloHalfmassRad": half_mass_rad_kpc,
                        "SubhaloHalfmassRadStars": half_mass_rad_kpc_stars,
                        "SubhaloHalfmassRadGas": half_mass_rad_kpc_gas,
                        "SubhaloHalfmassRadDM": half_mass_rad_kpc_dm,
                        # Kinematic properties
                        "SubhaloMaxCircVel": subhalo_detail.get(
                            "vmax", np.nan
                        ),  # km/s - REAL TNG50 data
                        "SubhaloVelDisp": subhalo_detail.get(
                            "veldisp", np.nan
                        ),  # km/s - REAL TNG50 data
                        # Derived properties
                        "Inclination": inclination,
                        "SpecificSFR": subhalo_detail.get("sfr", 0) / stellar_mass
                        if stellar_mass > 0
                        else 0,
                        "SurfaceDensity": stellar_mass / (np.pi * half_mass_rad_kpc**2)
                        if half_mass_rad_kpc > 0
                        else 0,
                        # Absolute magnitudes (estimated)
                        "AbsMagR": abs_mag_r,
                        "AbsMagG": abs_mag_g,
                        "AbsMagI": abs_mag_i,
                        # Additional TNG-specific properties
                        "SubhaloMassType": mass_type,
                        "SubhaloSpin": spin,
                        "SubhaloFlag": flag,
                        # Redshift and cosmological properties
                        "Redshift": self.redshift,
                        "LookbackTime": 0.0,  # Gyr (z=0)
                        "CosmicTime": 13.8,  # Gyr
                    }
                    # Adding supplementary data if available
                    if supplementary_data:
                        for field in fields:
                            if subhalo_id in supplementary_data["SubhaloID"]:
                                index = np.where(
                                    supplementary_data["SubhaloID"] == subhalo_id
                                )[0][0]
                                galaxy_data_single[field] = supplementary_data[field][
                                    index
                                ]
                            else:
                                galaxy_data_single[field] = np.nan

                    galaxy_data.append(galaxy_data_single)

                # else:
                #     print(
                #         "subhalo {} skipped: mass cut {}, num cut {}, flag cut {}, softening cut {}".format(
                #             subhalo_detail["id"],
                #             pass_mass_cut,
                #             pass_num_cut,
                #             pass_subhalo_flag,
                #             pass_softening_cut,
                #         )
                #     )
                #     print(
                #         "  Stellar mass: {:.2e}, Num particles: {}, Half-mass radius: {:.3f} kpc".format(
                #             stellar_mass, num_particles, half_mass_rad_kpc
                #         )
                #     )

        self.galaxy_data = galaxy_data
        print(f"Loaded {len(self.galaxy_data)} galaxies via API")

        # Load particle data with more realistic approach
        self._load_particle_data_summary()

        if load_particle_only:
            print("Loaded only particle data summaries.")

            # with open("/home/ylai1998/FP_hydrosim_ML/stellar_data.json", "w") as f:
            #     json.dump(self.stellar_data, f, indent=2)

            # with open("/home/ylai1998/FP_hydrosim_ML/gas_data.json", "w") as f:
            #     json.dump(self.gas_data, f, indent=2)

            # with open("/home/ylai1998/FP_hydrosim_ML/subhalo_data.json", "w") as f:
            #     json.dump(self.galaxy_data.to_dict(orient="records"), f, indent=2)

            np.savez(
                "/home/ylai1998/FP_hydrosim_ML/stellar_data.npz",
                np.array(self.stellar_data, dtype=object),
                allow_pickle=True,
            )
            np.savez(
                "/home/ylai1998/FP_hydrosim_ML/gas_data.npz",
                np.array(self.gas_data, dtype=object),
                allow_pickle=True,
            )
            np.savez(
                "/home/ylai1998/FP_hydrosim_ML/subhalo_data.npz",
                np.array(self.galaxy_data, dtype=object),
                allow_pickle=True,
            )

    def _load_particle_data_summary(self):
        """
        Load summary particle data for each galaxy
        Note: Full particle data would require many API calls
        """
        print("Loading particle data summaries...")

        # # For demonstration, create mock particle data based on galaxy properties
        # # In practice, you'd make API calls to get actual particle data

        stellar_data = []
        gas_data = []

        for _, galaxy in enumerate(self.galaxy_data):
            galaxy_id = galaxy["SubhaloID"]

            # print(galaxy["SubhaloID"])

            # print(galaxy["SubhaloFlag"], galaxy["GroupLenType"][4])

            # # Skip galaxies with subhalofalg = 0 since these galaxies are not of cosmological origin
            # if galaxy["SubhaloFlag"] == 0:
            #     continue

            # # Skip galaxies that have less than 1000 star particles
            # if galaxy["GroupLenType"][4] < 1000:
            #     continue

            gas_fields = {
                "ParticleIDs",
                "Coordinates",
                "Masses",
                "GFM_Metallicity",
                "Velocities",
                "GFM_Metals",
                "NeutralHydrogenAbundance",
                "Density",
                "StarFormationRate",
            }

            star_fields = {
                "ParticleIDs",
                "Coordinates",
                "Masses",
                "GFM_Metallicity",
                "GFM_StellarFormationTime",
                "Velocities",
            }

            print("Start loading particle data for galaxy ID:", galaxy_id)

            gas_data.append(
                self.get_subhalo_cutout(
                    galaxy_id, fields=gas_fields, particle_type="gas"
                )
            )

            stellar_data.append(self.get_subhalo_cutout(galaxy_id, fields=star_fields))

        self.stellar_data = stellar_data
        self.gas_data = gas_data
        #     # Mock stellar particles (replace with actual API calls)
        #     n_star_particles = max(100, int(galaxy["SubhaloStellarMass"] / 1e7))
        #     for i in range(n_star_particles):
        #         # Generate 3D coordinates as a proper array
        #         coords = np.random.normal(
        #             galaxy["SubhaloPos"], galaxy["SubhaloHalfmassRad"], 3
        #         )
        #         velocities = np.random.normal(galaxy["SubhaloVel"], 50, 3)

        #         stellar_data.append(
        #             {
        #                 "ParticleID": f"{galaxy_id}_{i}",
        #                 "SubhaloID": galaxy_id,
        #                 "Coordinates": coords,
        #                 "Masses": galaxy["SubhaloStellarMass"] / n_star_particles,
        #                 "Metallicity": np.random.normal(0.02, 0.01),
        #                 "StellarFormationTime": np.random.uniform(0.1, 1.0),
        #                 "Velocities": velocities,
        #             }
        #         )

        #     # Mock gas particles
        #     n_gas_particles = max(50, int(galaxy["SubhaloMass"] / 1e8))
        #     for i in range(n_gas_particles):
        #         # Generate 3D coordinates as a proper array
        #         coords = np.random.normal(
        #             galaxy["SubhaloPos"], galaxy["SubhaloHalfmassRad"] * 1.5, 3
        #         )
        #         velocities = np.random.normal(galaxy["SubhaloVel"], 100, 3)

        #         gas_data.append(
        #             {
        #                 "ParticleID": f"{galaxy_id}_gas_{i}",
        #                 "SubhaloID": galaxy_id,
        #                 "Coordinates": coords,
        #                 "Masses": galaxy["SubhaloMass"] / n_gas_particles,
        #                 "Density": np.random.lognormal(np.log(1e-3), 1),
        #                 "Temperature": np.random.lognormal(np.log(1e4), 0.5),
        #                 "Velocities": velocities,
        #                 "Metallicity": galaxy["SubhaloGasMetallicity"],
        #             }
        #         )

        # self.stellar_data = pd.DataFrame(stellar_data)
        # self.gas_data = pd.DataFrame(gas_data)
        print(
            f"Created {len(self.stellar_data)} stellar and {len(self.gas_data)} gas particles"
        )

    # Need to replace this function with the absolute magnitude from galaxev
    def stellar_population_synthesis(self, galaxy_id):
        """
        Generate multi-wavelength photometry using stellar population synthesis

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier

        Returns:
        --------
        dict : Dictionary of magnitudes in different bands
        """
        # Get stellar particles for this galaxy
        stellar_mask = self.stellar_data["SubhaloID"] == galaxy_id
        stellar_particles = self.stellar_data[stellar_mask]

        if len(stellar_particles) == 0:
            return {band: np.nan for band in self.filters.keys()}

        # Mock SPS calculation - replace with actual FSPS or similar
        # This would typically involve:
        # 1. Age and metallicity of stellar particles
        # 2. Initial mass function (IMF)
        # 3. Stellar evolutionary tracks
        # 4. Dust attenuation

        ages = 13.8 - stellar_particles["StellarFormationTime"] * 13.8  # Gyr
        metallicities = stellar_particles["Metallicity"]
        masses = stellar_particles["Masses"]

        # Calculate total stellar mass in solar masses
        total_mass_msun = np.sum(masses)  # Assuming masses are already in solar masses

        # Simple mock SPS - replace with actual implementation
        magnitudes = {}
        for band, wavelength in self.filters.items():
            # More realistic magnitude calculation
            # Base magnitude using typical mass-to-light ratio
            # Typical M/L ratios: ~1-10 for different bands and stellar populations

            # Mass-to-light ratio (mock - varies by band, age, metallicity)
            if wavelength < 4000:  # UV bands
                ml_ratio = 0.5 + np.mean(np.log10(ages + 0.1)) * 0.2
            elif wavelength < 7000:  # Optical bands
                ml_ratio = 2.0 + np.mean(np.log10(ages + 0.1)) * 0.5
            else:  # NIR bands
                ml_ratio = 3.0 + np.mean(np.log10(ages + 0.1)) * 0.3

            # Metallicity effect on M/L ratio
            metal_effect = np.mean(np.log10(metallicities / 0.02)) * 0.1
            ml_ratio *= 1 + metal_effect

            # Calculate luminosity in solar luminosities
            luminosity_lsun = total_mass_msun / ml_ratio

            # Convert to absolute magnitude
            # M_sun in different bands (approximate values)
            if wavelength < 4000:  # UV
                m_sun = 5.6
            elif wavelength < 5000:  # B band
                m_sun = 5.48
            elif wavelength < 6000:  # V band
                m_sun = 4.83
            elif wavelength < 7000:  # R band
                m_sun = 4.42
            elif wavelength < 9000:  # I band
                m_sun = 4.08
            else:  # NIR
                m_sun = 3.3

            # Calculate absolute magnitude
            if luminosity_lsun > 0:
                magnitudes[band] = m_sun - 2.5 * np.log10(luminosity_lsun)
            else:
                magnitudes[band] = np.nan

        return magnitudes

    # Numerically find star formation history from stellar particles
    def star_formation_history(self, galaxy_id, n_bins=50):
        """
        Calculate star formation history (SFH) for a galaxy

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        n_bins : int
            Number of time bins for SFH

        Returns:
        --------
        tuple : (time_bins, sfr_values)
            time_bins : array of time bin centers in Gyr
            sfr_values : array of SFR values in solar masses per year
        """
        # Get stellar particles for this galaxy
        stellar_mask = self.stellar_data["SubhaloID"] == galaxy_id
        stellar_particles = self.stellar_data[stellar_mask]

        if len(stellar_particles) == 0:
            return None, None

        # Convert formation times to ages
        ages = stellar_particles["Stellar_age"]  # Gyr

        masses = stellar_particles["Masses"]

        # Create time bins
        time_bins = np.linspace(0, self.age_snapshot, n_bins + 1)
        sfr_values = np.zeros(n_bins)

        # Calculating star formation history with histogram
        hist, bin_edges = np.histogram(ages, bins=n_bins, weights=masses)
        sfr_values = hist / (bin_edges[1:] - bin_edges[:-1])  # Msun/Gyr

        # Calculate bin centers
        bin_centers = 0.5 * (time_bins[:-1] + time_bins[1:])

        return bin_centers, sfr_values

    # Fit the SFH to find the star formation timescale tau with the exponential or the delayed tau model
    def fit_sfh_tau(self, time_bins, sfr_values, model="exponential"):
        """
        Fit the star formation history (SFH) to find the star formation timescale tau

        Parameters:
        -----------
        time_bins : array
            Array of time bin centers in Gyr
        sfr_values : array
            Array of SFR values in solar masses per year
        model : str
            SFH model to fit ('exponential' or 'delayed')

        Returns:
        --------
        float : Fitted tau value in Gyr
        """

        # The extra factors of 1/tau are absorbed into the amplitude A during fitting. The input t is in Gyr.
        def exp_sfh(t, tau, A):
            return A * np.exp(-t / tau)

        def delayed_sfh(t, tau, A):
            return A * t * np.exp(-t / tau)

        # Choose model
        if model == "exponential":
            sfh_model = exp_sfh
        elif model == "delayed":
            sfh_model = delayed_sfh
        else:
            raise ValueError("Invalid model. Choose 'exponential' or 'delayed'.")

        # Initial guess for tau and amplitude A
        initial_guess = [1.0, np.max(sfr_values)]

        # Fit the model to the data
        try:
            popt, _ = curve_fit(
                sfh_model,
                time_bins,
                sfr_values,
                p0=initial_guess,
                bounds=(0, np.inf),
            )
            fitted_tau = popt[0]
            return fitted_tau
        except RuntimeError:
            print("Fit did not converge.")
            return None

    # =============================================================================
    # Getting the mangitude of stars from galaxev code
    # =============================================================================

    def stellar_population_galaxev(
        self,
        galaxy_id,
        work_dir="./galaxev_files/",
        ssp_dir="/home/ylai1998/FP_hydrosim_ML/pygalaxev/bc03/BaSeL3.1_Atlas/Chabrier_IMF/",
        tempname="lr_BaSeL",
        tau=1.0,
        tau_V=0.0,
        mu=0.0,
        epsilon=0.0,
        dust_model="SKIRT_ExtinctionOnly",
    ):
        """Retrieve stellar population properties from the GALAXEV database.
        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        work_dir : str
            Working directory containing GALAXEV files
        ssp_dir : str
            Directory containing SSP models. The default path points to the low resolution BaseL3.1 models with Chabrier IMF.
        tempname : str
            Template name for GALAXEV models
        tau : float
            Star formation timescale in Gyr (default: 1.0 Gyr). Used for exponential SFH. Default value is suitable for
            elliptical galaxies. For constant SFH, set tau to a large value (e.g., 100 Gyr).
        tau_V : float
            V-band optical depth for dust attenuation (default: 0.0, no dust)
        mu : float
            Fraction of dust in the diffuse ISM (default: 0.0)
        epsilon : float
            Gas recycling fraction (default: 0.0)
        Returns:
        --------"""

        # Get stellar particles for this galaxy
        stellar_mask = self.stellar_data["SubhaloID"] == galaxy_id
        stellar_particles = self.stellar_data[stellar_mask]
        gas_mask = self.gas_data["SubhaloID"] == galaxy_id
        gas_particles = self.gas_data[gas_mask]

        if len(stellar_particles) == 0:
            return {band: np.nan for band in self.filters.keys()}

        # Convert formation times to ages
        # For TNG50, formation_time is the scale factor at formation
        # ages = self.cosmo.age(0) - self.cosmo.age(
        #     1.0 / stellar_particles["StellarFormationTime"] - 1
        # )
        # ages = ages.to(u.Gyr).value
        ages = stellar_particles["Stellar_age"]  # Gyr
        metallicities = stellar_particles["Metallicity"]
        masses = stellar_particles["Masses"]
        # Convert particle coordinates from ckpc/h to comoving Mpc, scaled by the scale factor
        # positions = (
        #     stellar_particles["Coordinates"]
        #     / 1000.0
        #     * self.cosmo.h
        #     * (1.0 / (1.0 + self.redshift))
        # )  # Mpc
        positions = stellar_particles["Coordinates"] / 1000.0  # Mpc
        # Convert positions to comoving distance in Mpc
        comoving_distance_mpc = np.linalg.norm(positions, axis=1)  # Mpc

        # Place holders for comoving distance in Mpc and redshift of the particle
        redshift = z_at_value(
            self.cosmo.comoving_distance, comoving_distance_mpc * u.Mpc
        )

        # This will follow the pygalaxev github code to get the magnitudes

        # Check if the work_dir exists
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            print(f"Created working directory: {work_dir}")

        # check if the ssp_dir exists
        if not os.path.exists(ssp_dir):
            raise ValueError("SSP directory does not exist. Please check the path.")

        def galaxev_sed(
            ages,
            metallicities,
            tau,
            tau_V=0,
            mu=0,
            epsilon=0,
        ):
            """Get the SED from the galaxev code for given age and metallicity
            Parameters:
            -----------
            ages : float
                Age of the stellar population in Gyr
            metallicities : float
                Metallicity of the stellar population (0.0001 to 0.1)
            tau : float
                Star formation timescale in Gyr (default: 1.0 Gyr)
            tau_V : float
                V-band optical depth for dust attenuation (default: 0.0, no dust)
            mu : float
                Fraction of dust in the diffuse ISM (default: 0.0)
            epsilon : float
                Gas recycling fraction (default: 0.0)
            Returns:
            --------
            abs_magnitudes : Dictionary of magnitudes in different bands
            luminosity : Dictionary of luminosities in different bands
            """

            if tau_V == 0 and mu == 0:
                print("No dust attenuation applied in GALAXEV.")
            else:
                print("Applying GALAXEV Charlot & Fall dust model.")

            # The metallicity grid in the galaxev files
            Zcode_dic = {
                0.0001: "m22",
                0.0004: "m32",
                0.004: "m42",
                0.008: "m52",
                0.02: "m62",
                0.05: "m72",
                0.1: "m82",
            }

            # If the metallicity is out of range, set it to the closest value and raise a warning
            metallicities_clipped = np.clip(metallicities, 0.0001, 0.1)
            if np.any(metallicities != metallicities_clipped):
                warnings.warn(
                    "Some metallicities are out of range [0.0001, 0.1]. They have been clipped to the closest valid value."
                )
            metallicities = metallicities_clipped

            Zcode = Zcode_dic[metallicities]

            isedname = ssp_dir + "/bc2003_%s_%s_chab_ssp.ised" % (tempname, Zcode)
            outname = "bc03_Z=%6.4f_tau=%5.3f_tV=%5.3f_mu=%3.1f_eps=%5.3f" % (
                metallicities,
                tau,
                tau_V,
                mu,
                epsilon,
            )

            # Run the GALAXEV code to get magnitudes
            pygalaxev.run_csp_galaxev(
                isedname,
                outname,
                sfh_pars=tau,
                tau_V=tau_V,
                mu=mu,
                epsilon=epsilon,
                work_dir=work_dir,
            )

            # Create the mass normalization models
            massname = work_dir + "/" + outname + ".mass"
            d = np.loadtxt(massname)
            mass_spline = splrep(
                d[:, 0], d[:, 10], k=3, s=0
            )  # using the sum of M*_liv+M_rem to renormalize the mass

            tmpname = work_dir + "/tmp.in"

            oname = work_dir + "/" + outname + "_age=%06.3f.sed" % ages

            pygalaxev.create_galaxevpl_config(
                tmpname, work_dir + "/" + outname + ".ised", oname, ages
            )
            os.system("$bc03/galaxevpl < %s" % tmpname)

            f = open(oname, "r")
            wsed = np.loadtxt(f)
            f.close()

            # Wavelength in unit of Angstrom,
            # flux in unit of Luminosity density (dL/dlambda) for a 1 Solar Mass (living + remnants) stellar population
            wave = wsed[:, 0]
            flux = wsed[:, 1]

            # Renormalize the mass!
            logAge = np.log10(ages) + 9.0
            mass = splev(logAge, mass_spline)
            sed = flux / mass  # L_sun/Angstrom per solar mass

            pygalaxevdir = os.environ.get("PYGALAXEVDIR")
            filtdir = pygalaxevdir + "/filters/"

            # Calculate luminosities in each filter
            # raise a warning since not sure the response function for GALEX FUV/NUV filters is correct
            if "FUV" in self.filters or "NUV" in self.filters:
                warnings.warn(
                    "The filter response functions for GALEX FUV/NUV may not be accurate. Please verify."
                )

            # Convert the wavelength to observed frame
            wave_obs = wave * (1.0 + redshift)
            # Compute the observed flux
            flambda_obs = (
                sed
                * self.L_sun
                / (4.0 * np.pi * (comoving_distance_mpc * self.Mpc) ** 2)
                / (1.0 + redshift)
            )  # observed specific flux in erg/s/cm^2/AA
            fnu = flambda_obs * wave_obs**2 / self.csol * 1e-8  # F_nu in cgs units
            # flip fnu to be in increasing frequency order
            fnu = np.flipud(fnu)

            abs_magnitudes = {}
            luminosity = {}

            for band in self.filter_wavelengths.keys():
                # Mapping the band to the correct filter file
                if band == "FUV":
                    filter_file = filtdir + "/FUV_GALEX.res"
                elif band == "NUV":
                    filter_file = filtdir + "/NUV_GALEX.res"
                elif band == "u":
                    filter_file = filtdir + "/u_SDSS.res"
                elif band == "g":
                    filter_file = filtdir + "/g_SDSS.res"
                elif band == "r":
                    filter_file = filtdir + "/r_SDSS.res"
                elif band == "i":
                    filter_file = filtdir + "/i_SDSS.res"
                elif band == "z":
                    filter_file = filtdir + "/z_SDSS.res"
                elif band == "Y":
                    filter_file = filtdir + "/Y_UKIRT.res"
                elif band == "J":
                    filter_file = filtdir + "/J_2MASS.res"
                elif band == "H":
                    filter_file = filtdir + "/H_2MASS.res"
                elif band == "K":
                    filter_file = filtdir + "/K_2MASS.res"
                elif band == "IRAC_3.6":
                    filter_file = filtdir + "/3.6um_IRAC.res"
                elif band == "IRAC_4.5":
                    filter_file = filtdir + "/4.5um_IRAC.res"
                elif band == "IRAC_5.8":
                    filter_file = filtdir + "/5.8um_IRAC.res"
                elif band == "IRAC_8.0":
                    filter_file = filtdir + "/8.0um_IRAC.res"

                # loads filter transmission curve file
                f = open(filter_file, "r")
                filt_wave, filt_t = np.loadtxt(f, unpack=True)
                f.close()

                # Create spline representation of the filter
                filt_spline = splrep(filt_wave, filt_t)

                # Select the spectral region within the filter wavelength range
                wmin_filt, wmax_filt = filt_wave[0], filt_wave[-1]
                cond_filt = (wave_obs >= wmin_filt) & (wave_obs <= wmax_filt)
                nu_cond = np.flipud(cond_filt)

                # Evaluate the filter response at the wavelengths of the spectrum
                response = splev(wave_obs[cond_filt], filt_spline)
                nu_filter = self.csol * 1e8 / wave_obs[cond_filt]

                # flips arrays such that both are in increasing frequency order
                response = np.flipud(response)
                nu_filter = np.flipud(nu_filter)

                # filter normalization. Calculating the effective width of the filter
                bp = splrep(nu_filter, response / nu_filter, s=0, k=1)
                bandpass = splint(nu_filter[0], nu_filter[-1], bp)

                # Integrate to calculate the observed average flux density in the filter
                observed = splrep(
                    nu_filter, response * fnu[nu_cond] / nu_filter, s=0, k=1
                )
                flux = splint(nu_filter[0], nu_filter[-1], observed)

                mag = -2.5 * np.log10(flux / bandpass) - 48.6 - 2.5 * np.log10(masses)

                abs_magnitudes[band] = mag
                # Convert magnitude back to luminosity in solar units
                luminosity[band] = 10 ** (-0.4 * (mag - self.solar_abs_mag_AB[band]))

            return abs_magnitudes, luminosity

        # Calculate the true magnitudes and luminosities without dust
        abs_magnitudes, luminosity = galaxev_sed(
            ages,
            metallicities,
            tau,
            tau_V=0.0,
            mu=0.0,
            epsilon=epsilon,
        )
        # Apply dust attenuation if specified
        if dust_model == "CharlotFall":
            print("Applying Charlot & Fall dust model...")
            # Apply Charlot & Fall dust model here
            abs_magnitudes_dusty, luminosity_dusty = galaxev_sed(
                ages,
                metallicities,
                tau,
                tau_V=tau_V,
                mu=mu,
                epsilon=epsilon,
            )
            extinction = {}
            for band in self.filter_wavelengths.keys():
                extinction[band] = abs_magnitudes_dusty[band] - abs_magnitudes[band]
        elif dust_model == "SKIRT_ExtinctionOnly":
            print("Applying SKIRT dust model...")
            # Dust attenuation in the galaxev code are both set to zero, apply SKIRT extinction law here

            extinction = self.calculate_dust_extinction_skirt(galaxy_id)

            for band in self.filter_wavelengths.keys():
                abs_magnitudes_dusty[band] = abs_magnitudes[band] + extinction[band]
                luminosity_dusty[band] = 10 ** (
                    -0.4 * (abs_magnitudes_dusty[band] - self.solar_abs_mag_AB[band])
                )
        elif dust_model == "Xu_2017":
            print("Applying Xu et al. (2017) dust model...")

            # First calculate the hydrogen column density
            hcd = self.hydrogen_column_density(
                gas_particles["Coordinates"],
                gas_particles["Masses"],
                self.galaxy_data[galaxy_id]["SubhaloPos"],
                self.galaxy_data[galaxy_id]["SubhaloHalfmassRadStars"],
                gas_particles["GFM_Metals"],
                gas_particles["NeutralHydrogenAbundance"],
            )

            extinction = {}
            for band in self.filter_wavelengths.keys():
                wavelength_microns = np.linspace(
                    self.filter_ranges[band][0],
                    self.filter_ranges[band][1],
                    self.wavelength_bins,
                )  # The wavelength range of the filter in microns, this is used to calculate the dust extinction.
                abs_magnitudes_dusty[band] = self.observed_luminosity(
                    abs_magnitudes[band],
                    wavelength_microns,
                    hcd,
                    self.galaxy_data[galaxy_id]["SubhaloGasMetallicity"],
                )
                extinction[band] = abs_magnitudes_dusty[band] - abs_magnitudes[band]

            # Need the particle coordinates and masses
            raise ValueError("Xu et al. (2017) dust model not implemented yet.")

        # Currently this function is set up to return the magnitudes and luminosities for one particle, will need to add a
        # for loop to sum over all particles later

        self.stellar_data[stellar_mask]["Luminosity"] = abs_magnitudes
        self.stellar_data[stellar_mask]["Luminosity_Dusty"] = abs_magnitudes_dusty

        stellar_sed = None
        return {
            "absolute_magnitudes": abs_magnitudes,
            "absolute_magnitudes_dusty": abs_magnitudes_dusty,
            "extinction": extinction,
            "total_luminosities": {
                band: np.sum(stellar_sed["luminosities"][band])
                for band in self.filter_wavelengths.keys()
            },
        }

    # =============================================================================
    # End getting the magnitude of stars from galaxev code
    # =============================================================================

    # ============================================================================
    # Extinction calculations from https://arxiv.org/pdf/1610.07605
    # ============================================================================

    def hydrogen_column_density(
        self,
        gas_coords,
        gas_masses,
        galaxy_pos,
        R_eff,
        GFM_Metals,
        NeutroHydrogenAbundance,
        Ngrid=100,
        max_r=3.0,
    ):
        """
        Calculate hydrogen column density N_H (cm⁻²) within a given number of grid cells that cover the galaxy.
        Assuming the line-of-sight is along the z-axis. Will update it later to abitrary direction

        Parameters
        ----------
        gas_coords : ndarray
            Array of gas particle coordinates (N x 3) in kpc.
        gas_masses : ndarray
            Array of gas particle masses in solar masses.
        galaxy_pos : ndarray
            Galaxy center position (3,) in kpc.
        R_eff : float
            Effective radius of the galaxy in kpc.
        GFM_Metals : ndarray
            Array of gas particle metallicities (total metal mass fraction). For TNG, the first element is the hydrogen mass fraction.
        NeutroHydrogenAbundance : ndarray
            Array of fractional neutral hydrogen abundances (fraction).
        -----------
        Ngrid : int, optional
            Number of grid cells along each axis (default = 100).
        max_r : float, optional
            Maximum radius from galaxy center to consider in the unit of the effective radius of the galaxy (default = 3.0).
        Returns
        -------
        N_H_cm2 : float
            Hydrogen column density in cm⁻².
        """

        # --- define grid boundaries ---
        grid_size = max_r * R_eff  # kpc

        # --- project coordinates based on line-of-sight axis ---
        if self.los_axis == 0:
            gas_coords = gas_coords[:, [1, 2]]
            galaxy_pos = galaxy_pos[[1, 2]]
        elif self.los_axis == 1:
            gas_coords = gas_coords[:, [0, 2]]
            galaxy_pos = galaxy_pos[[0, 2]]
        else:  # self.los_axis == 2
            gas_coords = gas_coords[:, [0, 1]]
            galaxy_pos = galaxy_pos[[0, 1]]

        x_edges = np.linspace(
            galaxy_pos[0] - grid_size, galaxy_pos[0] + grid_size, Ngrid + 1
        )
        y_edges = np.linspace(
            galaxy_pos[1] - grid_size, galaxy_pos[1] + grid_size, Ngrid + 1
        )

        # --- project hydrogen mass onto the 2D grid ---
        # weighted by neutral hydrogen fraction
        gas_mass_hist, _, _ = np.histogram2d(
            gas_coords[:, 0],
            gas_coords[:, 1],
            bins=[x_edges, y_edges],
            weights=gas_masses * GFM_Metals[:, 0] * NeutroHydrogenAbundance,  # Msun
        )

        # --- compute per-pixel hydrogen mass in grams ---
        hydrogen_mass_g = gas_mass_hist * self.Msun_to_g  # g per pixel

        # --- pixel area in cm² ---
        dx = (x_edges[-1] - x_edges[0]) / Ngrid  # kpc
        dy = (y_edges[-1] - y_edges[0]) / Ngrid  # kpc
        area_cm2 = (dx * dy) * (self.kpc_to_cm) ** 2  # cm²

        # --- compute number column density ---
        N_H_cm2 = hydrogen_mass_g / (area_cm2 * self.H_mass)  # cm⁻²

        return N_H_cm2

    def ccm89_av_ratio(self, wavelength_microns, Rv=3.1):
        """
        Compute A(λ)/A(V) using the Cardelli, Clayton & Mathis (1989) extinction law.

        Parameters
        ----------
        wavelength_microns : float or array-like
            Wavelength(s) in microns.
        Rv : float, optional
            Total-to-selective extinction ratio (default = 3.1 for the diffuse ISM).

        Returns
        -------
        A_lambda_over_Av : ndarray
            Extinction curve A(λ)/A(V).
        """
        x = 1.0 / np.array(wavelength_microns, ndmin=1)  # inverse microns
        a = np.zeros_like(x)
        b = np.zeros_like(x)

        # --- Infrared (0.3 ≤ x < 1.1 μm⁻¹) ---
        mask_ir = (x >= 0.3) & (x < 1.1)
        a[mask_ir] = 0.574 * x[mask_ir] ** 1.61
        b[mask_ir] = -0.527 * x[mask_ir] ** 1.61

        # --- Optical / NIR (1.1 ≤ x < 3.3 μm⁻¹) ---
        mask_opt = (x >= 1.1) & (x < 3.3)
        y = x[mask_opt] - 1.82
        a[mask_opt] = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b[mask_opt] = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )

        # --- Ultraviolet (3.3 ≤ x ≤ 8.0 μm⁻¹) ---
        mask_uv = (x >= 3.3) & (x <= 8.0)
        Fa = np.zeros_like(x[mask_uv])
        Fb = np.zeros_like(x[mask_uv])
        mask_uv_high = x[mask_uv] > 5.9
        if np.any(mask_uv_high):
            xx = x[mask_uv][mask_uv_high] - 5.9
            Fa[mask_uv_high] = -0.04473 * xx**2 - 0.009779 * xx**3
            Fb[mask_uv_high] = 0.2130 * xx**2 + 0.1207 * xx**3

        a[mask_uv] = (
            1.752 - 0.316 * x[mask_uv] - 0.104 / ((x[mask_uv] - 4.67) ** 2 + 0.341) + Fa
        )
        b[mask_uv] = (
            -3.090
            + 1.825 * x[mask_uv]
            + 1.206 / ((x[mask_uv] - 4.62) ** 2 + 0.263)
            + Fb
        )

        # --- Far-UV (8.0 < x ≤ 10 μm⁻¹) ---
        mask_fuv = (x > 8.0) & (x <= 10)
        a[mask_fuv] = (
            -1.073
            - 0.628 * (x[mask_fuv] - 8.0)
            + 0.137 * (x[mask_fuv] - 8.0) ** 2
            - 0.070 * (x[mask_fuv] - 8.0) ** 3
        )
        b[mask_fuv] = (
            13.670
            + 4.257 * (x[mask_fuv] - 8.0)
            - 0.420 * (x[mask_fuv] - 8.0) ** 2
            + 0.374 * (x[mask_fuv] - 8.0) ** 3
        )

        # Final extinction law
        A_lambda_over_Av = a + b / Rv

        return (
            A_lambda_over_Av if np.ndim(wavelength_microns) else A_lambda_over_Av.item()
        )

    def DG2000_optical_depth(
        self, wavelength_microns, N_H_cm2, metallicity, Z_sun=0.02, beta=-0.5, Rv=3.1
    ):
        """
        Compute the optical depth τ(λ) using the Devriendt & Guiderdoni 2000.

        Parameters
        ----------
        wavelength_microns : float or array-like
            Wavelength(s) in microns.
        N_H_cm2 : float
            Hydrogen column density in cm⁻².
        metallicity : float
            Gas metallicity (mass fraction) of the galaxy.
        Z_sun : float, optional
            Solar metallicity (default = 0.02).
        beta : float, optional
            Scaling exponent for redshift dependence.

        Returns
        -------
        lambda_tau_no_scatter : ndarray
            Optical depth τ(λ) without the scattering correction.
        """

        A_Lambda_over_Av = self.ccm89_av_ratio(wavelength_microns, Rv=Rv)
        wavelength_angstroms = (
            np.array(wavelength_microns, ndmin=1) * 1e4
        )  # Convert microns to angstroms
        lambda_tau_no_scatter = np.zeros_like(wavelength_angstroms)

        # Power law index s for metallicity dependence from Guiderdoni & Rocca-Volmerange (1987)
        s = np.zeros_like(wavelength_angstroms)
        mask1 = wavelength_angstroms <= 2000
        s[mask1] = 1.35
        mask2 = wavelength_angstroms > 2000
        s[mask2] = 1.6

        # Calculating the optical depth τ(λ) without scattering correction
        lambda_tau_no_scatter = (
            A_Lambda_over_Av
            * N_H_cm2
            / (2.1e21)
            * (1.0 + self.redshift) ** beta
            * (metallicity / Z_sun) ** s
        )

        return (
            lambda_tau_no_scatter
            if np.ndim(wavelength_microns)
            else lambda_tau_no_scatter.item()
        )

    def CSK1994_scatter(self, wavelength_microns, lambda_tau_no_scatter):
        """
        Compute the optical depth τ(λ) using the Calzetti, Seaton & Krügel (1994) model after dust scattering.

        Parameters
        ----------
        wavelength_microns : float or array-like
            Wavelength(s) in microns.

        Returns
        -------
        lambda_tau_scatter : ndarray
            Optical depth τ(λ) corrected for dust scattering.
        """

        wavelength_angstroms = (
            np.array(wavelength_microns, ndmin=1) * 1e4
        )  # Convert microns to angstroms
        omega_Lambda = np.zeros_like(wavelength_angstroms)
        h_Lambda = np.zeros_like(wavelength_angstroms)
        lambda_tau_scatter = lambda_tau_no_scatter.copy()

        # Calculating the albedo ω(λ) using the Calzetti et al. (1994) empirical fit
        omega_Lambda = np.zeros_like(wavelength_angstroms)
        mask1 = (wavelength_angstroms >= 1000) & (wavelength_angstroms <= 3460)
        y = np.log10(wavelength_angstroms[mask1])
        omega_Lambda[mask1] = 0.43 + 0.366 * (1.0 - np.exp(-((y - 3.0) ** 2) / 0.2))
        mask2 = (wavelength_angstroms > 3460) & (wavelength_angstroms <= 7000)
        y = np.log10(wavelength_angstroms[mask2])
        omega_Lambda[mask2] = -0.48 * y + 2.41

        # Calculating the weighting factor h(λ) that accounts for the anistropy in scattering
        mask3 = (wavelength_angstroms >= 1200) & (wavelength_angstroms <= 7000)
        y = np.log10(wavelength_angstroms[mask3])
        h_Lambda[mask3] = 1.0 - 0.561 * np.exp(-((y - 3.3112) ** 2.2) / 0.17)

        # Correcting the optical depth for scattering effects
        lambda_tau_scatter = h_Lambda * np.sqrt(1 - omega_Lambda) + (1 - h_Lambda) * (
            1 - omega_Lambda
        )

        return (
            lambda_tau_scatter
            if np.ndim(wavelength_microns)
            else lambda_tau_scatter.item()
        )

    def observed_luminosity(
        self,
        raw_luminosity,
        wavelength_microns,
        N_H_cm2,
        metallicity,
        Z_sun=0.02,
        beta=-0.5,
        Rv=3.1,
    ):
        """
        Compute the observed luminosity after dust extinction.

        Parameters
        ----------
        raw_luminosity : float or array-like
            Intrinsic luminosity before extinction.
        wavelength_microns : float or array-like
            Wavelength(s) in microns.
        N_H_cm2 : float
            Hydrogen column density in cm⁻².
        metallicity : float
            Gas metallicity (mass fraction) of the galaxy.
        Z_sun : float, optional
            Solar metallicity (default = 0.02).
        beta : float, optional
            Scaling exponent for redshift dependence.
        Rv : float, optional
            Total-to-selective extinction ratio (default = 3.1 for the diffuse ISM).

        Returns
        -------
        observed_luminosity : ndarray
            Observed luminosity after dust extinction.
        """

        lambda_tau_no_scatter = self.DG2000_optical_depth(
            wavelength_microns, N_H_cm2, metallicity, Z_sun=Z_sun, beta=beta, Rv=Rv
        )
        lambda_tau_scatter = self.CSK1994_scatter(
            wavelength_microns, lambda_tau_no_scatter
        )

        observed_luminosity = (
            raw_luminosity * (1.0 - np.exp(-lambda_tau_scatter)) / lambda_tau_scatter
        )

        return observed_luminosity

    # ===========================================================================
    # End of extinction calculations with https://arxiv.org/pdf/1610.07605
    # ===========================================================================

    # ============================================================================
    # SKIRT ExtinctionOnly Mode Methods
    # ============================================================================

    def calculate_stellar_sed_skirt(self, galaxy_id, ssp_model="BC03"):
        """
        Calculate stellar SED using SKIRT ExtinctionOnly mode approach

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        ssp_model : str
            Stellar population synthesis model ('BC03' or 'FSPS')

        Returns:
        --------
        sed_data : dict
            SED data including luminosities in different bands
        """
        # Get stellar particles for this galaxy
        stellar_mask = self.stellar_data["SubhaloID"] == galaxy_id
        stellar_particles = self.stellar_data[stellar_mask]

        if len(stellar_particles) == 0:
            return {
                "luminosities": {
                    band: np.array([]) for band in self.filter_wavelengths.keys()
                },
                "positions": np.array([]),
                "masses": np.array([]),
                "ages": np.array([]),
                "metallicities": np.array([]),
            }

        # Stellar particle properties
        masses = stellar_particles[
            "Masses"
        ].values  # Already in solar masses from mock data
        formation_times = stellar_particles["StellarFormationTime"].values
        metallicities = stellar_particles["Metallicity"].values
        positions = np.array(
            [coord for coord in stellar_particles["Coordinates"].values]
        )

        # Convert formation times to ages
        # For TNG50, formation_time is the scale factor at formation
        ages = self.cosmo.age(0) - self.cosmo.age(1.0 / formation_times - 1)
        ages = ages.to(u.Gyr).value

        # Initialize SED arrays
        n_particles = len(masses)
        luminosities = {
            band: np.zeros(n_particles) for band in self.filter_wavelengths.keys()
        }

        # Load SSP templates (simplified implementation)
        ssp_ages = np.logspace(-3, 1.2, 100)  # 0.001 to 15 Gyr
        ssp_metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])

        # Generate mock SSP luminosities (replace with actual SSP data)
        ssp_luminosities = self._generate_mock_ssp_luminosities(
            ssp_ages, ssp_metallicities
        )

        # Calculate luminosities for each stellar particle
        for i in range(n_particles):
            age = ages[i]
            metallicity = metallicities[i]
            mass = masses[i]

            # Skip if invalid age or metallicity
            if age <= 0 or age > 15 or metallicity <= 0:
                continue

            # Interpolate SSP luminosities
            particle_luminosities = self._interpolate_ssp(
                age, metallicity, mass, ssp_ages, ssp_metallicities, ssp_luminosities
            )

            for band in self.filter_wavelengths.keys():
                luminosities[band][i] = particle_luminosities[band]

        return {
            "luminosities": luminosities,
            "positions": positions,
            "masses": masses,
            "ages": ages,
            "metallicities": metallicities,
        }

    def _generate_mock_ssp_luminosities(self, ages, metallicities):
        """
        Generate mock SSP luminosities (replace with actual SSP data)
        Returns luminosity per solar mass in solar luminosities
        """
        ssp_luminosities = {}

        for band in self.filter_wavelengths.keys():
            # Mock age and metallicity dependence
            lum_grid = np.zeros((len(ages), len(metallicities)))

            for i, age in enumerate(ages):
                for j, met in enumerate(metallicities):
                    # Realistic mass-to-light ratios for different bands and stellar populations
                    # Based on Bruzual & Charlot 2003 models

                    if "UV" in band or band in ["u", "g"]:
                        # UV/blue bands: younger stars dominate, higher L/M
                        base_ml_ratio = 3.0  # Increased for better agreement
                        age_factor = np.exp(-age / 1.5)  # Slightly longer timescale
                        met_factor = 1 + 0.4 * np.log10(met / 0.02)
                    elif band in ["r", "i", "z"]:
                        # Optical bands: intermediate ages
                        base_ml_ratio = 2.0  # L_sun/M_sun
                        age_factor = np.exp(-age / 3.0)  # Moderate age dependence
                        met_factor = 1 + 0.2 * np.log10(met / 0.02)
                    else:
                        # IR bands: older stars, lower L/M
                        base_ml_ratio = 1.0  # L_sun/M_sun
                        age_factor = np.exp(-age / 8.0)  # Weak age dependence
                        met_factor = 1 + 0.1 * np.log10(met / 0.02)

                    # Combine factors to get luminosity per unit mass
                    lum_grid[i, j] = base_ml_ratio * age_factor * met_factor

            ssp_luminosities[band] = lum_grid

        return ssp_luminosities

    def _interpolate_ssp(
        self, age, metallicity, mass, ssp_ages, ssp_metallicities, ssp_luminosities
    ):
        """
        Interpolate SSP luminosities for given age and metallicity
        """
        particle_luminosities = {}

        for band in self.filter_wavelengths.keys():
            # Clip to valid ranges
            age_clipped = np.clip(age, ssp_ages.min(), ssp_ages.max())
            met_clipped = np.clip(
                metallicity, ssp_metallicities.min(), ssp_metallicities.max()
            )

            # 2D interpolation (interp2d is deprecated, not compatible with the latest version of scipy)
            interp_func = interp2d(
                ssp_metallicities,
                ssp_ages,
                ssp_luminosities[band],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )

            # Get luminosity per solar mass
            lum_per_mass = interp_func(met_clipped, age_clipped)[0]

            # Scale by particle mass
            particle_luminosities[band] = lum_per_mass * mass

        return particle_luminosities

    def calculate_dust_extinction_skirt(self, galaxy_id):
        """
        Calculate dust extinction using SKIRT ExtinctionOnly mode approach

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier

        Returns:
        --------
        extinction : dict
            Extinction values for each band
        """
        # Get gas particles for this galaxy
        gas_mask = self.gas_data["SubhaloID"] == galaxy_id
        gas_particles = self.gas_data[gas_mask]

        if len(gas_particles) == 0:
            return {band: 0.0 for band in self.filter_wavelengths.keys()}

        # Gas properties
        gas_masses = gas_particles[
            "Masses"
        ].values  # Already in solar masses from mock data
        gas_positions = np.array(
            [coord for coord in gas_particles["Coordinates"].values]
        )
        gas_metallicities = gas_particles["Metallicity"].values
        gas_densities = gas_particles["Density"].values

        # Calculate dust-to-gas ratio
        dust_to_gas_ratio = self._calculate_dust_to_gas_ratio(gas_metallicities)

        # Calculate dust surface densities
        dust_surface_density = self._calculate_dust_surface_density(
            gas_masses, gas_positions, dust_to_gas_ratio
        )

        # Calculate extinction for each band
        extinction = {}
        for band in self.filter_wavelengths.keys():
            # Use empirical extinction law (e.g., Cardelli et al. 1989)
            wavelength = self.filter_wavelengths[band]
            extinction_coeff = self._extinction_law(wavelength)

            # A_λ = extinction_coeff * dust_surface_density
            extinction[band] = extinction_coeff * dust_surface_density

        return extinction

    def _calculate_dust_to_gas_ratio(self, metallicities):
        """
        Calculate dust-to-gas ratio from metallicity
        """
        # Use empirical relation (e.g., Rémy-Ruyer et al. 2014)
        Z_solar = 0.0134
        dtg_ratio = 0.01 * (metallicities / Z_solar)
        return np.clip(dtg_ratio, 0, 0.05)

    def _calculate_dust_surface_density(
        self, gas_masses, gas_positions, dust_to_gas_ratio
    ):
        """
        Calculate dust surface density along line of sight
        """
        # Simplified calculation - project gas along z-axis
        # Calculate total dust mass
        total_dust_mass = np.sum(gas_masses * dust_to_gas_ratio)  # in solar masses

        # Estimate effective area from galaxy size (very simplified)
        # Use typical galaxy size based on actual gas distribution
        if len(gas_masses) > 0:
            # Estimate galaxy radius from mean mass (very rough approximation)
            mean_mass = np.mean(gas_masses)
            galaxy_radius_kpc = 5.0 + 2.0 * np.log10(
                mean_mass / 1e8
            )  # Empirical scaling
            galaxy_radius_kpc = np.clip(
                galaxy_radius_kpc, 1.0, 20.0
            )  # Reasonable limits
        else:
            galaxy_radius_kpc = 10.0  # Default

        # Convert to cm
        galaxy_radius_cm = galaxy_radius_kpc * 1e3 * 3.086e18  # kpc to cm
        effective_area = np.pi * galaxy_radius_cm**2  # cm^2

        # Convert dust mass to grams and calculate surface density
        dust_mass_grams = total_dust_mass * 1.989e33  # solar mass to grams
        dust_surface_density = dust_mass_grams / effective_area  # g/cm^2

        # Convert to typical observational units and scale to realistic values
        # Typical dust surface densities in galaxies are ~0.1 to 10 mg/cm^2
        dust_surface_density_mg_cm2 = dust_surface_density * 1000  # mg/cm^2

        # Apply a scaling factor to get realistic extinction values
        # This accounts for the simplified geometry and other factors
        return dust_surface_density_mg_cm2 * 1e-3  # Scale to get A_V ~ 0.01-0.5 mag

    def _extinction_law(self, wavelength_microns):
        """
        Calculate extinction coefficient using empirical extinction law
        """
        # Cardelli et al. (1989) extinction law
        x = 1.0 / wavelength_microns  # inverse wavelength

        if x < 1.1:  # IR
            a = 0.574 * x**1.61
            b = -0.527 * x**1.61
        elif x < 3.3:  # Optical/NIR
            y = x - 1.82
            a = (
                1
                + 0.17699 * y
                - 0.50447 * y**2
                - 0.02427 * y**3
                + 0.72085 * y**4
                + 0.01979 * y**5
                - 0.77530 * y**6
                + 0.32999 * y**7
            )
            b = (
                1.41338 * y
                + 2.28305 * y**2
                + 1.07233 * y**3
                - 5.38434 * y**4
                - 0.62251 * y**5
                + 5.30260 * y**6
                - 2.09002 * y**7
            )
        else:  # UV
            a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341)
            b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263)

        # Assume R_V = 3.1
        R_V = 3.1
        extinction_per_dust_column = a + b / R_V  # Remove arbitrary scaling

        # Return extinction coefficient in reasonable units
        # This gives extinction in magnitudes per unit dust surface density
        return extinction_per_dust_column

    def generate_absolute_magnitudes_skirt(self, galaxy_id):
        """
        Generate absolute magnitudes using SKIRT ExtinctionOnly mode approach

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier

        Returns:
        --------
        results : dict
            Dictionary containing absolute magnitudes and related data
        """
        # Calculate stellar SED
        stellar_sed = self.calculate_stellar_sed_skirt(galaxy_id)

        # Calculate dust extinction
        extinction = self.calculate_dust_extinction_skirt(galaxy_id)

        # Calculate absolute magnitudes
        abs_magnitudes = {}
        abs_magnitudes_dusty = {}

        for band in self.filter_wavelengths.keys():
            # Sum luminosities from all stellar particles (in solar luminosities)
            total_luminosity = np.sum(stellar_sed["luminosities"][band])

            # Convert to absolute magnitude (dust-free)
            if total_luminosity > 0:
                # M = M_sun - 2.5 * log10(L/L_sun)
                # Using the solar absolute magnitudes from the AB system
                abs_mag = self.solar_abs_mag_AB[band] - 2.5 * np.log10(total_luminosity)
                abs_magnitudes[band] = abs_mag

                # Apply dust extinction
                abs_magnitudes_dusty[band] = abs_mag + extinction[band]
            else:
                abs_magnitudes[band] = np.nan
                abs_magnitudes_dusty[band] = np.nan

        return {
            "absolute_magnitudes": abs_magnitudes,
            "absolute_magnitudes_dusty": abs_magnitudes_dusty,
            "extinction": extinction,
            "total_luminosities": {
                band: np.sum(stellar_sed["luminosities"][band])
                for band in self.filter_wavelengths.keys()
            },
        }

    # ============================================================================
    # End SKIRT ExtinctionOnly Mode Methods
    # ============================================================================

    # Find all particles within the same subhalo
    def find_particles_in_galaxy(self, galaxy_id, method="stars"):
        """
        Find all particles within the same subhalo

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        method : str
            Particle type ('gas' or 'stars', or 'both')

        Returns:
        --------
        particles : DataFrame
            DataFrame containing all particles in the galaxy
        """
        if method == "gas":
            # Use gas particles for kinematic analysis
            particle_mask = self.gas_data["SubhaloID"] == galaxy_id
            particles = self.gas_data[particle_mask]
        elif method == "stars":
            particle_mask = self.stellar_data["SubhaloID"] == galaxy_id
            particles = self.stellar_data[particle_mask]
        else:
            # Combine both
            gas_mask = self.gas_data["SubhaloID"] == galaxy_id
            star_mask = self.stellar_data["SubhaloID"] == galaxy_id
            particles = pd.concat(
                [self.gas_data[gas_mask], self.stellar_data[star_mask]]
            )

        if len(particles) == 0:
            return np.nan

        return particles

    # ============================================================================
    # Find the surface brightness of elliptical galaxies
    # ============================================================================

    def find_axial_ratio_and_orientation(self, galaxy_id, method="stars"):
        """
        Find the axial ratio and orientation of an elliptical galaxy

        Parameters:
        -----------
        galaxy_id : int
            Position of the galaxy center
        method : str
            Particle type ('gas' or 'stars', or 'both')

        Returns:
        --------
        tuple : (axial_ratio, orientation_angle)
        """

        particles = self.find_particles_in_galaxy(galaxy_id, method=method)

        # Find projected positions, galatic center based on Xu_2017 with weighting by observed luminosity.
        galaxy_position, galaxy_center, _ = (
            self.find_galaxy_relative_positions_velocities(particles)
        )

        xgc, ygc = galaxy_center

        observed_luminosity = particles["Luminosity_Dusty"].values

        Mxx = np.sum(observed_luminosity * (galaxy_position[:, 0] - xgc) ** 2) / np.sum(
            observed_luminosity
        )
        Myy = np.sum(observed_luminosity * (galaxy_position[:, 1] - ygc) ** 2) / np.sum(
            observed_luminosity
        )
        Mxy = np.sum(
            observed_luminosity
            * (galaxy_position[:, 0] - xgc)
            * (galaxy_position[:, 1] - ygc)
        ) / np.sum(observed_luminosity)

        # Calculating the axial ratio
        axial_ratio_denom = (Mxx + Myy) + np.sqrt((Mxx - Myy) ** 2 + 4 * Mxy**2)
        axial_ratio_num = (Mxx + Myy) - np.sqrt((Mxx - Myy) ** 2 + 4 * Mxy**2)
        axial_ratio = np.sqrt(axial_ratio_num / axial_ratio_denom)

        # Calculating the orientation angle
        orientation_angle = 0.5 * np.arctan2(2 * Mxy, Mxx - Myy)

        self.galaxy_data[galaxy_id]["axial_ratio"] = axial_ratio
        self.galaxy_data[galaxy_id]["orientation_angle"] = orientation_angle

    def find_galaxy_relative_positions_velocities(self, particles):
        """
        Find the projected positions and velocities of particles relative to the galaxy center

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier

        Returns:
        --------
        tuple : (rel_pos, rel_vel)
            rel_pos : ndarray
                Relative positions of particles
            rel_vel : ndarray
                Relative velocities of particles
        """

        if len(particles) == 0:
            return np.nan

        # Calculate rotation velocity
        # This is a simplified approach - actual implementation would be more sophisticated

        position = particles["Coordinates"].values
        velocity = particles["Velocities"].values
        luminosity_dusty = particles["Luminosity_Dusty"].values

        if self.los_axis == 0:
            # x-axis as line of sight
            pos_projected = np.array([position[:, 1], position[:, 2]]).T
            velocity_los = velocity[:, 0]
        elif self.los_axis == 1:
            # y-axis as line of sight
            pos_projected = np.array([position[:, 0], position[:, 2]]).T
            velocity_los = velocity[:, 1]
        else:
            # z-axis as line of sight
            pos_projected = np.array([position[:, 0], position[:, 1]]).T
            velocity_los = velocity[:, 2]

        galaxy_center_x = np.sum(luminosity_dusty * pos_projected[:, 0]) / np.sum(
            luminosity_dusty
        )
        galaxy_center_y = np.sum(luminosity_dusty * pos_projected[:, 1]) / np.sum(
            luminosity_dusty
        )
        galaxy_center = np.array([galaxy_center_x, galaxy_center_y])

        return pos_projected, galaxy_center, velocity_los

    def direct_effective_radius_surface_brightness_sersic_index(
        self,
        galaxy_id,
        r_lim=30.0,
        nbins=100,
        method="stars",
    ):
        """
        Compute 'direct' effective radius R_e = sqrt(a_e * b_e)
        as in Xu+2017 §2.2: the geometric mean of the semi-major
        and semi-minor axes of the elliptical isophote enclosing
        half of total luminosity.

        Parameters
        ----------
        galaxy_id : int
            Galaxy identifier
        r_lim : float
            Maximum radius to consider (same units as x, y)
        nbins : int
            Number of radial bins for surface brightness profile
        method : str
            Particle type ('gas' or 'stars')

        Returns
        -------
        tuple : (R_e_fit, I_e_fit, n_fit, L_model, R_e, L_cum)
            R_e_fit : float
                Effective radius from Sersic fit, R_eff^mod in Xu+2017
            I_e_fit : float
                Surface brightness at effective radius from Sersic fit
            n_fit : float
                Sersic index from fit
            L_model : ndarray
                Modeled luminosity profile from Sersic fit, L^mod in Xu+2017
            R_e : float
                Direct effective radius (geometric mean), R_eff^dir in Xu+2017
            L_cum : ndarray
                Cumulative luminosity profile, L^sum in Xu+2017
            is_elliptical : bool
                True if the galaxy is classified as elliptical, False otherwise
        """

        particles = self.find_particles_in_galaxy(galaxy_id, method=method)

        pos, galactic_center, _ = self.find_galaxy_relative_positions_velocities(
            particles, method=method
        )

        rel_pos = pos - galactic_center  # relative positions

        dx, dy = rel_pos[:, 0], rel_pos[:, 1]

        # Rotate coordinates to align with major/minor axes
        cos_phi, sin_phi = (
            np.cos(particles["orientation_angle"]),
            np.sin(particles["orientation_angle"]),
        )
        x_rot = dx * cos_phi + dy * sin_phi  # along major axis
        y_rot = -dx * sin_phi + dy * cos_phi  # along minor axis

        # Compute elliptical radius for each particle
        q = particles["axial_ratio"]
        if q <= 0:
            raise ValueError("Invalid axis ratio (b/a <= 0).")
        # Elliptical radius
        r_ell = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

        r_geo = r_ell * np.sqrt(q)  # approximate circularized radius

        # Apply radial limit
        within_limit = r_geo <= r_lim
        r_geo = r_geo[within_limit]
        observed_luminosity = particles["Luminosity_Dusty"][within_limit]

        # Sort by elliptical radius
        idx = np.argsort(r_geo)
        r_sorted = r_geo[idx]
        L_sorted = observed_luminosity[idx]

        # Compute cumulative luminosity profile
        Lcum = np.cumsum(L_sorted)
        Lhalf = 0.5 * np.sum(observed_luminosity)

        # Find radius enclosing half the light (semi-major axis a_e)
        a_e = np.interp(Lhalf, Lcum, r_sorted)
        b_e = q * a_e

        # Geometric mean, the direct effective radius in Xu+2017
        R_e = np.sqrt(a_e * b_e)

        # using the Sersic profile fitting method from Xu+2017 to estimate Sersic index
        R = r_sorted  # approximate circularized radius

        # Calculate surface brightness profile
        r_bins = np.linspace(0, r_lim, nbins + 1)
        surface_brightness = np.zeros(nbins)

        # Digitize radii to get bin indices
        bin_indices = np.digitize(R, r_bins) - 1  # bins are 0-indexed

        # Total luminosity per bin (vectorized)
        Lsum_per_bin = np.bincount(
            bin_indices, weights=L_sorted, minlength=len(r_bins) - 1
        )

        # Area per bin (elliptical annulus)
        area = np.pi * q * (r_bins[1:] ** 2 - r_bins[:-1] ** 2)

        # Surface brightness (avoid division by zero)
        surface_brightness = np.zeros_like(area)
        valid = area > 0
        surface_brightness[valid] = Lsum_per_bin[valid] / area[valid]

        # Fit Sersic profile to surface brightness data to estimate Sersic index
        def sersic_profile(r, I_e, R_e, n):
            """
            Sersic profile function

            r : radius
            I_e : surface brightness at effective radius
            R_e : effective radius
            n : Sersic index
            Returns: surface brightness at radius r"""
            b_n = find_bn(n)
            return I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))

        # Fit the de Vaucouleurs profile (n=4) to estimate R_e and I_e
        def de_vaucouleurs_profile(r, I_e, R_e):
            """
            de Vaucouleurs profile function (Sersic n=4)

            r : radius
            I_e : surface brightness at effective radius
            R_e : effective radius
            Returns: surface brightness at radius r"""

            return I_e * np.exp(-7.669 * ((r / R_e) ** (1 / 4) - 1))

        # Fit the exponential profile (n=1) to estimate R_e and I_e
        def exponential_profile(r, I_e, R_e):
            """
            Exponential profile function (Sersic n=1)

            r : radius
            I_e : surface brightness at effective radius
            R_e : effective radius
            Returns: surface brightness at radius r"""

            return I_e * np.exp(-1.678 * ((r / R_e) - 1))

        def find_bn(n):
            """
            Approximate b_n for given Sersic index n with https://arxiv.org/pdf/astro-ph/0208404

            n: Sersic index
            Returns: b_n
            """
            return (
                0.01945 - 0.8902 * n + 10.95 * n**2 - 19.67 * n**3 + 13.43 * n**4
            ) * np.heaviside(0.36 - n, 1) + (
                2 * n
                - 1 / 3
                + 4.0 / (405 * n)
                + 46.0 / (25515 * n**2)
                + 131 / (1148175 * n**3)
                - 2194697.0 / (30690717750 * n**4)
            ) * np.heaviside(n - 0.36, 0)

        # Compare the chi-squared values between the de Vaucouleurs and the exponential fits
        def compare_chi_squared(
            popt_deV, popt_exp, r_fit, surface_brightness, valid_fit
        ):
            """Compare chi-squared values of different profile fits
            popt_deV : array
                Best-fit parameters from de Vaucouleurs fit
            popt_exp : array
                Best-fit parameters from exponential fit
            r_fit : array
                Radii for fitting
            surface_brightness : array
                Surface brightness data
            valid_fit : array
                Boolean array indicating valid fit points
            Returns: bool
                True if de Vaucouleurs fit is better, False otherwise"""

            elliptical_galaxy = False
            y_fit_de_vaucouleurs = de_vaucouleurs_profile(r_fit[valid_fit], *popt_deV)
            y_fit_exponential = exponential_profile(r_fit[valid_fit], *popt_exp)
            chi_squared_de_vaucouleurs = np.sum(
                (surface_brightness[valid_fit] - y_fit_de_vaucouleurs) ** 2
            )
            chi_squared_exponential = np.sum(
                (surface_brightness[valid_fit] - y_fit_exponential) ** 2
            )

            if chi_squared_de_vaucouleurs < chi_squared_exponential:
                elliptical_galaxy = True
            return elliptical_galaxy

        # Prepare data for fitting
        r_fit = 0.5 * (r_bins[:-1] + r_bins[1:])  # bin centers

        # Follow Xu+2017 and fit only the binned radial profile between 0.05*R_e and 3.0*R_e
        fit_min = 0.05 * R_e
        fit_max = 3.0 * R_e
        valid_fit = (r_fit >= fit_min) & (r_fit <= fit_max) & (surface_brightness > 0)

        # Find the sersic index n, effective radius R_e, and surface brightness I_e by fitting the Sersic profile
        bestfit, cov = curve_fit(
            sersic_profile,
            r_fit[valid_fit],
            surface_brightness[valid_fit],
            p0=[np.max(surface_brightness) * 0.5, R_e, 2.0],
            bounds=(
                [0, 0, 0.1],
                [np.max(surface_brightness), r_lim, 10.0],
            ),  # Setting the bounds for Sersic index between 0.1 and 10, effective radius between 0 and r_lim, surface brightness positive
        )

        # Find the best-fit parameters for de Vaucouleurs (n=4) and exponential (n=1) profiles
        bestfit_deV, _ = curve_fit(
            de_vaucouleurs_profile,
            r_fit[valid_fit],
            surface_brightness[valid_fit],
            p0=[np.max(surface_brightness) * 0.5, R_e],
            bounds=(
                [0, 0],
                [np.max(surface_brightness), r_lim],
            ),
        )

        bestfit_exp, _ = curve_fit(
            exponential_profile,
            r_fit[valid_fit],
            surface_brightness[valid_fit],
            p0=[np.max(surface_brightness) * 0.5, R_e],
            bounds=(
                [0, 0],
                [np.max(surface_brightness), r_lim],
            ),
        )

        # Determine if the galaxy is elliptical based on chi-squared comparison
        is_elliptical = compare_chi_squared(
            bestfit_deV, bestfit_exp, r_fit, surface_brightness, valid_fit
        )

        # The best-fit parameters of surface brightness at the effective radius, effective radius, and Sersic index
        I_e_fit, R_e_fit, n_fit = bestfit

        # Calculate the "model" luminosity with the analytical formula in Xu+2017. The model is calculated within 7 times the effective radius.
        b_n_fit = find_bn(n_fit)
        L_model = (
            I_e_fit
            * np.exp(b_n_fit)
            * R_e_fit**2
            * 2
            * np.pi
            * n_fit
            / (b_n_fit ** (2 * n_fit))
            * gamma(2 * n_fit)
            * gammainc(2 * n_fit, b_n_fit * (7.0 * R_e_fit / R_e_fit) ** (1 / n_fit))
        )

        self.galaxy_data[galaxy_id]["R_e_fit"] = R_e_fit
        self.galaxy_data[galaxy_id]["I_e_fit"] = I_e_fit
        self.galaxy_data[galaxy_id]["n_fit"] = n_fit
        self.galaxy_data[galaxy_id]["L_model"] = L_model
        self.galaxy_data[galaxy_id]["R_e_direct"] = R_e
        self.galaxy_data[galaxy_id]["is_elliptical"] = is_elliptical
        self.galaxy_data[galaxy_id]["L_cumulative"] = Lcum

        return R_e_fit, I_e_fit, n_fit, L_model, R_e, Lcum, is_elliptical

    def find_velocity_dispersion(
        self,
        galaxy_id,
        R_e_fit,
        orientation_angle,
        axial_ratio,
        method="stars",
        Rmin=0.3,
    ):
        """
        Find velocity dispersion for a galaxy within 0.5*R_e_fit

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        R_e_fit : float
            Effective radius from Sersic fit
        orientation_angle : float
            Position angle (radians), major axis orientation
        axial_ratio : float
            Axis ratio (b/a)
        method : str
            Method for velocity calculation ('stars', 'gas', 'both')
        Rmin : float
            Minimum radius in kpc to consider for velocity dispersion. This is set to the softening length of TNG simulations.
            The default value is 0.3 kpc for TNG50. Use 0.7 kpc for TNG100 and 1.5 kpc for TNG300.

        Returns:
        --------
        float : Projected line-of-sight stellar velocity dispersion in km/s
        """

        rel_pos, rel_vel = self.find_galaxy_relative_positions_velocities(
            galaxy_id, method=method
        )

        dx, dy = rel_pos[:, 0], rel_pos[:, 1]

        # Rotate coordinates to align with major/minor axes
        cos_phi, sin_phi = np.cos(orientation_angle), np.sin(orientation_angle)
        x_rot = dx * cos_phi + dy * sin_phi  # along major axis
        y_rot = -dx * sin_phi + dy * cos_phi  # along minor axis

        # Compute elliptical radius for each particle
        q = axial_ratio
        if q <= 0:
            raise ValueError("Invalid axis ratio (b/a <= 0).")
        r_ell = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

        # Select particles within 0.5*R_e_fit and greater than Rmin
        within_re = r_ell <= 0.5 * R_e_fit
        within_rmin = r_ell > Rmin
        selected_velocities = rel_vel[within_re & within_rmin]

        # The projected velocity in the z-direction. Will implement other options later.
        projected_velocities = selected_velocities[:, 2]

        # Find the velocity dispersion in the projected direction
        velocity_dispersion = np.std(projected_velocities)
        return velocity_dispersion

    def fit_Fundamental_Plane(
        self,
        R_e_fit,
        velocity_dispersion,
        third_variable,
        method="FP",
        uncertainties=0.05,
    ):
        """
        Fit the Fundamental Plane relation: log(R_e) = a * log(σ) + b * log(I_e) + c if method = 'FP'
        or the mass plane relation: log(R_e) = a * log(σ) + b * log(M_*) + c if method = 'MP' with hyperfit.

        Parameters:
        -----------
        R_e_fit : array
            Effective radii from Sersic fits in kpc
        velocity_dispersion : array
            Velocity dispersions in km/s
        third_variable : array
            Surface brightness I_e (in solar luminosity/kpc^2)if method = 'FP' or Stellar mass (in solar masses) M_*
            if method = 'MP'
        method : str
            Method to use ('FP' for Fundamental Plane, 'MP' for Mass Plane)
        uncertainties : float or array
            Uncertainties for all variables in the order of velocity dispersion, R_e_fit, and third_variable
            (fractional, e.g., 0.05 for 5%). If the input is float, it is assumed to be the same for all variables.

        Returns:
        --------
        results : dict
            Dictionary containing best-fit coefficients, intrinsic scatter, correlation coefficient, and number of galaxies
        """

        # Prepare data for fitting
        if method == "FP":
            # The generic definition of the Fundamental Plane
            X = np.vstack(
                [
                    np.log10(velocity_dispersion),
                    np.log10(R_e_fit),
                    np.log10(third_variable),
                ]
            ).T

            # The second axis is the dependent variable
            vertaxis = 1

        # Using the Mass Plane definition in https://arxiv.org/pdf/1208.3522 equation 27.
        elif method == "MP":
            X = np.vstack(
                [
                    np.log10(velocity_dispersion / 130.0),
                    np.log10(R_e_fit / 2.0),
                    np.log10(third_variable),
                ]
            ).T

            # The third axis is the dependent variable
            vertaxis = 2
        else:
            raise ValueError(
                "Invalid method. Use 'FP' for Fundamental Plane or 'MP' for Mass Plane."
            )

        # The covariance matrix with NxNx3 dimention, where N is the number of galaxies.
        cov = np.zeros((len(R_e_fit), len(R_e_fit), 3))
        # The diagonal elements are calculated from the uncertainties.
        if isinstance(uncertainties, float) or isinstance(uncertainties, int):
            frac_uncertainties = uncertainties
            for i in range(3):
                cov[:, :, i] = np.diag((frac_uncertainties * X[:, i]) ** 2)  # Variance
        else:
            frac_uncertainties = uncertainties
            for i in range(3):
                cov[:, :, i] = np.diag(
                    (frac_uncertainties[i] * X[:, i]) ** 2
                )  # Variance

        # Using hyperfit to fit the plane and obtain the dispersion on R_e_fit
        hf_setup = hf.linfit.LinFit(X, cov, vertaxis=vertaxis)

        # Run an optimization. Bounds on all coefficients, the last one is the bound on the intrinsic scatter.
        bounds = ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1.0e-5, 1.0))
        bestfit, vert_scat, _ = hf_setup.optimize(bounds=bounds, verbose=True)
        mcmc_samples, _ = hf_setup.emcee(bounds=bounds, maxiter=50000, verbose=True)
        error = np.std(mcmc_samples, axis=0)

        # Find predicted R_e_fit from the bestfit
        if method == "FP":
            log_Re_pred = (
                bestfit[1] * np.log10(velocity_dispersion)
                + bestfit[2] * np.log10(third_variable)
                + bestfit[0]
            )
        elif method == "MP":
            # The extra +log10(2.0) term is to convert R_e/2 back to R_e.
            log_Re_pred = (
                bestfit[1] * np.log10(velocity_dispersion / 130.0)
                + bestfit[2] * np.log10(third_variable)
                + bestfit[0]
            ) + np.log10(2.0)

        # Calculate correlation coefficient
        correlation = np.corrcoef(log_Re_pred, np.log10(R_e_fit))[0, 1]

        results = {
            "a": bestfit[0],
            "a_error": error[0],
            "b": bestfit[1],
            "b_error": error[1],
            "c": bestfit[2],
            "c_error": error[2],
            "intrinsic_scatter": vert_scat,
            "intrinsic_scatter_error": error[3],
            "correlation": correlation,
            "n_galaxies": len(R_e_fit),
        }

        return results

    # ===========================================================================
    # End Fundamental Plane Methods
    # ===========================================================================

    def calculate_rotation_velocity(self, galaxy_id, method="gas"):
        """
        Calculate rotation velocity for a galaxy

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        method : str
            Method for velocity calculation ('gas', 'stars', 'both')

        Returns:
        --------
        float : Rotation velocity in km/s
        """
        if method == "gas":
            # Use gas particles for kinematic analysis
            particle_mask = self.gas_data["SubhaloID"] == galaxy_id
            particles = self.gas_data[particle_mask]
        elif method == "stars":
            particle_mask = self.stellar_data["SubhaloID"] == galaxy_id
            particles = self.stellar_data[particle_mask]
        else:
            # Combine both
            gas_mask = self.gas_data["SubhaloID"] == galaxy_id
            star_mask = self.stellar_data["SubhaloID"] == galaxy_id
            particles = pd.concat(
                [self.gas_data[gas_mask], self.stellar_data[star_mask]]
            )

        if len(particles) == 0:
            return np.nan

        # Calculate rotation velocity
        # This is a simplified approach - actual implementation would be more sophisticated

        # Get galaxy center
        galaxy_center = self.galaxy_data[self.galaxy_data["SubhaloID"] == galaxy_id][
            "SubhaloPos"
        ].iloc[0]

        # Convert particle coordinates and velocities to proper numpy arrays
        coordinates = np.array([coord for coord in particles["Coordinates"].values])
        velocities = np.array([vel for vel in particles["Velocities"].values])

        # Calculate relative positions and velocities
        rel_pos = coordinates - galaxy_center
        rel_vel = velocities

        # Calculate cylindrical coordinates
        R = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
        phi = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])

        # Calculate rotational velocity component
        v_rot = -rel_vel[:, 0] * np.sin(phi) + rel_vel[:, 1] * np.cos(phi)

        # Take velocity at 2.2 times half-mass radius (typical for TF relation)
        r_half = self.galaxy_data[self.galaxy_data["SubhaloID"] == galaxy_id][
            "SubhaloHalfmassRad"
        ].iloc[0]
        target_radius = 2.2 * r_half

        # Find particles near target radius
        radius_mask = np.abs(R - target_radius) < 0.5 * r_half

        if np.sum(radius_mask) > 10:
            v_rot_target = np.median(v_rot[radius_mask])
        else:
            # Use all particles if not enough near target radius
            v_rot_target = np.median(v_rot)

        return np.abs(v_rot_target)

    def generate_tully_fisher_data(self, output_file="tully_fisher_data.csv"):
        """
        Generate complete Tully-Fisher relation dataset

        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        print("Generating Tully-Fisher relation data...")

        # Ensure output directory exists and get full path
        output_file = self._ensure_output_directory(output_file, "models")

        results = []

        for i, galaxy_id in enumerate(self.galaxy_data["SubhaloID"]):
            if i % 100 == 0:
                print(f"Processing galaxy {i}/{len(self.galaxy_data)}")

            # Get galaxy properties
            galaxy_row = self.galaxy_data[
                self.galaxy_data["SubhaloID"] == galaxy_id
            ].iloc[0]

            # Calculate photometry using original method
            magnitudes = self.stellar_population_synthesis(galaxy_id)

            # Calculate photometry using SKIRT ExtinctionOnly mode approach
            skirt_results = self.generate_absolute_magnitudes_skirt(galaxy_id)
            skirt_magnitudes = skirt_results["absolute_magnitudes"]
            skirt_magnitudes_dusty = skirt_results["absolute_magnitudes_dusty"]
            skirt_extinction = skirt_results["extinction"]

            # Calculate rotation velocity (prefer real TNG50 vmax over mock calculation)
            vmax_tng = galaxy_row.get("SubhaloMaxCircVel", np.nan)
            if not np.isnan(vmax_tng) and vmax_tng > 0:
                # Use TNG50's vmax as a proxy for rotation velocity
                v_rot = (
                    vmax_tng * 0.7
                )  # Empirical factor to convert vmax to v_rot at 2.2 R_eff
            else:
                # Fallback to mock particle calculation
                v_rot = self.calculate_rotation_velocity(galaxy_id, method="gas")

            # Store comprehensive results
            result = {
                # Basic identifiers
                "galaxy_id": galaxy_id,
                "snap_num": galaxy_row["SnapNum"],
                # Masses
                "stellar_mass": galaxy_row["SubhaloStellarMass"],
                "total_mass": galaxy_row["SubhaloMass"],
                "gas_mass": galaxy_row["SubhaloGasMass"],
                "dm_mass": galaxy_row["SubhaloDMMass"],
                "baryonic_mass": galaxy_row["SubhaloBaryonicMass"],
                "halo_mass": galaxy_row["SubhaloHaloMass"],
                # Mass ratios
                "stellar_to_halo_ratio": galaxy_row["StellarToHaloMassRatio"],
                "gas_to_stellar_ratio": galaxy_row["GasToStellarMassRatio"],
                "baryonic_to_halo_ratio": galaxy_row["BaryonicToHaloMassRatio"],
                # Positions and distances
                "pos_x": galaxy_row["SubhaloPosX"],
                "pos_y": galaxy_row["SubhaloPosY"],
                "pos_z": galaxy_row["SubhaloPosZ"],
                "distance_mpc": galaxy_row["DistanceMpc"],
                "distance_modulus": galaxy_row["DistanceModulus"],
                # Velocities
                "vel_x": galaxy_row["SubhaloVelX"],
                "vel_y": galaxy_row["SubhaloVelY"],
                "vel_z": galaxy_row["SubhaloVelZ"],
                "velocity_magnitude": galaxy_row["VelocityMagnitude"],
                "rotation_velocity": v_rot,  # DERIVED from mock particle kinematics
                "log_rotation_velocity": np.log10(v_rot)
                if not np.isnan(v_rot)
                else np.nan,
                "max_circular_velocity": galaxy_row[
                    "SubhaloMaxCircVel"
                ],  # REAL TNG50 vmax
                "velocity_dispersion": galaxy_row[
                    "SubhaloVelDisp"
                ],  # REAL TNG50 veldisp
                # Physical properties
                "sfr": galaxy_row["SubhaloSFR"],
                "specific_sfr": galaxy_row["SpecificSFR"],
                "gas_metallicity": galaxy_row["SubhaloGasMetallicity"],
                "stellar_metallicity": galaxy_row["SubhaloStellarMetallicity"],
                "half_mass_radius": galaxy_row["SubhaloHalfmassRad"],
                "surface_density": galaxy_row["SurfaceDensity"],
                "inclination": galaxy_row["Inclination"],
                # Absolute magnitudes (from mass-to-light relations)
                "abs_mag_g": galaxy_row["AbsMagG"],
                "abs_mag_r": galaxy_row["AbsMagR"],
                "abs_mag_i": galaxy_row["AbsMagI"],
                # Cosmological properties
                "redshift": galaxy_row["Redshift"],
                "lookback_time": galaxy_row["LookbackTime"],
                "cosmic_time": galaxy_row["CosmicTime"],
                # Multi-wavelength photometry (original mock SPS results)
                **magnitudes,
            }

            # Add SKIRT ExtinctionOnly mode results with prefixes
            for band in self.filter_wavelengths.keys():
                result[f"skirt_abs_mag_{band}"] = skirt_magnitudes.get(band, np.nan)
                result[f"skirt_abs_mag_{band}_dusty"] = skirt_magnitudes_dusty.get(
                    band, np.nan
                )
                result[f"skirt_extinction_{band}"] = skirt_extinction.get(band, np.nan)

            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Remove galaxies with invalid data
        df = df.dropna(subset=["rotation_velocity"])

        # Save to file
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} galaxies to {output_file}")

        return df

    def fit_tully_fisher_relation(self, df, band="r", mass_type="stellar"):
        """
        Fit Tully-Fisher relation for a given band

        Parameters:
        -----------
        df : pandas.DataFrame
            Tully-Fisher data
        band : str
            Photometric band
        mass_type : str
            Type of mass ('stellar', 'baryonic', 'luminosity')

        Returns:
        --------
        dict : Fit results
        """
        # Prepare data
        if mass_type == "stellar":
            y = np.log10(df["stellar_mass"])
            ylabel = r"$\log_{10}(M_*/M_{\odot})$"
        elif mass_type == "luminosity":
            # Convert magnitude to luminosity
            # Using distance modulus for nearby universe
            abs_mag = df[band]  # Assuming these are absolute magnitudes
            y = -0.4 * abs_mag  # log10(L/L_sun)
            ylabel = rf"$\log_{{10}}(L_{{{band}}}/L_{{\odot}})$"

        x = df["log_rotation_velocity"]

        # Remove invalid data
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        # Fit linear relation: y = a * x + b
        def linear_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(linear_func, x, y)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        # Calculate scatter
        y_pred = linear_func(x, slope, intercept)
        scatter = np.std(y - y_pred)

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        results = {
            "band": band,
            "mass_type": mass_type,
            "slope": slope,
            "slope_error": slope_err,
            "intercept": intercept,
            "intercept_error": intercept_err,
            "scatter": scatter,
            "correlation": correlation,
            "n_galaxies": len(x),
        }

        return results

    def plot_tully_fisher_relation(
        self, df, band="r", mass_type="stellar", output_file="tully_fisher_plot.png"
    ):
        """
        Plot Tully-Fisher relation

        Parameters:
        -----------
        df : pandas.DataFrame
            Tully-Fisher data
        band : str
            Photometric band
        mass_type : str
            Type of mass
        output_file : str
            Output plot filename
        """
        # Ensure output directory exists and get full path
        output_file = self._ensure_output_directory(output_file, "plots")

        # Helper function to format band names for LaTeX
        def format_band_for_latex(band_name):
            """Format band name for LaTeX, escaping underscores and handling special cases"""
            # Remove common prefixes for cleaner display
            clean_name = (
                band_name.replace("abs_mag_", "")
                .replace("skirt_abs_mag_", "")
                .replace("_dusty", "")
            )
            # Escape underscores for LaTeX
            clean_name = clean_name.replace("_", r"\_")
            return clean_name

        # Fit the relation
        fit_results = self.fit_tully_fisher_relation(df, band, mass_type)

        # Prepare data for plotting
        if mass_type == "stellar":
            y = np.log10(df["stellar_mass"])
            ylabel = r"$\log_{10}(M_*/M_{\odot})$"
        elif mass_type == "luminosity":
            abs_mag = df[band]
            y = -0.4 * abs_mag
            # Use the helper function to format band name
            clean_band = format_band_for_latex(band)
            ylabel = rf"$\log_{{10}}(L_{{{clean_band}}}/L_{{\odot}})$"

        x = df["log_rotation_velocity"]

        # Remove invalid data
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, alpha=0.6, s=20, c="blue", label="TNG50 galaxies")

        # Plot fit line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = fit_results["slope"] * x_fit + fit_results["intercept"]
        plt.plot(
            x_fit,
            y_fit,
            "r-",
            linewidth=2,
            label=f"Fit: slope = {fit_results['slope']:.2f} ± {fit_results['slope_error']:.2f}",
        )

        # Add scatter region
        plt.fill_between(
            x_fit,
            y_fit - fit_results["scatter"],
            y_fit + fit_results["scatter"],
            alpha=0.2,
            color="red",
            label=f"±1σ scatter = {fit_results['scatter']:.2f}",
        )

        plt.xlabel(r"$\log_{10}(V_{\rm rot})$ [km/s]")
        plt.ylabel(ylabel)

        # Format title with clean band name
        clean_band_title = format_band_for_latex(band)
        plt.title(f"Tully-Fisher Relation ({clean_band_title}-band, TNG50)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add fit statistics
        stats_text = f"N = {fit_results['n_galaxies']}\n"
        stats_text += f"r = {fit_results['correlation']:.3f}\n"
        stats_text += f"σ = {fit_results['scatter']:.3f}"
        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()

        return fit_results

    def analyze_parameter_correlations(
        self, df, output_file="parameter_correlations.png"
    ):
        """
        Analyze correlations between different galaxy parameters

        Parameters:
        -----------
        df : pandas.DataFrame
            Galaxy data
        output_file : str
            Output plot filename
        """
        # Ensure output directory exists and get full path
        output_file = self._ensure_output_directory(output_file, "plots")
        # Select key parameters for correlation analysis
        key_params = [
            "stellar_mass",
            "halo_mass",
            "gas_mass",
            "rotation_velocity",
            "max_circular_velocity",
            "velocity_dispersion",
            "sfr",
            "half_mass_radius",
            "surface_density",
            "distance_mpc",
            "gas_metallicity",
            "stellar_metallicity",
            "inclination",
        ]

        # Create correlation matrix
        corr_data = df[key_params].copy()

        # Log transform mass parameters for better visualization
        mass_params = ["stellar_mass", "halo_mass", "gas_mass"]
        for param in mass_params:
            if param in corr_data.columns:
                corr_data[f"log_{param}"] = np.log10(corr_data[param])
                corr_data.drop(param, axis=1, inplace=True)

        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title("Galaxy Parameter Correlations (TNG50)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()

        return correlation_matrix

    def create_comprehensive_plots(self, df, output_dir="../plots"):
        """
        Create comprehensive plots showing various galaxy relationships

        Parameters:
        -----------
        df : pandas.DataFrame
            Galaxy data
        output_dir : str
            Directory to save plots
        """
        # Ensure output directory exists using the helper method
        # This normalizes the path and ensures it goes to ../plots
        if not output_dir.startswith("../"):
            output_dir = "../plots"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Stellar mass vs Halo mass
        plt.figure(figsize=(10, 8))
        plt.scatter(
            np.log10(df["halo_mass"]),
            np.log10(df["stellar_mass"]),
            alpha=0.6,
            s=20,
            c=df["sfr"],
            cmap="viridis",
        )
        plt.colorbar(label="SFR [M☉/yr]")
        plt.xlabel(r"$\log_{10}(M_{\rm halo}/M_{\odot})$")
        plt.ylabel(r"$\log_{10}(M_*/M_{\odot})$")
        plt.title("Stellar Mass vs Halo Mass (TNG50)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/stellar_vs_halo_mass.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 2. Size-Mass relation
        plt.figure(figsize=(10, 8))
        plt.scatter(
            np.log10(df["stellar_mass"]),
            df["half_mass_radius"],
            alpha=0.6,
            s=20,
            c=df["sfr"],
            cmap="plasma",
        )
        plt.colorbar(label="SFR [M☉/yr]")
        plt.xlabel(r"$\log_{10}(M_*/M_{\odot})$")
        plt.ylabel(r"$R_{\rm eff}$ [kpc]")
        plt.title("Size-Mass Relation (TNG50)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/size_mass_relation.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 3. Velocity-Mass relations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Rotation velocity vs stellar mass
        ax1.scatter(
            np.log10(df["stellar_mass"]),
            np.log10(df["rotation_velocity"]),
            alpha=0.6,
            s=20,
            c=df["inclination"],
            cmap="coolwarm",
        )
        ax1.set_xlabel(r"$\log_{10}(M_*/M_{\odot})$")
        ax1.set_ylabel(r"$\log_{10}(V_{\rm rot})$ [km/s]")
        ax1.set_title("Tully-Fisher Relation")
        ax1.grid(True, alpha=0.3)

        # Max circular velocity vs halo mass
        ax2.scatter(
            np.log10(df["halo_mass"]),
            np.log10(df["max_circular_velocity"]),
            alpha=0.6,
            s=20,
            c=df["gas_to_stellar_ratio"],
            cmap="viridis",
        )
        ax2.set_xlabel(r"$\log_{10}(M_{\rm halo}/M_{\odot})$")
        ax2.set_ylabel(r"$\log_{10}(V_{\rm max})$ [km/s]")
        ax2.set_title("Halo Mass vs Max Velocity")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/velocity_mass_relations.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print(f"Comprehensive plots saved to {output_dir}/")

    def compare_magnitude_methods(self, df, output_file="magnitude_comparison.png"):
        """
        Compare absolute magnitudes from different methods

        Parameters:
        -----------
        df : pandas.DataFrame
            Galaxy data
        output_file : str
            Output plot filename
        """
        # Ensure output directory exists and get full path
        output_file = self._ensure_output_directory(output_file, "plots")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Compare r-band magnitudes
        bands_to_compare = ["g", "r", "i"]

        for i, band in enumerate(bands_to_compare):
            ax = axes[0, i]

            # Mass-based vs SKIRT dust-free
            if band in ["g", "r", "i"]:
                mass_based = df[f"abs_mag_{band}"]
                skirt_dustfree = df[f"skirt_abs_mag_{band}"]

                valid_mask = np.isfinite(mass_based) & np.isfinite(skirt_dustfree)
                if np.sum(valid_mask) > 10:
                    ax.scatter(
                        mass_based[valid_mask],
                        skirt_dustfree[valid_mask],
                        alpha=0.6,
                        s=20,
                        c=df["stellar_mass"][valid_mask],
                        cmap="viridis",
                        norm=plt.Normalize(vmin=1e9, vmax=1e12),
                    )

                    # Add 1:1 line
                    lims = [
                        min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1]),
                    ]
                    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
                    ax.set_xlim(lims)
                    ax.set_ylim(lims)

                    # Calculate correlation
                    corr = np.corrcoef(
                        mass_based[valid_mask], skirt_dustfree[valid_mask]
                    )[0, 1]
                    ax.text(
                        0.05,
                        0.95,
                        f"r = {corr:.3f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

            ax.set_xlabel(f"Mass-based M_{{{band}}} (AB mag)")
            ax.set_ylabel(f"SKIRT M_{{{band}}} (dust-free)")
            ax.set_title(f"{band}-band: Mass-based vs SKIRT")
            ax.grid(True, alpha=0.3)

        # Compare dust-free vs dusty SKIRT magnitudes
        for i, band in enumerate(bands_to_compare):
            ax = axes[1, i]

            skirt_dustfree = df[f"skirt_abs_mag_{band}"]
            skirt_dusty = df[f"skirt_abs_mag_{band}_dusty"]
            extinction = df[f"skirt_extinction_{band}"]

            valid_mask = np.isfinite(skirt_dustfree) & np.isfinite(skirt_dusty)
            if np.sum(valid_mask) > 10:
                ax.scatter(
                    skirt_dustfree[valid_mask],
                    skirt_dusty[valid_mask],
                    alpha=0.6,
                    s=20,
                    c=extinction[valid_mask],
                    cmap="Reds",
                    vmin=0,
                    vmax=1,
                )

                # Add 1:1 line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)

                # Calculate mean extinction
                mean_ext = np.mean(extinction[valid_mask])
                ax.text(
                    0.05,
                    0.95,
                    f"<A_{{{band}}}> = {mean_ext:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel(f"SKIRT M_{{{band}}} (dust-free)")
            ax.set_ylabel(f"SKIRT M_{{{band}}} (dusty)")
            ax.set_title(f"{band}-band: Dust-free vs Dusty")
            ax.grid(True, alpha=0.3)

        # Add colorbars
        sm1 = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=1e9, vmax=1e12)
        )
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, ax=axes[0, :], location="top", shrink=0.8)
        cbar1.set_label("Stellar Mass [M☉]")

        sm2 = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=1))
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=axes[1, :], location="bottom", shrink=0.8)
        cbar2.set_label("Extinction [mag]")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Magnitude comparison plot saved to {output_file}")

        # Print comparison statistics
        print("\n=== Magnitude Method Comparison ===")
        for band in bands_to_compare:
            if band in ["g", "r", "i"]:
                mass_based = df[f"abs_mag_{band}"]
                skirt_dustfree = df[f"skirt_abs_mag_{band}"]
                skirt_dusty = df[f"skirt_abs_mag_{band}_dusty"]
                extinction = df[f"skirt_extinction_{band}"]

                valid_mask = np.isfinite(mass_based) & np.isfinite(skirt_dustfree)
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(
                        mass_based[valid_mask], skirt_dustfree[valid_mask]
                    )[0, 1]
                    mean_diff = np.mean(
                        mass_based[valid_mask] - skirt_dustfree[valid_mask]
                    )
                    std_diff = np.std(
                        mass_based[valid_mask] - skirt_dustfree[valid_mask]
                    )
                    mean_ext = np.mean(extinction[valid_mask])

                    print(f"\n{band}-band:")
                    print(f"  Correlation (mass vs SKIRT): {corr:.3f}")
                    print(f"  Mean difference: {mean_diff:.3f} ± {std_diff:.3f} mag")
                    print(f"  Mean extinction: {mean_ext:.3f} mag")

    def debug_magnitude_calculations(self, galaxy_id):
        """
        Debug method to compare magnitude calculation methods

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier

        Returns:
        --------
        debug_info : dict
            Detailed comparison information
        """
        # Get galaxy properties
        galaxy_row = self.galaxy_data[self.galaxy_data["SubhaloID"] == galaxy_id].iloc[
            0
        ]
        stellar_mass = galaxy_row["SubhaloStellarMass"]

        print(f"\n=== Debug Galaxy {galaxy_id} ===")
        print(f"Stellar Mass: {stellar_mass:.2e} M_sun")

        # Method 1: Mass-based magnitudes
        mass_mags = {}
        for band in ["g", "r", "i"]:
            mass_mags[band] = self._estimate_absolute_magnitude(stellar_mass, band)
            print(f"Mass-based M_{band}: {mass_mags[band]:.3f}")

        # Method 2: SKIRT magnitudes
        skirt_results = self.generate_absolute_magnitudes_skirt(galaxy_id)
        skirt_mags = skirt_results["absolute_magnitudes"]
        skirt_extinctions = skirt_results["extinction"]

        print(f"\nSKIRT results:")
        for band in ["g", "r", "i"]:
            if band in skirt_mags:
                print(
                    f"SKIRT M_{band}: {skirt_mags[band]:.3f} (extinction: {skirt_extinctions[band]:.3f})"
                )

        # Calculate stellar SED details
        stellar_sed = self.calculate_stellar_sed_skirt(galaxy_id)
        print(f"\nStellar SED details:")
        print(f"Number of stellar particles: {len(stellar_sed['masses'])}")
        print(f"Total stellar mass: {np.sum(stellar_sed['masses']):.2e} M_sun")

        for band in ["g", "r", "i"]:
            if band in stellar_sed["luminosities"]:
                total_lum = np.sum(stellar_sed["luminosities"][band])
                print(f"Total L_{band}: {total_lum:.2e} L_sun")

        # Gas and dust properties
        gas_mask = self.gas_data["SubhaloID"] == galaxy_id
        gas_particles = self.gas_data[gas_mask]
        if len(gas_particles) > 0:
            gas_masses = gas_particles["Masses"].values * 1e10 / self.h
            gas_metallicities = gas_particles["Metallicity"].values
            dust_to_gas = self._calculate_dust_to_gas_ratio(gas_metallicities)
            dust_surface_density = self._calculate_dust_surface_density(
                gas_masses, None, dust_to_gas
            )
            print(f"\nDust properties:")
            print(f"Total gas mass: {np.sum(gas_masses):.2e} M_sun")
            print(f"Mean dust-to-gas ratio: {np.mean(dust_to_gas):.4f}")
            print(f"Dust surface density: {dust_surface_density:.6f}")

        return {
            "galaxy_id": galaxy_id,
            "stellar_mass": stellar_mass,
            "mass_based_mags": mass_mags,
            "skirt_mags": skirt_mags,
            "extinctions": skirt_extinctions,
            "stellar_sed": stellar_sed,
        }

    def compare_magnitude_methods_updated(
        self, df, output_file="magnitude_comparison_fixed.png"
    ):
        """
        Compare absolute magnitudes from different methods with improved statistics

        Parameters:
        -----------
        df : pandas.DataFrame
            Galaxy data
        output_file : str
            Output plot filename
        """
        # Ensure output directory exists and get full path
        output_file = self._ensure_output_directory(output_file, "plots")

        print("\n=== Magnitude Method Comparison (Updated) ===")

        bands_to_compare = ["g", "r", "i"]

        for band in bands_to_compare:
            if band in ["g", "r", "i"]:
                # Get magnitudes
                mass_based = df[f"abs_mag_{band}"]
                skirt_dustfree = df[f"skirt_abs_mag_{band}"]
                skirt_dusty = df[f"skirt_abs_mag_{band}_dusty"]
                extinction = df[f"skirt_extinction_{band}"]

                # Calculate statistics for valid data
                valid_mask = np.isfinite(mass_based) & np.isfinite(skirt_dustfree)

                if np.sum(valid_mask) > 10:
                    correlation = np.corrcoef(
                        mass_based[valid_mask], skirt_dustfree[valid_mask]
                    )[0, 1]
                    diff = mass_based[valid_mask] - skirt_dustfree[valid_mask]
                    mean_diff = np.mean(diff)
                    std_diff = np.std(diff)
                    mean_extinction = np.mean(extinction[valid_mask])

                    print(f"\n{band}-band:")
                    print(f"Correlation (mass vs SKIRT): {correlation:.3f}")
                    print(f"Mean difference: {mean_diff:.3f} ± {std_diff:.3f} mag")
                    print(f"Mean extinction: {mean_extinction:.3f} mag")

                    # Check for reasonable values
                    print(
                        f"Mass-based range: {np.min(mass_based[valid_mask]):.1f} to {np.max(mass_based[valid_mask]):.1f}"
                    )
                    print(
                        f"SKIRT range: {np.min(skirt_dustfree[valid_mask]):.1f} to {np.max(skirt_dustfree[valid_mask]):.1f}"
                    )
                    print(
                        f"Extinction range: {np.min(extinction[valid_mask]):.3f} to {np.max(extinction[valid_mask]):.3f}"
                    )

        # Create updated comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for i, band in enumerate(bands_to_compare):
            # Top row: Mass-based vs SKIRT dust-free
            ax = axes[0, i]

            mass_based = df[f"abs_mag_{band}"]
            skirt_dustfree = df[f"skirt_abs_mag_{band}"]

            valid_mask = np.isfinite(mass_based) & np.isfinite(skirt_dustfree)
            if np.sum(valid_mask) > 10:
                ax.scatter(
                    mass_based[valid_mask],
                    skirt_dustfree[valid_mask],
                    alpha=0.6,
                    s=20,
                    c=df["stellar_mass"][valid_mask],
                    cmap="viridis",
                    norm=plt.Normalize(vmin=1e9, vmax=1e12),
                )

                # Add 1:1 line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            ax.set_xlabel(f"Mass-based M_{{{band}}} (AB mag)")
            ax.set_ylabel(f"SKIRT M_{{{band}}} (dust-free)")
            ax.set_title(f"{band}-band: Mass-based vs SKIRT")
            ax.grid(True, alpha=0.3)

            # Bottom row: Dust-free vs dusty SKIRT magnitudes
            ax = axes[1, i]

            skirt_dusty = df[f"skirt_abs_mag_{band}_dusty"]
            extinction = df[f"skirt_extinction_{band}"]

            valid_mask = np.isfinite(skirt_dustfree) & np.isfinite(skirt_dusty)
            if np.sum(valid_mask) > 10:
                ax.scatter(
                    skirt_dustfree[valid_mask],
                    skirt_dusty[valid_mask],
                    alpha=0.6,
                    s=20,
                    c=extinction[valid_mask],
                    cmap="Reds",
                    vmin=0,
                    vmax=1,
                )

                # Add 1:1 line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)

                # Calculate mean extinction
                mean_ext = np.mean(extinction[valid_mask])
                ax.text(
                    0.05,
                    0.95,
                    f"<A_{{{band}}}> = {mean_ext:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel(f"SKIRT M_{{{band}}} (dust-free)")
            ax.set_ylabel(f"SKIRT M_{{{band}}} (dusty)")
            ax.set_title(f"{band}-band: Dust-free vs Dusty")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Updated magnitude comparison plot saved to {output_file}")

    def run_full_analysis(self, n_galaxies=1000, load_particle_only=False):
        """Run complete Tully-Fisher analysis with real TNG50 data"""
        print("Running TNG50 Tully-Fisher analysis with real data...")

        # Load real TNG50 data
        self.load_tng50_data(limit=n_galaxies, load_particle_only=load_particle_only)
        print("Loaded real TNG50 data")

        if load_particle_only:
            print(
                "Particle data loaded. Exiting analysis as per 'load_particle_only' flag."
            )
            return None
        else:
            # Generate Tully-Fisher data
            print("Generating Tully-Fisher relation data...")
            df = self.generate_tully_fisher_data("tng50_tully_fisher_complete.csv")
            print(f"Generated data for {len(df)} galaxies")

            # Create magnitude comparison plot
            print("Creating magnitude comparison plot...")
            self.compare_magnitude_methods_updated(
                df, "magnitude_comparison_complete.png"
            )

            # Create Tully-Fisher relation plots for different bands
            print("Creating Tully-Fisher relation plots...")
            for band in ["g", "r", "i"]:
                print(f"  Fitting and plotting {band}-band Tully-Fisher relation...")
                fit_results = self.plot_tully_fisher_relation(
                    df,
                    band=f"abs_mag_{band}",
                    mass_type="luminosity",
                    output_file=f"tully_fisher_{band}_band.png",
                )
                print(
                    f"    {band}-band: slope = {fit_results['slope']:.2f} ± {fit_results['slope_error']:.2f}"
                )

            # Also create stellar mass Tully-Fisher relation
            print("  Fitting and plotting stellar mass Tully-Fisher relation...")
            mass_fit_results = self.plot_tully_fisher_relation(
                df,
                band="r",
                mass_type="stellar",
                output_file="tully_fisher_stellar_mass.png",
            )
            print(
                f"    Stellar mass: slope = {mass_fit_results['slope']:.2f} ± {mass_fit_results['slope_error']:.2f}"
            )

            # Create parameter correlation analysis
            print("Creating parameter correlation analysis...")
            correlation_matrix = self.analyze_parameter_correlations(
                df, "parameter_correlations.png"
            )
            print("Parameter correlation analysis completed")

            # Create comprehensive plots showing galaxy relationships
            print("Creating comprehensive galaxy relationship plots...")
            self.create_comprehensive_plots(df, output_dir="plots")

            print("\n✅ Full analysis completed successfully!")
            print("Output files:")
            print("  - tng50_tully_fisher_complete.csv (galaxy data)")
            print("  - magnitude_comparison_complete.png (magnitude comparison)")
            print("  - tully_fisher_g_band.png (g-band Tully-Fisher relation)")
            print("  - tully_fisher_r_band.png (r-band Tully-Fisher relation)")
            print("  - tully_fisher_i_band.png (i-band Tully-Fisher relation)")
            print(
                "  - tully_fisher_stellar_mass.png (stellar mass Tully-Fisher relation)"
            )
            print("  - parameter_correlations.png (galaxy parameter correlations)")
            print("  - plots/ directory (comprehensive relationship plots)")

            return df


def main(load_particle_only=False):
    """Main function for TNG50 Tully-Fisher analysis"""
    import sys
    import os

    print("TNG50 Tully-Fisher Generator")
    print("=" * 40)

    # Get number of galaxies from command line argument or use default
    n_galaxies = 1000  # Default number of galaxies
    if len(sys.argv) > 1:
        try:
            n_galaxies = int(sys.argv[1])
            print(f"Loading {n_galaxies} galaxies from command line argument")
        except ValueError:
            print(f"Invalid number '{sys.argv[1]}', using default: {n_galaxies}")
    else:
        print(f"Using default number of galaxies: {n_galaxies}")

    print("Running analysis with real TNG50 data...")
    try:
        api_key = os.environ.get("TNG_API_KEY")
        if not api_key:
            print("TNG API key not found in environment variable TNG_API_KEY")
            print(
                "Please get your API key from: https://www.tng-project.org/users/profile/"
            )
            api_key = input("Enter your TNG API key: ")

        generator = TNG50TullyFisherGenerator(api_key=api_key)
        df = generator.run_full_analysis(
            n_galaxies=n_galaxies, load_particle_only=load_particle_only
        )
        print("\n✅ Real data analysis completed!")

    except Exception as e:
        print(f"❌ Real data analysis failed: {e}")


if __name__ == "__main__":
    main(load_particle_only=True)
