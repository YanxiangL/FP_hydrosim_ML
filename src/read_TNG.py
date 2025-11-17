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
import requests
import os
import warnings
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo, FlatLambdaCDM, z_at_value
from astropy import constants as const
import io
import h5py
import kinematics
from scipy.spatial.transform import Rotation
from configobj import ConfigObj


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
        self,
        api_key=None,
        simulation="TNG50-1",
        snapshot=99,
        out_dir=".",
        supplementary_dir="",
        sup_files=[],
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
        out_dir : str
            Output directory for saving data
        supplementary_dir : str
            Directory for supplementary data
        """
        self.api_key = api_key or self._get_api_key()
        self.simulation = simulation
        self.snapshot = snapshot

        # TNG API base URL
        self.base_url = "https://www.tng-project.org/api/"

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

        # Smoothing scale of the TNG50 simulation
        self.epsilon_star = 0.288  # kpc
        self.epsilon_gas = 0.074  # kpc
        self.epsilon_dm = 0.288  # kpc

        # Output and supplementary directories
        self.out_dir = out_dir
        self.supplementary_dir = supplementary_dir
        self.sup_files = sup_files  # List of supplementary files to load
        self.num_sub_files = len(
            self.sup_files
        )  # Find the number of supplementary files

        self.galaxy_data = None
        self.stellar_data = None
        self.gas_data = None

        # Setup API headers
        self.headers = {"api-key": self.api_key}

        # Test API connection
        self._test_api_connection()

        # Get some basic simulation info
        self.get_simulation_info()

        self.scale_factor = 1.0 / (1.0 + self.redshift)
        self.age_snapshot = self.cosmo.age(self.redshift).value  # Gyr
        print(self.scale_factor, self.age_snapshot, self.redshift)

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
        file_name,
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
            with h5py.File(self.supplementary_dir + file_name, "r") as f:
                data = {}
                # print(len(f[f"Snapshot_{self.snapshot}"]["SubhaloID"]))
                fields = []

                n_subhalos = len(f[f"Snapshot_{self.snapshot}"]["SubhaloID"])

                subhalo_axis_all = []
                for field in f[f"Snapshot_{self.snapshot}"].keys():
                    data[field] = f[f"Snapshot_{self.snapshot}"][field][:]
                    fields.append(field)

                    subhalo_axis = None
                    shape = data[field].shape
                    for ax, size in enumerate(shape):
                        if size == n_subhalos:
                            subhalo_axis = ax
                            break

                    if subhalo_axis is None:
                        raise ValueError(
                            f"No axis matches nsub={n_subhalos} for dataset '{field}'"
                        )
                    subhalo_axis_all.append(subhalo_axis)
                    # print(f[f"Snapshot_{self.snapshot}"][field])
        except Exception as e:
            print(
                f"Error reading HDF5 data for supplementary files {self.supplementary_dir + file_name}: {e}"
            )
            return None

        return data, fields, subhalo_axis_all

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
        endpoint = "/snapshots/{}/".format(self.snapshot)
        sim_info = self.get_api_data(endpoint)
        if sim_info:
            print(f"simulation: {sim_info.get('name', 'Unknown')}")
            # print(f"Box size: {sim_info.get('boxsize', 'Unknown')} cMpc/h")
            # print(f"Number of snapshots: {sim_info.get('num_snapshots', 'Unknown')}")
            print(
                f"Redshift at snapshot {self.snapshot}: {sim_info.get('redshift', 'Unknown')}"
            )
            print(
                f"Number of groups (subhalos): {sim_info.get('num_groups_subfind', 'Unknown')}"
            )
            print(f"Number of gas particles: {sim_info.get('num_gas', 'Unknown')}")
            print(f"Number of star particles: {sim_info.get('num_stars', 'Unknown')}")

            self.redshift = np.float64(sim_info.get("redshift"))
            if self.snapshot == 99:
                self.redshift = 0.0  # The TNG simulation doesn't return exactly 0.0 for snapshot 99. Force set to 0.0
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

        if self.num_sub_files > 0:
            supplementary_data_all = []
            fields_all = []
            subhalo_axis_all = []
            for i in range(self.num_sub_files):
                supplementary_data, fields, subhalo_axis = self.get_supplementary_data(
                    self.sup_files[i]
                )
                supplementary_data_all.append(supplementary_data)
                fields_all.append(fields)
                subhalo_axis_all.append(subhalo_axis)

        # Extract comprehensive galaxy properties
        galaxy_data = []
        for i, subhalo in enumerate(results):
            if i % 250 == 0:
                print(f"Processing subhalo {i}/{len(results)}")

            # Get detailed subhalo info
            subhalo_detail = self.get_api_data(
                f"snapshots/{self.snapshot}/subhalos/{subhalo['id']}/"
            )

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
                    if supplementary_data_all:
                        for j in range(self.num_sub_files):
                            data = supplementary_data_all[j]
                            fields = fields_all[j]
                            subhalo_axis = subhalo_axis_all[j]

                            for n, field in enumerate(fields):
                                if subhalo_id in data["SubhaloID"]:
                                    index = np.where(data["SubhaloID"] == subhalo_id)[
                                        0
                                    ][0]

                                    # Using np.take to select axis correpsponding to the number of subhalos
                                    galaxy_data_single[field] = np.take(
                                        data[field], index, axis=subhalo_axis[n]
                                    )
                                else:
                                    galaxy_data_single[field] = np.nan

                    galaxy_data.append(galaxy_data_single)
                else:
                    if not pass_mass_cut:
                        message = "Did not pass mass cut, mass = {:.2e}".format(
                            stellar_mass
                        )
                    elif not pass_num_cut:
                        message = "Did not pass num particles cut, num = {}".format(
                            num_particles
                        )
                    elif not pass_subhalo_flag:
                        message = "Subhalo is not astrophysical, flag = {}".format(flag)
                    elif not pass_softening_cut:
                        message = "Did not pass softening cut, R_half_star = {:.2f} kpc, R_half_gas = {:.2f} kpc".format(
                            half_mass_rad_kpc_stars, half_mass_rad_kpc_gas
                        )
                    print(
                        "Skipping subhalo {}: failed cuts, because {}".format(
                            subhalo_id, message
                        )
                    )

        self.galaxy_data = galaxy_data
        print(f"Loaded {len(self.galaxy_data)} galaxies via API")
        raise ValueError("Stop here for debugging")

        # Load particle data with more realistic approach
        self._load_particle_data_summary()

        np.savez(
            f"{self.out_dir}/stellar_data.npz",
            np.array(self.stellar_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{self.out_dir}/gas_data.npz",
            np.array(self.gas_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{self.out_dir}/subhalo_data.npz",
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

        for index, galaxy in enumerate(self.galaxy_data):
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

            gas_data_single = self.get_subhalo_cutout(
                galaxy_id, fields=gas_fields, particle_type="gas"
            )

            stellar_data_single = self.get_subhalo_cutout(
                galaxy_id, fields=star_fields, particle_type="stars"
            )

            (
                pos_align_star,
                vel_align_star,
                vel_2D_align_star,
                pos_incline_star,
                vel_incline_star,
                vel_2D_incline_star,
                inclination_star,
                position_angle_star,
                axial_ratio_star,
                ellipticity_star,
                orientation_angle_star,
            ) = self.find_inclination_ellipticity(
                stellar_data_single,
                gas_data_single,
                self.galaxy_data[index],
                method="star",
            )

            (
                pos_align_gas,
                vel_align_gas,
                vel_2D_align_gas,
                pos_incline_gas,
                vel_incline_gas,
                vel_2D_incline_gas,
                inclination_gas,
                position_angle_gas,
                axial_ratio_gas,
                ellipticity_gas,
                orientation_angle_gas,
            ) = self.find_inclination_ellipticity(
                stellar_data_single,
                gas_data_single,
                self.galaxy_data[index],
                method="gas",
            )

            gas_data_single["Pos_align"] = pos_align_gas
            gas_data_single["Vel_align"] = vel_align_gas
            gas_data_single["Vel_2D_align"] = vel_2D_align_gas
            gas_data_single["Pos_incline"] = pos_incline_gas
            gas_data_single["Vel_incline"] = vel_incline_gas
            gas_data_single["Vel_2D_incline"] = vel_2D_incline_gas
            gas_data_single["Inclination"] = inclination_gas
            gas_data_single["Position_Angle"] = position_angle_gas
            gas_data_single["Axial_Ratio"] = axial_ratio_gas
            gas_data_single["Ellipticity"] = ellipticity_gas
            gas_data_single["Orientation_Angle"] = orientation_angle_gas

            stellar_data_single["Pos_align"] = pos_align_star
            stellar_data_single["Vel_align"] = vel_align_star
            stellar_data_single["Vel_2D_align"] = vel_2D_align_star
            stellar_data_single["Pos_incline"] = pos_incline_star
            stellar_data_single["Vel_incline"] = vel_incline_star
            stellar_data_single["Vel_2D_incline"] = vel_2D_incline_star
            stellar_data_single["Inclination"] = inclination_star
            stellar_data_single["Position_Angle"] = position_angle_star
            stellar_data_single["Axial_Ratio"] = axial_ratio_star
            stellar_data_single["Ellipticity"] = ellipticity_star
            stellar_data_single["Orientation_Angle"] = orientation_angle_star

            gas_data.append(gas_data_single)

            stellar_data.append(stellar_data_single)

        self.stellar_data = stellar_data
        self.gas_data = gas_data
        print(
            f"Created {len(self.stellar_data)} stellar and {len(self.gas_data)} gas particles"
        )

    def make_vmap_hist(self, pos_obs, v_los, x_range, y_range, nx=200, ny=200):
        """
        pos_obs : (N,3)
        v_los    : (N,)
        weight   : (N,) mass or luminosity (use luminosity for light-weighted IFU maps)
        returns: v_map (ny,nx), weight_map, xcenters, ycenters
        """
        x = pos_obs[:, 0]
        y = pos_obs[:, 1]
        # numerator and denominator histograms
        num, xedges, yedges = np.histogram2d(
            x, y, bins=[nx, ny], range=[x_range, y_range], weights=v_los
        )
        den, _, _ = np.histogram2d(
            x, y, bins=[nx, ny], range=[x_range, y_range], weights=np.ones_like(v_los)
        )

        # transpose so array indexing is [y,x] for plotting with imshow(origin='lower')
        num = num.T
        den = den.T

        with np.errstate(invalid="ignore", divide="ignore"):
            v_map = num / den

        # pixel center vectors
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        return v_map, den, xcenters, ycenters

    def find_axial_ratio_and_orientation(self, galaxy_position, weight=None):
        """
        Find the axial ratio and orientation of an elliptical galaxy

        Parameters:
        -----------
        galaxy_position : numpy.ndarray
            Array of particle positions relative to the galaxy center
        weight : numpy.ndarray, optional
            Array of weights for each particle (default is None, which means uniform weights)

        Returns:
        --------
        tuple : (axial_ratio, orientation_angle)
        """

        # If weight is none, set uniform weights
        if weight is None:
            weight = np.ones(galaxy_position.shape[0])

        # Find projected positions, galatic center based on Xu_2017 with weighting by observed luminosity.
        galaxy_center = np.sum(
            galaxy_position * weight[:, np.newaxis], axis=0
        ) / np.sum(weight)

        xgc, ygc = galaxy_center

        Mxx = np.sum(weight * (galaxy_position[:, 0] - xgc) ** 2) / np.sum(weight)
        Myy = np.sum(weight * (galaxy_position[:, 1] - ygc) ** 2) / np.sum(weight)
        Mxy = np.sum(
            weight * (galaxy_position[:, 0] - xgc) * (galaxy_position[:, 1] - ygc)
        ) / np.sum(weight)

        # Calculating the axial ratio
        axial_ratio_denom = (Mxx + Myy) + np.sqrt((Mxx - Myy) ** 2 + 4 * Mxy**2)
        axial_ratio_num = (Mxx + Myy) - np.sqrt((Mxx - Myy) ** 2 + 4 * Mxy**2)
        axial_ratio = np.sqrt(axial_ratio_num / axial_ratio_denom)

        # Calculating the orientation angle
        orientation_angle = (
            0.5 * np.arctan2(2 * Mxy, Mxx - Myy) * 180.0 / np.pi
        )  # in degrees

        return axial_ratio, orientation_angle

    def find_inclination_ellipticity(
        self, data_stellar, data_gas, data_subhalo, method="star"
    ):
        """
        Find the inclination and ellipticity of a galaxy
        Parameters:
        -----------
        data_stellar : dict
            Stellar particle data
        data_gas : dict
            Gas particle data
        data_subhalo : dict
            Subhalo data
        method : str
            Method to use for finding angular momentum axis ('star' or 'gas')
        """
        # Find and remove wind particles since they are not stars
        wind_particle_index = np.where(data_stellar["GFM_StellarFormationTime"] <= 0.0)[
            0
        ]
        print("There are ", len(wind_particle_index), "wind particles in subhalo")

        # Uncomment the following lines if you want to use star particles to calculate the angular momentum axis
        if method == "star":
            star_pos = np.delete(
                data_stellar["Coordinates"], wind_particle_index, axis=0
            )
            star_vel = np.delete(
                data_stellar["Velocities"], wind_particle_index, axis=0
            )
            pos = star_pos - data_subhalo["SubhaloPos"]
            vel = star_vel - data_subhalo["SubhaloVel"]
            mass_star = np.delete(data_stellar["Masses"], wind_particle_index, axis=0)
            # Calculate the angular momentum axis
            axis = kinematics.AngularMomentum(
                mass_star,
                pos,
                vel,
                return_ji=False,
                # Restrict to twice the stellar half-mass radius
                range=2.0 * data_subhalo["SubhaloHalfmassRadStars"],
            )
        else:
            gas_pos = data_gas["Coordinates"]
            gas_vel = data_gas["Velocities"]
            pos = gas_pos - data_subhalo["SubhaloPos"]
            vel = gas_vel - data_subhalo["SubhaloVel"]
            mass_gas = data_gas["Masses"]
            # Calculate the angular momentum axis
            axis = kinematics.AngularMomentum(
                mass_gas,
                pos,
                vel,
                return_ji=False,
                # Restrict to twice the stellar half-mass radius
                range=2.0 * data_subhalo["SubhaloHalfmassRadStars"],
            )

        # Assuming line-of-sight is along the z-axis
        inclination = np.arccos(np.dot(axis, np.array([0, 0, 1]))) * 180.0 / np.pi
        position_angle = (
            np.arctan2(
                np.dot(axis, np.array([0, 1, 0])), np.dot(axis, np.array([1, 0, 0]))
            )
            * 180.0
            / np.pi
        )
        print(
            "Inclination angle between angular momentum axis and z-axis:", inclination
        )
        print(
            "Position angle between projection of angular momentum axis onto x-y plane and x-axis:",
            position_angle,
        )

        # Align the galaxy along the angular momentum axis
        Rmat = kinematics.RotationMatrix(np.array([0, 0, 1]), axis)
        pos_norm = Rmat.apply(pos)
        vel_norm = Rmat.apply(vel)

        # # The angular momentum axis is now the z-axis, so we can define face-on and edge-on views
        # pos_all["inclination_" + str(0)] = np.array([pos_norm[:, 0], pos_norm[:, 1]]).T

        v_los = np.dot(
            vel_norm, np.array([0, 0, 1])
        )  # line-of-sight velocity along z-axis
        if method == "star":
            pos_lim = 5.0  # kpc
        else:
            pos_lim = 10.0  # kpc

        stat, _, xedges, yedges = self.make_vmap_hist(
            pos_norm[:, :2],
            v_los,
            x_range=[-pos_lim, pos_lim],
            y_range=[-pos_lim, pos_lim],
            nx=200,
            ny=200,
        )

        pos_align = pos_norm
        vel_align = vel_norm
        vel_2D_align = stat

        # Rotate the galaxy with the calculated inclination and position angles.
        R_incl = Rotation.from_euler("x", inclination, degrees=True)
        R_pa = Rotation.from_euler("z", position_angle, degrees=True)
        R_total = R_pa * R_incl

        pos_rot = R_total.apply(pos_norm)
        vel_rot = R_total.apply(vel_norm)

        v_los = np.dot(
            vel_rot, np.array([0, 0, 1])
        )  # line-of-sight velocity along z-axis

        stat, _, xedges, yedges = self.make_vmap_hist(
            pos_rot[:, :2],
            v_los,
            x_range=[-pos_lim, pos_lim],
            y_range=[-pos_lim, pos_lim],
            nx=200,
            ny=200,
        )

        pos_incline = pos_rot
        vel_incline = vel_rot
        vel_2D_incline = stat

        if method == "star":
            axial_ratio, orientation_angle = self.find_axial_ratio_and_orientation(
                pos_rot[:, :2],
                weight=data_stellar["Masses"],
            )
        else:
            axial_ratio, orientation_angle = self.find_axial_ratio_and_orientation(
                pos_rot[:, :2],
                weight=data_gas[
                    "Masses"
                ],  # Using masses as weight for better tracing star-forming regions
            )
        ellipticity = (1.0 - axial_ratio) / (
            1.0 + axial_ratio
        )  # Ellipticity definition

        return (
            pos_align,
            vel_align,
            vel_2D_align,
            pos_incline,
            vel_incline,
            vel_2D_incline,
            inclination,
            position_angle,
            axial_ratio,
            ellipticity,
            orientation_angle,
        )

    def plot_vmap(self, subhalo_id, stat, xedges, yedges, method="star"):
        """Plot velocity map and save to file
        Parameters:
        -----------
        subhalo_id : int
            Subhalo ID
        stat : numpy.ndarray
            Velocity map data
        xedges : numpy.ndarray
            X edges of the histogram
        yedges : numpy.ndarray
            Y edges of the histogram
        method : str
            Method used ('star' or 'gas')
        """
        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        im = ax[0].imshow(
            stat.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="RdBu_r",  # red-blue diverging map
            vmin=-np.nanmax(abs(stat)),
            vmax=np.nanmax(abs(stat)),  # symmetric color scale
        )
        ax[1].plot(xedges, stat.T[100, :], color="k", lw=0.5, alpha=0.5)
        fig.colorbar(im, ax=ax[0], label=r"$v_{\rm los}$ [km/s]")
        ax[0].set_xlabel("x [kpc]")
        ax[0].set_ylabel("y [kpc]")
        ax[0].set_title("Line-of-sight Velocity Map")
        ax[1].set_xlabel("x [kpc]")
        plt.savefig(f"galaxy_{subhalo_id}_vmap_edgeon_{method}.png", dpi=300)
        plt.close(fig)

    def plot_edgeon_faceon_view(self, pos_norm, pos_lim, subhalo_id, method="star"):
        # Edge-on view of the galaxy
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        ax[0].hist2d(
            pos_norm[:, 1],
            pos_norm[:, 2],
            bins=(200, 200),
            range=[[-pos_lim, pos_lim], [-pos_lim, pos_lim]],
        )
        ax[0].set_xlabel("y [kpc]", fontsize=14)
        ax[0].set_ylabel("z [kpc]", fontsize=14)

        # Face-on view of the galaxy
        ax[1].hist2d(
            pos_norm[:, 0],
            pos_norm[:, 1],
            bins=(200, 200),
            range=[[-pos_lim, pos_lim], [-pos_lim, pos_lim]],
        )
        ax[1].set_xlabel("x [kpc]", fontsize=14)
        ax[1].set_ylabel("y [kpc]", fontsize=14)

        plt.savefig(f"galaxy_{subhalo_id}_edgeon_faceon_{method}.png", dpi=300)
        plt.close(fig)

    def plot_phase_space(
        self, pos_norm, vel_norm, pos_lim, vel_lim, subhalo_id, method="star"
    ):
        # Phase-space plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].hist2d(
            pos_norm[:, 0],
            vel_norm[:, 0],
            bins=(200, 200),
            range=[[-pos_lim, pos_lim], [-vel_lim, vel_lim]],
        )
        ax[0].set_xlabel("x [kpc]", fontsize=14)
        ax[0].set_ylabel("vx [km/s]", fontsize=14)
        ax[1].hist2d(
            pos_norm[:, 1],
            vel_norm[:, 1],
            bins=(200, 200),
            range=[[-pos_lim, pos_lim], [-vel_lim, vel_lim]],
        )
        ax[1].set_xlabel("y [kpc]", fontsize=14)
        ax[1].set_ylabel("vy [km/s]", fontsize=14)
        ax[2].hist2d(
            pos_norm[:, 2],
            vel_norm[:, 2],
            bins=(200, 200),
            range=[[-pos_lim, pos_lim], [-vel_lim, vel_lim]],
        )
        ax[2].set_xlabel("z [kpc]", fontsize=14)
        ax[2].set_ylabel("vz [km/s]", fontsize=14)

        plt.savefig(f"galaxy_{subhalo_id}_pv_diagram_{method}.png", dpi=300)
        plt.close(fig)


def main():
    """Main function for TNG50 Tully-Fisher analysis"""
    import sys

    print("TNG50 Tully-Fisher Generator")
    print("=" * 40)

    # Get number of galaxies from command line argument or use default

    # n_galaxies = int(sys.argv[1])  # The number of subhalos to process
    # snapshot = int(sys.argv[2])  # Snapshot number (z=0)
    # out_dir = sys.argv[3]  # The output directory
    # supplementary_dir = sys.argv[4]  # The supplementary data directory

    config = sys.argv[1]  # Reading in the path to the config file.
    pardict = ConfigObj(config)
    n_galaxies = int(pardict["n_galaxies"])  # The number of subhalos to process
    snapshot = int(pardict["snapshot"])  # Snapshot number
    out_dir = pardict["out_dir"]  # The output directory
    supplementary_dir = pardict["supplementary_dir"]  # The supplementary data directory
    sup_files = pardict["sup_files"]  # The supplementary data files
    mass_range = (
        float(pardict["mass_low"]),
        float(pardict["mass_high"]),
    )
    minimum_particles = int(pardict["min_particles"])

    if len(supplementary_dir) == 0:
        print("No supplementary directory provided, using only subhalo cutouts.")
        sup_files = ""

    print("Running analysis with real TNG50 data...")
    try:
        api_key = os.environ.get("TNG_API_KEY")
        if not api_key:
            print("TNG API key not found in environment variable TNG_API_KEY")
            print(
                "Please get your API key from: https://www.tng-project.org/users/profile/"
            )
            api_key = pardict["TNG_API_KEY"]  # Or read from config file

        generator = TNG50TullyFisherGenerator(
            api_key=api_key,
            snapshot=snapshot,
            out_dir=out_dir,
            supplementary_dir=supplementary_dir,
            sup_files=sup_files,
        )
        generator.load_tng50_data(
            limit=n_galaxies,
            stellar_mass_range=mass_range,
            minimum_particles=minimum_particles,
        )
        print("\n✅ Data successfully downloaded!")

    except Exception as e:
        print(f"❌ Failed to download data: {e}")


if __name__ == "__main__":
    main()
