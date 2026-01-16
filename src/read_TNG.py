#!/usr/bin/env python3
"""
TNG Tully-Fisher Relation Data Generation Framework

This script generates multi-wavelength Tully-Fisher relation data using
TNG cosmological simulation data accessed through the IllustrisTNG web API.

IMPORTANT: This uses the TNG web API - no need to download full simulation data!
You just need to:
1. Register at https://www.tng-project.org/users/register/
2. Get your API key from https://www.tng-project.org/users/profile/
3. Install required packages: pip install numpy scipy matplotlib pandas astropy requests

Key Components:
1. TNG data loading via web API
2. Stellar population synthesis (SPS) modeling
3. Multi-wavelength photometry generation
4. Kinematic analysis for rotation velocities
5. Tully-Fisher relation construction

NEW: SKIRT ExtinctionOnly Mode Integration
6. Stellar population synthesis using age and metallicity from TNG stellar particles
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
from astropy.cosmology import LambdaCDM, z_at_value
from astropy import constants as const
import io
import h5py
import kinematics
from scipy.spatial.transform import Rotation
from configobj import ConfigObj
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev, splint
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.special import gamma, gammainc
from functools import partial
import illustris_python as il
import sys

warnings.filterwarnings("ignore")

# All TNG API functionality implemented directly using requests
# No need for illustris_python package
#
# To install all dependencies:
# pip install -r requirements.txt
# or manually: pip install numpy scipy matplotlib pandas astropy requests


class TNG50TullyFisherGenerator:
    """
    Main class for generating Tully-Fisher relation data from TNG simulation
    Uses the TNG web API directly with requests - no additional packages needed!
    """

    def __init__(
        self,
        api_key=None,
        simulation="TNG50-1",
        snapshot=99,
        pardict=None,
        job_id=None,
    ):
        """
        Initialize the TNG Tully-Fisher data generator

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
        if pardict["mode_load"] == "API":
            self.api_key = api_key or self._get_api_key()
            # TNG API base URL
            self.base_url = "https://www.tng-project.org/api/"

        self.simulation = simulation
        self.snapshot = snapshot

        # Physical constants
        self.c = 299792.458  # km/s
        # The cosmological parameters are those used in the TNG simulations
        self.h = float(pardict["h"])  # Hubble parameter
        self.Om0 = float(pardict["omega_matter"])  # Matter density parameter

        self.Mpc = 3.085677581491367e24  # cm
        self.L_sun = 3.826e33  # erg/s. The default value in galaxev
        self.csol = 2.99792458e10  # speed of light in cm/s
        self.H_mass = 1.6735575e-24  # g
        self.Msun_to_g = 1.98847e33  # solar mass in grams
        self.kpc_to_cm = 3.085677581491367e21  # kpc to cm
        self.H_mass_u = 1.00784  # atomic mass unit for hydrogen
        self.O_mass_u = 15.999  # atomic mass unit for oxygen
        self.Gyr_to_s = 3.1536e16  # Gyr to seconds
        self.k_B = 1.380649e-16  # erg/K
        self.m_p = 1.67262192369e-24  # g, mass of proton
        self.gamma = 5.0 / 3.0  # Adiabatic index for ideal monoatomic gas

        # Set up cosmology for SKIRT calculations
        # self.cosmo = FlatLambdaCDM(H0=self.h * 100, Om0=self.Om0)
        self.cosmo = LambdaCDM(
            H0=self.h * 100,
            Om0=self.Om0,
            Ob0=float(pardict["omega_b"]),
            Ode0=float(pardict["omega_lambda"]),
        )

        # Smoothing scale of the TNG50 simulation. Will move this to user config in the future
        if (
            self.simulation == "TNG50-1"
            or self.simulation == "TNG50-2"
            or self.simulation == "TNG50-3"
        ):
            self.epsilon_star = 0.288  # kpc
            self.epsilon_gas = 0.074  # kpc
            self.epsilon_dm = 0.288  # kpc
        elif (
            self.simulation == "TNG100-1"
            or self.simulation == "TNG100-2"
            or self.simulation == "TNG100-3"
        ):
            self.epsilon_star = 0.74  # kpc
            self.epsilon_gas = 0.185  # kpc
            self.epsilon_dm = 0.74  # kpc
        else:
            raise ValueError(
                "Unsupported simulation. Please use TNG50-1, TNG50-2, TNG50-3, TNG100-1, TNG100-2, or TNG100-3."
            )

        # Output and supplementary directories
        self.out_dir = pardict["out_dir"]  # The output directory
        self.supplementary_dir = pardict[
            "supplementary_dir"
        ]  # The supplementary data directory
        self.sup_files = pardict["sup_files"]  # The supplementary data files
        if len(self.supplementary_dir) == 0:
            print("No supplementary directory provided, using only subhalo cutouts.")
            self.sup_files = ""
        self.pardict = pardict
        self.num_sub_files = len(
            self.sup_files
        )  # Find the number of supplementary files

        self.galaxy_data = None
        self.stellar_data = None
        self.gas_data = None
        self.galaxy_index_below = None
        self.hdf5_num = None

        # The line-of-sight axis for projection (0=x, 1=y, 2=z)
        self.los_axis = int(pardict["los_axis"])

        if pardict["mode_load"] == "API":
            # Setup API headers
            self.headers = {"api-key": self.api_key}
            # Test API connection
            self._test_api_connection()
            # Get some basic simulation info
            self.get_simulation_info()
        else:
            self.find_redshift()

        self.scale_factor = 1.0 / (1.0 + self.redshift)
        self.age_snapshot = self.cosmo.age(self.redshift).value  # Gyr
        print(self.scale_factor, self.age_snapshot, self.redshift)

        self.SubhaloID_to_index = []
        self.job_id = job_id

        # Loading the magnitude grids for SPS-based magnitudes
        if pardict["mode"] == "sfh":
            print("Outputing star formation hisotry only.")
        elif pardict["mode"] == "analysis":
            self.load_saved_data()

            print("Loading SPS magnitude grids from pygalaxev...")
            if pardict["sfh_model"] == "custom":
                self._load_galaxev_models_custom()

            else:
                self._load_galaxev_models()

        else:
            raise ValueError("Invalid mode. Choose 'sfh' or 'analysis'.")

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
    def get_subhalo_cutout(
        self,
        subhalo_id,
        fields=None,
        particle_type="stars",
        index=None,
    ):
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

        try:
            if particle_type == "stars":
                particle_str = "PartType4"
                particle_num = 4
            elif particle_type == "gas":
                particle_str = "PartType0"
                particle_num = 0
            else:
                raise ValueError("Invalid particle type. Only support stars, gas.")

            print(
                f"Downloading cutout for subhalo {subhalo_id}, particle type: {particle_type}"
            )

            if particle_type == "stars":
                cutout_len = self.galaxy_data[index]["SubhaloLenStars"]
                print(
                    "Galaxy data catalogue states this subhalo has",
                    self.galaxy_data[index]["SubhaloLenStars"],
                    "star particles.",
                )

            else:
                cutout_len = self.galaxy_data[index]["SubhaloLenGas"]
                print(
                    "Galaxy data catalogue states this subhalo has",
                    self.galaxy_data[index]["SubhaloLenGas"],
                    "gas particles.",
                )

            if cutout_len == 0:
                # Some elliptical galaxies have no gas particles at all
                if (
                    particle_type == "gas"
                    and self.pardict["galaxy_type"] == "elliptical"
                ):
                    data = {}
                    for field in fields:
                        data[field] = None
                    return data

            if self.pardict["mode_load"] == "API":
                endpoint = (
                    f"snapshots/{self.snapshot}/subhalos/{subhalo_id}/cutout.hdf5"
                )
                params = {particle_type: ",".join(fields)} if fields else {}
                # params = {
                #     "gas": "Coordinates,Velocities,Masses,ParticleIDs,GFM_Metals,Density,GFM_Metallicity,NeutralHydrogenAbundance",  # Gas (PartType0)
                # }

                print(f"{self.base_url}{self.simulation}/{endpoint}")
                response = requests.get(
                    f"{self.base_url}{self.simulation}/{endpoint}",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                try:
                    with h5py.File(io.BytesIO(response.content), "r") as f:
                        data = {}
                        for field in fields:
                            data[field] = f[particle_str][field][:]
                except Exception as e:
                    print(f"Error reading HDF5 data for subhalo {subhalo_id}: {e}")
                    return None
            else:
                # path = (
                #     self.pardict["TNG_dir"]
                #     + f"/snap_{self.snapshot:03d}.{self.hdf5_num}.hdf5"
                # )

                # print("Using local cutout file:", path)

                data = il.snapshot.loadSubhalo(
                    self.pardict["TNG_dir"],
                    self.snapshot,
                    subhalo_id,
                    partType=particle_num,
                    fields=fields,
                )

                del data["count"]

                for field in fields:
                    data[field] = np.array(data[field], dtype=np.float64)

            print("Total particles retrieved:", len(data["Coordinates"]))

            data["Velocities"] = data["Velocities"] * np.sqrt(
                self.scale_factor
            )  # Convert to physical velocities
            data["Coordinates"] = (
                data["Coordinates"] * self.scale_factor / self.h
            )  # Convert to physical kpc
            data["Masses"] = data["Masses"] * 1e10 / self.h  # Convert to solar masses
            # data["SubhaloID"] = subhalo_id

            if particle_type == "gas":
                data["Density"] = (
                    data["Density"] * 1e10 * self.h**2 / self.scale_factor**3
                )  # Convert to solar masses per kpc^3

                mean_molecular_weight = (
                    4.0
                    * self.m_p
                    / (1.0 + 3.0 * 0.76 + 4 * 0.76 * data["ElectronAbundance"])
                )  # in g
                data["Temperature"] = (
                    (self.gamma - 1.0)
                    * data["InternalEnergy"]
                    / self.k_B
                    * mean_molecular_weight
                    * (self.kpc_to_cm / self.Gyr_to_s) ** 2
                )  # in K

                # Remove the electron abundance keys since they are not needed anymore
                del data["InternalEnergy"]

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

                wind_particle_index = np.where(data["GFM_StellarFormationTime"] <= 0.0)[
                    0
                ]

                if len(wind_particle_index) > 0:
                    print(
                        "There are ",
                        len(wind_particle_index),
                        "wind particles in subhalo. Removing them.",
                    )
                    for key in data.keys():
                        data[key] = np.delete(data[key], wind_particle_index, axis=0)

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

    # Getting the redshift of the simulation snapshot from the group catalog
    def find_redshift(self):
        file_path = (
            self.pardict["TNG_dir"]
            + f"/groups_{int(self.snapshot):03d}/fof_subhalo_tab_{int(self.pardict['snapshot']):03d}.0.hdf5"
        )
        with h5py.File(file_path, "r") as f:
            redshift = f["Header"].attrs["Redshift"]

        if self.snapshot == 99:
            redshift = 0.0  # The TNG simulation doesn't return exactly 0.0 for snapshot 99. Force set to 0.0
        self.redshift = redshift
        print(f"Redshift at snapshot {self.snapshot}: {self.redshift}")

    def find_subhalo_ID(self):
        """
        Generating the subhalo ID list from all the hdf5 files

        :param self: Description
        :return: Description
        -----------
        tot : np.ndarray
            Array of subhalo counts per file
        ssfr_ridge : float
            The sSFR ridge value in log10(Gyr^-1)
        """
        tot = []
        index = -1
        mass_all = []
        sfr_all = []
        for i in range(int(self.pardict["num_catalog"])):
            file_path = (
                self.pardict["TNG_dir"]
                + f"/groups_{int(self.pardict['snapshot']):03d}/fof_subhalo_tab_{int(self.pardict['snapshot']):03d}.{i}.hdf5"
            )
            with h5py.File(file_path, "r") as f:
                # test = f["Subhalo"]["SubhaloFlag"][()]
                # num_single = len(test)
                # tot.append(np.array([i, num_single]))

                if len(f["Subhalo"].keys()) > 0:
                    num_single = len(f["Subhalo"]["SubhaloFlag"][()])
                    mass_star = (
                        f["Subhalo"]["SubhaloMassType"][()][:, 4]
                        * 1e10
                        / float(self.pardict["h"])
                    )
                    sfr = f["Subhalo"]["SubhaloSFR"][()]
                    mass_all.append(mass_star)
                    sfr_all.append(sfr)
                    index += 1
                else:
                    num_single = 0

                tot.append(np.array([i, num_single, index]))

            if i % 100 == 0:
                print(f"Processed file {i}")

        tot = np.array(tot)
        num_tot = len(np.where(tot[:, 1] > 0)[0])
        print(num_tot, np.sum(tot[:, 1]))
        mass_all = np.concatenate(mass_all)
        sfr_all = np.concatenate(sfr_all)
        print("Total number of subhalos:", len(mass_all))
        index_useful = np.where((mass_all > 1e9) & (mass_all < 10**10.5))[0]
        print("Number of useful subhalos:", len(index_useful))
        ssfr_ridge = np.mean(sfr_all[index_useful] * 1e9 / (mass_all[index_useful]))
        print("sSFR ridge:", np.log10(ssfr_ridge))

        index_mass = np.where(mass_all >= 1e9)[0]
        print("Number of subhalos with mass > 1e9 Msun:", len(index_mass))
        index_sfr = np.where(
            sfr_all * 1e9 / mass_all <= 10 ** (np.log10(ssfr_ridge) - 1.0)
        )[0]
        print("Number of subhalos with sSFR < sSFR_ridge - 1 dex:", len(index_sfr))
        index_quench = np.intersect1d(index_mass, index_sfr)
        print(
            "Number of quenched subhalos with mass > 1e9 Msun and sSFR < sSFR_ridge - 1 dex:",
            len(index_quench),
        )
        np.savetxt(self.pardict["out_dir"] + "/subhalo_num_per_file.txt", tot, fmt="%d")
        return tot, np.log10(ssfr_ridge)

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

    def round_sig(self, x, p=6):
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    def load_tng50_data(
        self,
        stellar_mass_range=(1e9, 1e12),
        minimum_particles=1000,
        limit=5000,
        index_local=None,
    ):
        """
        Load TNG galaxy, stellar, and gas data via web API with comprehensive parameters

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

        if self.pardict["mode_load"] == "API":
            print(f"Loading {self.simulation} data via web API...")

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
            subhalo_loaded = False
            self.ssfr_ridge = float(self.pardict["sSFR_ridge"])
        else:
            print(f"Loading {self.simulation} data from downloaded hdf5 files...")
            # Load from local HDF5 file
            # try:
            #     index_to_hdf5_num_table = np.loadtxt(
            #         self.pardict["out_dir"] + "/subhalo_num_per_file.txt"
            #     )
            #     self.ssfr_ridge = float(self.pardict["ssfr_ridge"])
            #     print("Successfully loaded subhalo_num_per_file.txt.")
            # except Exception as e:
            #     print("Cannot find subhalo_num_per_file.txt. Generating it now...")
            #     index_to_hdf5_num_table, self.ssfr_ridge = self.find_subhalo_ID()
            #     print("Generated subhalo_num_per_file.txt.")

            index_to_hdf5_num_table = np.loadtxt(
                self.pardict["out_dir"] + "/subhalo_num_per_file.txt"
            )
            self.ssfr_ridge = float(self.pardict["sSFR_ridge"])
            print(
                "Successfully loaded subhalo_num_per_file.txt. The sSFR ridge in Gyr^-1 is ",
                self.ssfr_ridge,
            )

            # Find the hdf5 number corresponding to the given index, since some of the files do not contain subhalos.
            hdf5_num = np.where(
                (index_to_hdf5_num_table[:, 2] == index_local)
                & (index_to_hdf5_num_table[:, 1] > 0)
            )[0][0]
            self.hdf5_num = hdf5_num

            galaxy_index_below = np.sum(index_to_hdf5_num_table[:hdf5_num, 1])
            self.galaxy_index_below = galaxy_index_below

            path = (
                self.pardict["TNG_dir"]
                + f"/groups_{int(self.snapshot):03d}/fof_subhalo_tab_{int(self.snapshot):03d}"
                + "."
                + str(hdf5_num)
                + ".hdf5"
            )

            # subhalo_detail = il.groupcat.loadSubhalos(
            #     self.pardict["TNG_dir"],
            #     self.snapshot,
            #     fields=[
            #         "SubhaloPos",
            #         "SubhaloVel",
            #         "SubhaloFlag",
            #         "SubhaloMass",
            #         "SubhaloMassType",
            #         "SubhaloLen",
            #         "SubhaloLenType",
            #         "SubhaloHalfmassRad",
            #         "SubhaloHalfmassRadType",
            #         "SubhaloSFR",
            #         "SubhaloVmax",
            #         "SubhaloVelDisp",
            #         "SubhaloSpin",
            #         "SubhaloGasMetallicity",
            #         "SubhaloStarMetallicity",
            #     ],
            # )

            # total_mass_all = subhalo_detail["SubhaloMass"] * 1e10 / self.h
            # mass_type_all = subhalo_detail["SubhaloMassType"] * 1e10 / self.h
            # stellar_mass_all = mass_type_all[:, 4]

            # num_particles_all = subhalo_detail["SubhaloLen"]
            # num_particles_stars_all = subhalo_detail["SubhaloLenType"][:, 4]
            # num_particles_gas_all = subhalo_detail["SubhaloLenType"][:, 0]
            # flag_all = subhalo_detail["SubhaloFlag"]
            # # Calculate half-mass radius (R_eff) in kpc
            # half_mass_rad_kpc_type_all = (
            #     subhalo_detail["SubhaloHalfmassRadType"] * self.scale_factor / self.h
            # )
            # half_mass_rad_kpc_all = (
            #     subhalo_detail["SubhaloHalfmassRad"] * self.scale_factor / self.h
            # )
            # half_mass_rad_kpc_gas_all = half_mass_rad_kpc_type_all[:, 0]
            # half_mass_rad_kpc_dm_all = half_mass_rad_kpc_type_all[:, 1]
            # half_mass_rad_kpc_stars_all = half_mass_rad_kpc_type_all[:, 4]

            # sfr_all = subhalo_detail["SubhaloSFR"]
            # V_max_all = subhalo_detail["SubhaloVmax"]
            # vel_disp_all = subhalo_detail["SubhaloVelDisp"]
            # pos_all = subhalo_detail["SubhaloPos"] * self.scale_factor / self.h
            # vel_all = subhalo_detail["SubhaloVel"]
            # spin_all = subhalo_detail["SubhaloSpin"] / self.h

            # gas_metallicity_all = subhalo_detail[
            #     "SubhaloGasMetallicity"
            # ]  # Gas metallicity
            # star_metallicity_all = subhalo_detail[
            #     "SubhaloStarMetallicity"
            # ]  # Star metallicity

            # results = flag_all
            # subhalo_loaded = True

            with h5py.File(path, "r") as f:
                results = f["Subhalo"]["SubhaloFlag"][()]
                subhalo_detail = f["Subhalo"]
                subhalo_loaded = True

                total_mass_all = (
                    np.float64(subhalo_detail["SubhaloMass"][()]) * 1e10 / self.h
                )
                mass_type_all = (
                    np.float64(subhalo_detail["SubhaloMassType"][()]) * 1e10 / self.h
                )
                stellar_mass_all = mass_type_all[:, 4]

                num_particles_all = subhalo_detail["SubhaloLen"][()]
                num_particles_stars_all = subhalo_detail["SubhaloLenType"][()][:, 4]
                num_particles_gas_all = subhalo_detail["SubhaloLenType"][()][:, 0]
                flag_all = subhalo_detail["SubhaloFlag"][()]
                # Calculate half-mass radius (R_eff) in kpc
                half_mass_rad_kpc_type_all = (
                    np.float64(subhalo_detail["SubhaloHalfmassRadType"][()])
                    * self.scale_factor
                    / self.h
                )
                half_mass_rad_kpc_all = (
                    np.float64(subhalo_detail["SubhaloHalfmassRad"][()])
                    * self.scale_factor
                    / self.h
                )

                half_mass_rad_kpc_gas_all = half_mass_rad_kpc_type_all[:, 0]
                half_mass_rad_kpc_dm_all = half_mass_rad_kpc_type_all[:, 1]
                half_mass_rad_kpc_stars_all = half_mass_rad_kpc_type_all[:, 4]

                sfr_all = np.float64(subhalo_detail["SubhaloSFR"][()])
                V_max_all = np.float64(subhalo_detail["SubhaloVmax"][()])
                vel_disp_all = np.float64(subhalo_detail["SubhaloVelDisp"][()])

                pos_all = (
                    np.float64(subhalo_detail["SubhaloPos"][()])
                    * self.scale_factor
                    / self.h
                )
                vel_all = np.float64(subhalo_detail["SubhaloVel"][()])
                spin_all = np.float64(subhalo_detail["SubhaloSpin"][()]) / self.h

                gas_metallicity_all = np.float64(
                    subhalo_detail["SubhaloGasMetallicity"][()]
                )  # Gas metallicity
                star_metallicity_all = np.float64(
                    subhalo_detail["SubhaloStarMetallicity"][()]
                )  # Star metallicity

        # Loading in the supplementary data if provided
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
            if self.pardict["mode_load"] == "API":
                if i % 250 == 0:
                    print(f"Processing subhalo {i}/{len(results)}")
            else:
                if i % 10000 == 0:
                    print(f"Processing subhalo {i}/{len(results)}")

            # Get detailed subhalo info
            if self.pardict["mode_load"] == "API":
                subhalo_detail = self.get_api_data(
                    f"snapshots/{self.snapshot}/subhalos/{subhalo['id']}/"
                )

            if subhalo_detail and not subhalo_loaded:
                pass_mass_cut = False
                pass_num_cut = False
                pass_subhalo_flag = False
                pass_softening_cut = False
                selection_flag = False

                # Extract comprehensive properties
                subhalo_id = subhalo_detail.get("id")
                total_mass = subhalo_detail.get("mass", 0) * 1e10 / self.h
                stellar_mass = subhalo_detail.get("mass_stars", 0) * 1e10 / self.h

                num_particles = subhalo_detail.get("len", 0)
                num_particles_stars = subhalo_detail.get("len_stars", 0)
                num_particles_gas = subhalo_detail.get("len_gas", 0)
                flag = subhalo_detail.get("subhaloflag", 0)
                if subhalo_id == 0 and self.simulation == "TNG50-1":
                    flag = 0
                    print(
                        "Subhalo ID is 0, TNG50-1 API has trouble downloading cutout catalogue for this galaxy, skipping this galaxy."
                    )
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

                if self.pardict["galaxy_type"] == "spiral":
                    if (
                        half_mass_rad_kpc_stars > self.epsilon_star
                        and half_mass_rad_kpc_gas > self.epsilon_gas
                        and half_mass_rad_kpc_dm > self.epsilon_dm
                    ):
                        pass_softening_cut = True
                elif self.pardict["galaxy_type"] == "elliptical":
                    # elliptical galaxies only need to satisfy star and dark matter softening length criteria, it usually contains little gas.
                    if (
                        half_mass_rad_kpc_stars > self.epsilon_star
                        and half_mass_rad_kpc_dm > self.epsilon_dm
                    ):
                        pass_softening_cut = True
                else:
                    raise ValueError("galaxy_type must be 'spiral' or 'elliptical'")

                ssfr = (
                    (
                        subhalo_detail.get("sfr", 0) / stellar_mass * 1e9  # in Gyr^-1
                    )
                    if stellar_mass > 0
                    else 0.0
                )
                V_max = subhalo_detail.get("vmax", np.nan)
                vel_disp = subhalo_detail.get("veldisp", np.nan)

                if bool(int(self.pardict["sel_cut"])):
                    if (
                        self.pardict["sel_params"] == "ssfr"
                        or self.pardict["sel_params"] == "both"
                    ):
                        ssfr_flag = False

                        if self.pardict["galaxy_type"] == "spiral":
                            if ssfr >= float(self.pardict["ssfr_cut"]):
                                ssfr_flag = True
                        elif self.pardict["galaxy_type"] == "elliptical":
                            # if ssfr < float(self.pardict["ssfr_cut"]):
                            #     ssfr_flag = True
                            if ssfr <= 10 ** (self.ssfr_ridge - 1.0):
                                ssfr_flag = True
                        else:
                            raise ValueError(
                                "galaxy_type must be 'spiral' or 'elliptical'"
                            )

                    if (
                        self.pardict["sel_params"] == "V_max_disp"
                        or self.pardict["sel_params"] == "both"
                    ):
                        v_ratio_flag = False
                        if vel_disp == 0 or np.isnan(vel_disp) or np.isnan(V_max):
                            pass
                        else:
                            if self.pardict["galaxy_type"] == "spiral":
                                if V_max / vel_disp >= float(
                                    self.pardict["Vmax_disp_cut"]
                                ):
                                    v_ratio_flag = True
                            elif self.pardict["galaxy_type"] == "elliptical":
                                raise ValueError(
                                    "V_max/disp selection only valid for 'spiral' galaxy_type for now."
                                )
                                if V_max / vel_disp < float(
                                    self.pardict["Vmax_disp_cut"]
                                ):
                                    v_ratio_flag = True
                            else:
                                raise ValueError(
                                    "galaxy_type must be 'spiral' or 'elliptical'"
                                )

                    if self.pardict["sel_params"] == "both":
                        selection_flag = ssfr_flag and v_ratio_flag
                    elif self.pardict["sel_params"] == "ssfr":
                        selection_flag = ssfr_flag
                    elif self.pardict["sel_params"] == "V_max_disp":
                        selection_flag = v_ratio_flag

                else:
                    print(" No selection cut applied.")
                    selection_flag = True

                pass_flag = (
                    pass_mass_cut
                    and pass_num_cut
                    and pass_subhalo_flag
                    and pass_softening_cut
                    and selection_flag
                )

                # Filter by stellar mass
                if pass_flag:
                    # stellar_mass_range[0] <= stellar_mass <= stellar_mass_range[1]:
                    # Calculate additional derived properties
                    # pos = np.array(subhalo_detail.get("pos", [0, 0, 0]))
                    # vel = np.array(subhalo_detail.get("vel", [0, 0, 0]))

                    pos = np.array(
                        [
                            subhalo_detail.get("pos_x", 0),
                            subhalo_detail.get("pos_y", 0),
                            subhalo_detail.get("pos_z", 0),
                        ]
                    )
                    pos = pos * self.scale_factor / self.h

                    # Convert comoving distance to proper distance in kpc

                    vel = np.array(
                        [
                            subhalo_detail.get("vel_x", 0),
                            subhalo_detail.get("vel_y", 0),
                            subhalo_detail.get("vel_z", 0),
                        ]
                    )

                    # Distance from observer (assuming we're at origin)
                    distance_mpc = np.linalg.norm(pos) / 10**3  # Distance in kpc
                    # Find the redshift of the galaxy relative to the observer at the center of the box.
                    redshift = z_at_value(
                        self.cosmo.comoving_distance,
                        distance_mpc * u.Mpc,
                        zmin=0,
                        zmax=float(self.pardict["z_grid_lim"][1]),
                        ztol=1e-8,
                    ).value

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

                    # # Debug: Print a few examples to check
                    # if i < 5:  # Only for first few galaxies
                    #     print(f"Galaxy {subhalo_id}: mass_type = {mass_type}")
                    #     print(
                    #         f"  Gas: {gas_mass:.2e}, DM: {dm_mass:.2e}, Stars: {stellar_mass:.2e}"
                    #     )
                    #     print(f"  Total mass: {total_mass:.2e}")

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

                    galaxy_data_single = {
                        # Basic identifiers
                        "SubhaloID": np.int32(subhalo_id),
                        "SnapNum": self.snapshot,
                        # Masses (in solar masses)
                        "SubhaloLen": num_particles,
                        "SubhaloLenStars": num_particles_stars,
                        "SubhaloLenGas": num_particles_gas,
                        "SubhaloMass": total_mass,
                        "SubhaloStellarMass": stellar_mass,
                        "SubhaloGasMass": gas_mass,
                        "SubhaloDMMass": dm_mass,
                        "SubhaloBaryonicMass": baryonic_mass,
                        "GasToStellarMassRatio": gas_mass / stellar_mass
                        if stellar_mass > 0
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
                        ),  # km/s - REAL TNG data
                        "SubhaloVelDisp": subhalo_detail.get(
                            "veldisp", np.nan
                        ),  # km/s - REAL TNG data
                        # Derived properties
                        "SpecificSFR": subhalo_detail.get("sfr", 0) / stellar_mass
                        if stellar_mass > 0
                        else 0,
                        "SurfaceDensity": stellar_mass / (np.pi * half_mass_rad_kpc**2)
                        if half_mass_rad_kpc > 0
                        else 0,
                        # Additional TNG-specific properties
                        "SubhaloMassType": mass_type,
                        "SubhaloSpin": spin,
                        "SubhaloFlag": flag,
                        # Redshift and cosmological properties
                        "Redshift": redshift,
                        "LookbackTime": 0.0,  # Gyr (z=0)
                        "CosmicTime": self.age_snapshot,  # Gyr
                    }
                    # Adding supplementary data if available
                    if self.num_sub_files > 0:
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
                    self.SubhaloID_to_index.append(subhalo_id)
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
                    elif not selection_flag:
                        try:
                            message = "Did not pass selection cut based on galaxy type and selection parameters. ssfr = {:.2e} Gyr^-1, Vmax/disp = {:.2f}".format(
                                ssfr,
                                galaxy_data_single["SubhaloMaxCircVel"]
                                / galaxy_data_single["SubhaloVelDisp"],
                            )
                        except Exception as e:
                            message = "Did not pass selection cut based on galaxy type and selection parameters. ssfr = {:.2e} Gyr^-1, Vmax/disp = {:.2f}".format(
                                ssfr,
                                V_max / vel_disp,
                            )
                    print(
                        "Skipping subhalo {}: failed cuts, because {}".format(
                            subhalo_id, message
                        )
                    )

            elif subhalo_loaded:
                pass_mass_cut = False
                pass_num_cut = False
                pass_subhalo_flag = False
                pass_softening_cut = False
                selection_flag = False

                # Extract comprehensive properties
                subhalo_id = np.int32(self.galaxy_index_below + i)
                total_mass = total_mass_all[i]
                mass_type = mass_type_all[i]
                stellar_mass = stellar_mass_all[i]

                num_particles = num_particles_all[i]
                flag = flag_all[i]
                if subhalo_id == 0 and self.simulation == "TNG50-1":
                    flag = 0
                    print(
                        "Subhalo ID is 0, TNG50-1 hdf5 has trouble reading in cutout catalogue for this galaxy, skipping this galaxy."
                    )
                # Calculate half-mass radius (R_eff) in kpc
                half_mass_rad_kpc = half_mass_rad_kpc_all[i]

                half_mass_rad_kpc_gas = half_mass_rad_kpc_gas_all[i]
                half_mass_rad_kpc_dm = half_mass_rad_kpc_dm_all[i]
                half_mass_rad_kpc_stars = half_mass_rad_kpc_stars_all[i]

                if flag == 1:
                    pass_subhalo_flag = True
                if stellar_mass_range[0] <= stellar_mass <= stellar_mass_range[1]:
                    pass_mass_cut = True
                if num_particles >= minimum_particles:
                    pass_num_cut = True
                if self.pardict["galaxy_type"] == "spiral":
                    if (
                        half_mass_rad_kpc_stars > self.epsilon_star
                        and half_mass_rad_kpc_gas > self.epsilon_gas
                        and half_mass_rad_kpc_dm > self.epsilon_dm
                    ):
                        pass_softening_cut = True
                elif self.pardict["galaxy_type"] == "elliptical":
                    # elliptical galaxies only need to satisfy star and dark matter softening length criteria, it usually contains little gas.
                    if (
                        half_mass_rad_kpc_stars > self.epsilon_star
                        and half_mass_rad_kpc_dm > self.epsilon_dm
                    ):
                        pass_softening_cut = True
                else:
                    raise ValueError("galaxy_type must be 'spiral' or 'elliptical'")

                ssfr = (
                    sfr_all[i] / stellar_mass * 1e9  # in Gyr^-1
                    if stellar_mass > 0
                    else 0
                )
                V_max = V_max_all[i]
                vel_disp = vel_disp_all[i]

                if bool(int(self.pardict["sel_cut"])):
                    if (
                        self.pardict["sel_params"] == "ssfr"
                        or self.pardict["sel_params"] == "both"
                    ):
                        ssfr_flag = False
                        if self.pardict["galaxy_type"] == "spiral":
                            if ssfr >= float(self.pardict["ssfr_cut"]):
                                ssfr_flag = True
                        elif self.pardict["galaxy_type"] == "elliptical":
                            # if ssfr < float(self.pardict["ssfr_cut"]):
                            #     ssfr_flag = True
                            if ssfr <= 10 ** (self.ssfr_ridge - 1.0):
                                ssfr_flag = True
                        else:
                            raise ValueError(
                                "galaxy_type must be 'spiral' or 'elliptical'"
                            )

                    if (
                        self.pardict["sel_params"] == "V_max_disp"
                        or self.pardict["sel_params"] == "both"
                    ):
                        v_ratio_flag = False
                        if vel_disp == 0 or np.isnan(vel_disp) or np.isnan(V_max):
                            pass
                        else:
                            if self.pardict["galaxy_type"] == "spiral":
                                if V_max / vel_disp >= float(
                                    self.pardict["Vmax_disp_cut"]
                                ):
                                    v_ratio_flag = True
                            elif self.pardict["galaxy_type"] == "elliptical":
                                raise ValueError(
                                    "V_max/disp selection only valid for 'spiral' galaxy_type for now."
                                )
                                if V_max / vel_disp < float(
                                    self.pardict["Vmax_disp_cut"]
                                ):
                                    v_ratio_flag = True
                            else:
                                raise ValueError(
                                    "galaxy_type must be 'spiral' or 'elliptical'"
                                )

                    if self.pardict["sel_params"] == "both":
                        selection_flag = ssfr_flag and v_ratio_flag
                    elif self.pardict["sel_params"] == "ssfr":
                        selection_flag = ssfr_flag
                    elif self.pardict["sel_params"] == "V_max_disp":
                        selection_flag = v_ratio_flag

                else:
                    print(" No selection cut applied.")
                    selection_flag = True

                pass_flag = (
                    pass_mass_cut
                    and pass_num_cut
                    and pass_subhalo_flag
                    and pass_softening_cut
                    and selection_flag
                )

                # Filter by stellar mass
                if pass_flag:
                    # stellar_mass_range[0] <= stellar_mass <= stellar_mass_range[1]:
                    # Calculate additional derived properties
                    # pos = np.array(subhalo_detail.get("pos", [0, 0, 0]))
                    # vel = np.array(subhalo_detail.get("vel", [0, 0, 0]))

                    pos = pos_all[i]
                    # Convert comoving distance to proper distance in kpc

                    vel = vel_all[i]

                    # Distance from observer (assuming we're at origin)
                    distance_mpc = np.linalg.norm(pos) / 10**3  # Distance in mpc
                    # Find the redshift of the galaxy relative to the observer at the center of the box.
                    redshift = z_at_value(
                        self.cosmo.comoving_distance,
                        distance_mpc * u.Mpc,
                        zmin=0,
                        zmax=float(self.pardict["z_grid_lim"][1]),
                        ztol=1e-8,
                    ).value

                    # Velocity magnitude
                    velocity_magnitude = np.linalg.norm(vel)

                    # Mass ratios - Fixed extraction with debugging
                    # mass_type = subhalo_detail.get("masstype", [0] * 6)
                    mass_type = mass_type_all[i]

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

                    # # Debug: Print a few examples to check
                    # if i < 5:  # Only for first few galaxies
                    #     print(f"Galaxy {subhalo_id}: mass_type = {mass_type}")
                    #     print(
                    #         f"  Gas: {gas_mass:.2e}, DM: {dm_mass:.2e}, Stars: {stellar_mass:.2e}"
                    #     )
                    #     print(f"  Total mass: {total_mass:.2e}")

                    baryonic_mass = stellar_mass + gas_mass

                    spin = spin_all[i]

                    galaxy_data_single = {
                        # Basic identifiers
                        "SubhaloID": np.int32(subhalo_id),
                        "SnapNum": self.snapshot,
                        # Masses (in solar masses)
                        "SubhaloLen": num_particles_all[i],
                        "SubhaloLenStars": num_particles_stars_all[i],
                        "SubhaloLenGas": num_particles_gas_all[i],
                        "SubhaloMass": total_mass,
                        "SubhaloStellarMass": stellar_mass,
                        "SubhaloGasMass": gas_mass,
                        "SubhaloDMMass": dm_mass,
                        "SubhaloBaryonicMass": baryonic_mass,
                        "GasToStellarMassRatio": gas_mass / stellar_mass
                        if stellar_mass > 0
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
                        "SubhaloSFR": sfr_all[i],
                        "SubhaloGasMetallicity": gas_metallicity_all[i],
                        "SubhaloStellarMetallicity": star_metallicity_all[i],
                        "SubhaloHalfmassRad": half_mass_rad_kpc,
                        "SubhaloHalfmassRadStars": half_mass_rad_kpc_stars,
                        "SubhaloHalfmassRadGas": half_mass_rad_kpc_gas,
                        "SubhaloHalfmassRadDM": half_mass_rad_kpc_dm,
                        # Kinematic properties
                        "SubhaloMaxCircVel": V_max_all[i],  # km/s - REAL TNG data
                        "SubhaloVelDisp": vel_disp_all[i],  # km/s - REAL TNG data
                        # Derived properties
                        "SpecificSFR": sfr_all[i] / stellar_mass
                        if stellar_mass > 0
                        else 0,
                        "SurfaceDensity": stellar_mass / (np.pi * half_mass_rad_kpc**2)
                        if half_mass_rad_kpc > 0
                        else 0,
                        # Additional TNG-specific properties
                        "SubhaloMassType": mass_type,
                        "SubhaloSpin": spin,
                        "SubhaloFlag": flag,
                        # Redshift and cosmological properties
                        "Redshift": redshift,
                        "LookbackTime": 0.0,  # Gyr (z=0)
                        "CosmicTime": self.age_snapshot,  # Gyr
                    }
                    # Adding supplementary data if available
                    if self.num_sub_files > 0:
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
                    self.SubhaloID_to_index.append(subhalo_id)
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
                    elif not selection_flag:
                        try:
                            message = "Did not pass selection cut based on galaxy type and selection parameters. ssfr = {:.2e} Gyr^-1, Vmax/disp = {:.2f}".format(
                                ssfr,
                                galaxy_data_single["SubhaloMaxCircVel"]
                                / galaxy_data_single["SubhaloVelDisp"],
                            )
                        except Exception as e:
                            message = "Did not pass selection cut based on galaxy type and selection parameters. ssfr = {:.2e} Gyr^-1, Vmax/disp = {:.2f}".format(
                                ssfr,
                                V_max / vel_disp,
                            )

                    if int(self.pardict["verbose"]) == 1:
                        print(
                            "Skipping subhalo {}: failed cuts, because {}".format(
                                subhalo_id, message
                            )
                        )

        self.galaxy_data = galaxy_data
        print(f"Loaded {len(self.galaxy_data)} galaxies")

        # Load particle data with more realistic approach
        self._load_particle_data_summary()
        print("Finished loading particle data summaries for all galaxies.")

        if len(self.galaxy_data) == 0:
            print(
                "No galaxies satisfy the selection criteria in the hdf5 file number:"
                + str(hdf5_num)
            )
        else:
            # For custom sfh, we don't need to grid the redshift. Assuming all particles in the same galaxy have the same redshift.
            # redshift_list = []
            # ages_list = []
            for i in range(len(self.galaxy_data)):
                self.calculate_and_store_sfh(i)
                keyword = "kinematics_only"
                #     redshift_list.append(self.galaxy_data[i]["Redshift"])
                #     ages_list.append(
                #         np.min(self.age_snapshot - self.stellar_data[i]["Stellar_age"])
                #     )
                self.galaxy_data[i]["min_stellar_age"] = np.min(
                    self.age_snapshot - self.stellar_data[i]["Stellar_age"]
                )
                self.galaxy_data[i]["max_stellar_age"] = np.max(
                    self.age_snapshot - self.stellar_data[i]["Stellar_age"]
                )

            # out_list = np.vstack((np.array(redshift_list), np.array(ages_list))).T

            if self.pardict["mode_load"] == "local":
                keyword += "_" + str(hdf5_num)

            # np.savetxt(
            #     f"{self.out_dir}/redshift_age_list_" + keyword + ".txt",
            #     np.array(out_list),
            # )

            np.savez(
                f"{self.out_dir}/stellar_data_" + keyword + ".npz",
                np.array(self.stellar_data, dtype=object),
                allow_pickle=True,
            )
            np.savez(
                f"{self.out_dir}/gas_data_" + keyword + ".npz",
                np.array(self.gas_data, dtype=object),
                allow_pickle=True,
            )
            np.savez(
                f"{self.out_dir}/subhalo_data_" + keyword + ".npz",
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
        index_fail_cut = []

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
                # "ParticleIDs",
                "Coordinates",
                "Masses",
                "GFM_Metallicity",
                "Velocities",
                "GFM_Metals",
                "NeutralHydrogenAbundance",
                "Density",
                "StarFormationRate",
                "InternalEnergy",
                "ElectronAbundance",
            }

            star_fields = {
                # "ParticleIDs",
                "Coordinates",
                "Masses",
                "GFM_Metallicity",
                "GFM_StellarFormationTime",
                "Velocities",
            }

            print("Start loading particle data for galaxy ID:", galaxy_id)

            gas_data_single = self.get_subhalo_cutout(
                galaxy_id, fields=gas_fields, particle_type="gas", index=index
            )

            stellar_data_single = self.get_subhalo_cutout(
                galaxy_id, fields=star_fields, particle_type="stars", index=index
            )

            if stellar_data_single is None:
                pass_cutout_selection = False
                message = "Error loading star particle data with illustris python. Skipping this galaxy. Check snapshot hdf5 files integrity"
            else:
                pass_cutout_selection, message = self.cutout_selection(
                    data_stellar=stellar_data_single, data_subhalo=galaxy, stage=1
                )

                calculate_inclination_ellipticity = True

            if pass_cutout_selection is False:
                print(
                    f"Skipping galaxy ID {galaxy_id} due to cutout selection criteria: {message}"
                )
                index_fail_cut.append(index)
                continue
            elif (
                np.min(
                    np.linalg.norm(
                        stellar_data_single["Coordinates"] - galaxy["SubhaloPos"],
                        axis=1,
                    )
                )
                > 2.0 * galaxy["SubhaloHalfmassRadStars"]
            ):
                print(
                    f"Skipping galaxy ID {galaxy_id} because all star particles are beyond 2*R_half from galaxy center. This is unphysical"
                )
                index_fail_cut.append(index)
                continue
            elif gas_data_single["Coordinates"] is not None:
                if (
                    np.min(
                        np.linalg.norm(
                            gas_data_single["Coordinates"] - galaxy["SubhaloPos"],
                            axis=1,
                        )
                    )
                    > 2.0 * galaxy["SubhaloHalfmassRadStars"]
                ):
                    print(
                        f"Galaxy ID {galaxy_id} has all gas particles are beyond 2*R_half from galaxy center. Will not calculate inclination and ellipticity for gas component."
                    )
                    calculate_inclination_ellipticity = False

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
                self.galaxy_data[index],
                data_stellar=stellar_data_single,
                method="star",
            )

            if (
                gas_data_single["Coordinates"] is not None
                and calculate_inclination_ellipticity
            ):
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
                    data_gas=gas_data_single,
                    data_subhalo=self.galaxy_data[index],
                    method="gas",
                )
            else:
                pos_align_gas = None
                vel_align_gas = None
                vel_2D_align_gas = None
                pos_incline_gas = None
                vel_incline_gas = None
                vel_2D_incline_gas = None
                inclination_gas = None
                position_angle_gas = None
                axial_ratio_gas = None
                ellipticity_gas = None
                orientation_angle_gas = None

            # gas_SF_index = np.where(gas_data_single["StarFormationRate"] > 0)[0]
            # gas_data_SF = {
            #     key: gas_data_single[key][gas_SF_index] for key in gas_data_single
            # }

            # print(
            #     "There are ",
            #     len(gas_SF_index),
            #     " star forming gas particles in subhalo",
            # )

            # (
            #     pos_align_SF_gas,
            #     vel_align_SF_gas,
            #     vel_2D_align_SF_gas,
            #     pos_incline_SF_gas,
            #     vel_incline_SF_gas,
            #     vel_2D_incline_SF_gas,
            #     inclination_SF_gas,
            #     position_angle_SF_gas,
            #     axial_ratio_SF_gas,
            #     ellipticity_SF_gas,
            #     orientation_angle_SF_gas,
            # ) = self.find_inclination_ellipticity(
            #     data_gas=gas_data_SF,
            #     data_subhalo=self.galaxy_data[index],
            #     method="gas",
            # )

            # Only saving the plotting output when plotting the diagonstic figures
            if (
                int(self.pardict["plot_vmap"]) == 1
                or int(self.pardict["plot_pos_view"]) == 1
                or int(self.pardict["plot_phase"]) == 1
            ):
                self.galaxy_data[index]["Pos_align_gas"] = pos_align_gas
                self.galaxy_data[index]["Vel_align_gas"] = vel_align_gas
                self.galaxy_data[index]["Vel_2D_align_gas"] = vel_2D_align_gas
                self.galaxy_data[index]["Pos_incline_gas"] = pos_incline_gas
                self.galaxy_data[index]["Vel_incline_gas"] = vel_incline_gas
                self.galaxy_data[index]["Vel_2D_incline_gas"] = vel_2D_incline_gas

                self.galaxy_data[index]["Pos_align_star"] = pos_align_star
                self.galaxy_data[index]["Vel_align_star"] = vel_align_star
                self.galaxy_data[index]["Vel_2D_align_star"] = vel_2D_align_star
                self.galaxy_data[index]["Pos_incline_star"] = pos_incline_star
                self.galaxy_data[index]["Vel_incline_star"] = vel_incline_star
                self.galaxy_data[index]["Vel_2D_incline_star"] = vel_2D_incline_star
                # self.galaxy_data[index]["Pos_align__SF_gas"] = pos_align_SF_gas
                # self.galaxy_data[index]["Vel_align__SF_gas"] = vel_align_SF_gas
                # self.galaxy_data[index]["Vel_2D_align__SF_gas"] = vel_2D_align_SF_gas
                # self.galaxy_data[index]["Pos_incline__SF_gas"] = pos_incline_SF_gas
                # self.galaxy_data[index]["Vel_incline__SF_gas"] = vel_incline_SF_gas
                # self.galaxy_data[index]["Vel_2D_incline__SF_gas"] = (
                #     vel_2D_incline_SF_gas
                # )
            # self.galaxy_data[index]["Inclination_SF_gas"] = inclination_SF_gas
            # self.galaxy_data[index]["Position_Angle_SF_gas"] = position_angle_SF_gas
            # self.galaxy_data[index]["Mass_Axial_Ratio_SF_gas"] = axial_ratio_SF_gas
            # self.galaxy_data[index]["Mass_Ellipticity_SF_gas"] = ellipticity_SF_gas
            # self.galaxy_data[index]["Mass_Orientation_Angle_SF_gas"] = (
            #     orientation_angle_SF_gas
            # )

            self.galaxy_data[index]["Inclination_star"] = inclination_star
            self.galaxy_data[index]["Position_Angle_star"] = position_angle_star
            self.galaxy_data[index]["Mass_Axial_Ratio_star"] = axial_ratio_star
            self.galaxy_data[index]["Mass_Ellipticity_star"] = ellipticity_star
            self.galaxy_data[index]["Mass_Orientation_Angle_star"] = (
                orientation_angle_star
            )
            self.galaxy_data[index]["Inclination_gas"] = inclination_gas
            self.galaxy_data[index]["Position_Angle_gas"] = position_angle_gas
            self.galaxy_data[index]["Mass_Axial_Ratio_gas"] = axial_ratio_gas
            self.galaxy_data[index]["Mass_Ellipticity_gas"] = ellipticity_gas
            self.galaxy_data[index]["Mass_Orientation_Angle_gas"] = (
                orientation_angle_gas
            )

            gas_data.append(gas_data_single)

            stellar_data.append(stellar_data_single)

        if len(index_fail_cut) > 0:
            print(
                f"{len(index_fail_cut)} galaxies did not pass the cutout selection and will be removed."
            )

            # Cut out all galaxies that did not pass the cutout selection
            self.galaxy_data = [
                g for i, g in enumerate(self.galaxy_data) if i not in index_fail_cut
            ]
            self.stellar_data = stellar_data
            self.gas_data = gas_data
            print(
                f"Created {len(self.stellar_data)} stellar,  {len(self.gas_data)} gas particles, and {len(self.galaxy_data)} subhalo data entries after cutout selection."
            )
        elif len(self.galaxy_data) > 0:
            self.stellar_data = stellar_data
            self.gas_data = gas_data
            print(
                f"Created {len(self.stellar_data)} stellar,  {len(self.gas_data)} gas particles, and {len(self.galaxy_data)} subhalo data entries."
            )

    def cutout_selection(
        self,
        data_subhalo,
        data_stellar=None,
        stage=1,
        band=None,
        ssr_deV=None,
        ssr_exp=None,
        sersic_unc=None,
    ):
        """
        Apply selection criteria to particle cutout data

        Parameters:
        -----------
        data_stellar : dict
            Stellar particle data
        data_subhalo : dict
            Subhalo data
        stage : int
            Stage of selection (1 or 2). 1 for initial cutout with avilable TNG properties. 2 for final cutout with derived properties from galaxev.
        ssr_sersic : float
            Sersic fit sum of the squared residuals
        ssr_deV : float
            de Vaucouleurs fit sum of the squared residuals
        ssr_exp : float
            Exponential fit sum of the squared residuals
        sersic_unc : float
            Uncertainty in Sersic index fit
        Returns:
        --------
        bool : Whether the galaxy passes the selection criteria
        """

        pass_cutout_selection = False
        if stage == 1:
            # The portion of total kinetic energy in ordered rotation must be greater than 0.5
            try:
                keppa_rotate = kinematics.KappaRotation(
                    data_stellar["Masses"],
                    data_stellar["Coordinates"]
                    - data_subhalo["SubhaloPos"],  # position relative to subhalo center
                    data_stellar["Velocities"]
                    - data_subhalo[
                        "SubhaloVel"
                    ],  # velocity relative to subhalo velocity
                    ez_range=2.0
                    * data_subhalo[
                        "SubhaloHalfmassRadStars"
                    ],  # Only use particles within 2*R_half
                )
                if self.pardict["galaxy_type"] == "spiral":
                    if keppa_rotate > float(self.pardict["kappa_lim"]):
                        pass_cutout_selection = True
                        message = None
                        print("Passing cutout selection, keppa_rotate =", keppa_rotate)
                    else:
                        message = "keppa_rotate = {:.2f}".format(keppa_rotate)
                elif self.pardict["galaxy_type"] == "elliptical":
                    # velocity = data_stellar["Velocities"] - data_subhalo["SubhaloVel"]
                    # position = data_stellar["Coordinates"] - data_subhalo["SubhaloPos"]
                    # index = np.where(
                    #     np.linalg.norm(position, axis=1)
                    #     < data_subhalo["SubhaloHalfmassRadStars"]
                    # )[0]
                    # vel_disp = np.std(velocity[index], axis=0)
                    # if (
                    #     np.any(vel_disp > float(self.pardict["v_disp_range"][0]))
                    #     and np.any(vel_disp < float(self.pardict["v_disp_range"][1]))
                    #     and keppa_rotate < float(self.pardict["kappa_lim"])
                    # ):
                    #     pass_cutout_selection = True
                    #     message = None
                    #     print(
                    #         "passing cutout selection, vel_disp =, keppa_rotate =",
                    #         vel_disp,
                    #         keppa_rotate,
                    #     )
                    # else:
                    #     message = "vel_disp = {}".format(vel_disp)
                    if keppa_rotate <= float(self.pardict["kappa_lim"]):
                        pass_cutout_selection = True
                        message = None
                        print("Passing cutout selection, keppa_rotate =", keppa_rotate)
                    else:
                        message = "keppa_rotate = {:.2f}".format(keppa_rotate)
            except Exception as e:
                message = "Error calculating keppa_rotate, probably because all star particles are over 2*R_half from galaxy center. Min position is {:.2f} kpc and the stellar half mass radius is {:.2f} kpc".format(
                    np.min(
                        np.linalg.norm(
                            data_stellar["Coordinates"] - data_subhalo["SubhaloPos"],
                            axis=1,
                        )
                    ),
                    data_subhalo["SubhaloHalfmassRadStars"],
                )

        elif stage == 2:
            # Cutting galaxies based on the Sersic index derived from GALAXEV photometry
            if self.pardict["galaxy_type"] == "spiral":
                sersic_band = data_subhalo["n_fit_" + band]
                # if sersic_band <= float(self.pardict["sersic_lim"]):
                # If the upper uncertainty bound of the sersic index is less than the limit, we accept the galaxy
                if sersic_band + sersic_unc <= float(self.pardict["sersic_lim"]):
                    pass_cutout_selection = True
                    message = None
                # In case the uncertainty is too big, if the ssr of exponential fit is better than de Vaucouleurs fit, we also accept the galaxy follow Xu et al. 2017
                elif ssr_exp < ssr_deV:
                    pass_cutout_selection = True
                    message = None
                else:
                    message = "Sersic_n = {:.2f}".format(data_subhalo["n_fit_" + band])

            elif self.pardict["galaxy_type"] == "elliptical":
                sersic_band = data_subhalo["n_fit_" + band]
                # if sersic_band > float(self.pardict["sersic_lim"]):
                if sersic_band - sersic_unc > float(self.pardict["sersic_lim"]):
                    pass_cutout_selection = True
                    message = None
                elif ssr_deV < ssr_exp:
                    pass_cutout_selection = True
                    message = None
                else:
                    message = "Sersic_n = {:.2f}".format(data_subhalo["n_fit_" + band])
        else:
            raise ValueError("stage must be 1 or 2")

        return pass_cutout_selection, message

    # Loading the saved npz files
    def load_saved_data(self, keyword="kinematics_only"):
        """
        Load previously saved stellar, gas, and subhalo data from .npz files

        Parameters:
        -----------
        keyword : str
            Keyword to identify the specific dataset (e.g., 'full_photometry' or 'kinematics_only')
        """
        print("Loading saved data from .npz files...")

        stellar_path = f"{self.out_dir}/stellar_data_{keyword}.npz"
        gas_path = f"{self.out_dir}/gas_data_{keyword}.npz"
        subhalo_path = f"{self.out_dir}/subhalo_data_{keyword}.npz"

        if self.pardict["mode_load"] == "local":
            out_dir = self.pardict["out_dir"]
            index_to_hdf5_num_table = np.loadtxt(out_dir + "/subhalo_num_per_file.txt")
            n_catalogue_hdf5 = np.int32(index_to_hdf5_num_table[-1, -1] + 1)

            tot_job, reminder = divmod(
                n_catalogue_hdf5, int(self.pardict["index_per_job"])
            )
            if reminder > 0 and self.job_id == tot_job:
                start_index = self.job_id * int(self.pardict["index_per_job"])
                end_index = n_catalogue_hdf5
            else:
                start_index = self.job_id * int(self.pardict["index_per_job"])
                end_index = (self.job_id + 1) * int(self.pardict["index_per_job"])

            print(start_index, end_index, n_catalogue_hdf5)

            self.start_index = start_index
            self.end_index = end_index

            # Check if the files exist
            if (
                not os.path.exists(stellar_path)
                or not os.path.exists(gas_path)
                or not os.path.exists(subhalo_path)
            ):
                print("One or more data files do not exist, generating them now...")
                self.combined_sfh_output_files(
                    job_id=self.job_id,
                    index_to_hdf5_num_table=index_to_hdf5_num_table,
                    out_dir=out_dir,
                )
            else:
                self.stellar_data = np.load(stellar_path, allow_pickle=True)[
                    "arr_0"
                ].tolist()
                self.gas_data = np.load(gas_path, allow_pickle=True)["arr_0"].tolist()
                self.galaxy_data = np.load(subhalo_path, allow_pickle=True)[
                    "arr_0"
                ].tolist()
        else:
            self.stellar_data = np.load(stellar_path, allow_pickle=True)[
                "arr_0"
            ].tolist()
            self.gas_data = np.load(gas_path, allow_pickle=True)["arr_0"].tolist()
            self.galaxy_data = np.load(subhalo_path, allow_pickle=True)[
                "arr_0"
            ].tolist()

        # self.galaxy_data, self.stellar_data, self.gas_data = np.load(
        #     "../plots/output_54.npz", allow_pickle=True
        # )["arr_0"].tolist()

        # self.galaxy_data = [self.galaxy_data]
        # self.stellar_data = [self.stellar_data]
        # self.gas_data = [self.gas_data]

        print(
            f"Loaded {len(self.galaxy_data)} galaxies, {len(self.stellar_data)} stellar datasets, and {len(self.gas_data)} gas datasets."
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
            warnings.warn(
                "Using uniform weights for axial ratio calculation.", UserWarning
            )

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

        return axial_ratio, orientation_angle, galaxy_position - galaxy_center

    def find_inclination_ellipticity(
        self, data_subhalo, data_stellar=None, data_gas=None, method="star"
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

        # Uncomment the following lines if you want to use star particles to calculate the angular momentum axis
        if method == "star":
            # Find and remove wind particles since they are not stars

            # wind_particle_index = np.where(
            #     data_stellar["GFM_StellarFormationTime"] <= 0.0
            # )[0]
            # print("There are ", len(wind_particle_index), "wind particles in subhalo")
            # star_pos = np.delete(
            #     data_stellar["Coordinates"], wind_particle_index, axis=0
            # )
            # star_vel = np.delete(
            #     data_stellar["Velocities"], wind_particle_index, axis=0
            # )
            # mass_star = np.delete(data_stellar["Masses"], wind_particle_index, axis=0)
            star_pos = data_stellar["Coordinates"]
            star_vel = data_stellar["Velocities"]
            mass_star = data_stellar["Masses"]

            pos = star_pos - data_subhalo["SubhaloPos"]
            vel = star_vel - data_subhalo["SubhaloVel"]

            # Calculate the angular momentum axis
            axis = kinematics.AngularMomentum(
                mass_star,
                pos,
                vel,
                return_ji=False,
                # Restrict to twice the stellar half-mass radius
                range=2.0 * data_subhalo["SubhaloHalfmassRadStars"],
            )
            pos_lim = float(self.pardict["pos_lim_star"])

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
            pos_lim = float(self.pardict["pos_lim_gas"])

        # Using the user defined line-of-sight (LOS) direction
        if self.los_axis == 2:
            los_vector = np.array([0, 0, 1])
            vector_y = np.array([0, 1, 0])
            vector_x = np.array([1, 0, 0])
            los_direction = "z"
            inclination_direction = "x"
        elif self.los_axis == 1:
            los_vector = np.array([0, 1, 0])
            vector_y = np.array([0, 0, 1])
            vector_x = np.array([1, 0, 0])
            los_direction = "y"
            inclination_direction = "x"
        elif self.los_axis == 0:
            los_vector = np.array([1, 0, 0])
            vector_y = np.array([0, 0, 1])
            vector_x = np.array([0, 1, 0])
            los_direction = "x"
            inclination_direction = "y"
        else:
            raise ValueError("los_axis must be 0 (x), 1 (y), or 2 (z)")

        inclination = np.arccos(np.dot(axis, los_vector)) * 180.0 / np.pi
        position_angle = (
            np.arctan2(np.dot(axis, vector_y), np.dot(axis, vector_x)) * 180.0 / np.pi
        )

        print(
            "Inclination angle between angular momentum axis and z-axis:", inclination
        )
        print(
            "Position angle between projection of angular momentum axis onto x-y plane and x-axis:",
            position_angle,
        )

        # Align the galaxy along the angular momentum axis
        Rmat = kinematics.RotationMatrix(los_vector, axis)
        pos_norm = Rmat.apply(pos)
        vel_norm = Rmat.apply(vel)

        # # The angular momentum axis is now the z-axis, so we can define face-on and edge-on views
        # pos_all["inclination_" + str(0)] = np.array([pos_norm[:, 0], pos_norm[:, 1]]).T

        v_los = np.dot(vel_norm, los_vector)  # line-of-sight velocity along z-axis

        # if self.los_axis == 2:
        #     pos_norm_los = pos_norm[:, :2]
        # elif self.los_axis == 1:
        #     pos_norm_los = pos_norm[:, [0, 2]]
        # elif self.los_axis == 0:
        #     pos_norm_los = pos_norm[:, 1:]
        pos_norm_los = self.project_coordinates(pos=pos_norm)

        stat, _, xedges, yedges = self.make_vmap_hist(
            pos_norm_los,
            v_los,
            x_range=[-pos_lim, pos_lim],
            y_range=[-pos_lim, pos_lim],
            nx=200,
            ny=200,
        )

        if bool(int(self.pardict["plot_pos_view"])):
            self.plot_edgeon_faceon_view(
                pos_norm, pos_lim, data_subhalo["SubhaloID"], method=method
            )
        if bool(int(self.pardict["plot_phase"])):
            self.plot_phase_space(
                pos_norm,
                vel_norm,
                pos_lim,
                pos_lim,
                data_subhalo["SubhaloID"],
                method=method,
            )

        pos_align = pos_norm
        vel_align = vel_norm
        vel_2D_align = stat

        # Rotate the galaxy with the calculated inclination and position angles.
        R_incl = Rotation.from_euler(inclination_direction, inclination, degrees=True)
        R_pa = Rotation.from_euler(los_direction, position_angle, degrees=True)
        R_total = R_pa * R_incl

        pos_rot = R_total.apply(pos_norm)
        vel_rot = R_total.apply(vel_norm)

        v_los = np.dot(vel_rot, los_vector)  # line-of-sight velocity along z-axis

        # if self.los_axis == 2:
        #     pos_rot_los = pos_rot[:, :2]
        # elif self.los_axis == 1:
        #     pos_rot_los = pos_rot[:, [0, 2]]
        # elif self.los_axis == 0:
        #     pos_rot_los = pos_rot[:, 1:]

        pos_rot_los = self.project_coordinates(pos=pos_rot)

        stat, _, xedges, yedges = self.make_vmap_hist(
            pos_rot_los,
            v_los,
            x_range=[-pos_lim, pos_lim],
            y_range=[-pos_lim, pos_lim],
            nx=200,
            ny=200,
        )
        if bool(int(self.pardict["plot_vmap"])):
            self.plot_vmap(
                data_subhalo["SubhaloID"], stat, xedges, yedges, method=method
            )

        pos_incline = pos_rot
        vel_incline = vel_rot
        vel_2D_incline = stat

        if method == "star":
            axial_ratio, orientation_angle, _ = self.find_axial_ratio_and_orientation(
                pos_rot_los,
                weight=mass_star,
            )
        else:
            axial_ratio, orientation_angle, _ = self.find_axial_ratio_and_orientation(
                pos_rot_los,
                weight=mass_gas,
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

    # Find the projected coordinates given the line-of-sight axis
    def project_coordinates(self, particles=None, pos=None):
        """
        Project particle coordinates based on the line-of-sight axis
        Parameters:
        -----------
        particles : dict
            Particle data containing 'Coordinates' key
        Returns:
        --------
        numpy.ndarray
            Projected particle coordinates
        """
        # --- project coordinates based on line-of-sight axis ---
        if particles is not None:
            if self.los_axis == 0:
                particle_coords = particles["Coordinates"][:, [1, 2]]
            elif self.los_axis == 1:
                particle_coords = particles["Coordinates"][:, [0, 2]]
            else:  # self.los_axis == 2
                particle_coords = particles["Coordinates"][:, [0, 1]]
        elif pos is not None:
            if self.los_axis == 0:
                particle_coords = pos[:, [1, 2]]
            elif self.los_axis == 1:
                particle_coords = pos[:, [0, 2]]
            else:  # self.los_axis == 2
                particle_coords = pos[:, [0, 1]]

        else:
            raise ValueError(
                "Either provide a dictionary of particles or a position array."
            )

        return particle_coords

    # Numerically find star formation history from stellar particles
    def star_formation_history(self, stellar_particles, n_bins=45, norm=True):
        """
        Calculate star formation history (SFH) for a galaxy

        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        n_bins : int
            Number of time bins for SFH. Should be odd to use Simpson's rule.
        norm : bool
            Whether to normalize the SFR to total mass formed

        Returns:
        --------
        tuple : (time_bins, sfr_values)
            time_bins : array of time bin centers in Gyr
            sfr_values : array of SFR values in solar masses per year
        """

        if len(stellar_particles) == 0:
            return None, None

        # Convert formation times to age since the Big Bang
        ages = stellar_particles["Stellar_age"]  # Gyr

        formation_times = self.age_snapshot - ages  # Gyr

        masses = stellar_particles["Masses"]

        # Create time bins
        sfr_values = np.zeros(n_bins)

        # Calculating star formation history with histogram
        hist, bin_edges = np.histogram(formation_times, bins=n_bins, weights=masses)
        sfr_values = hist / (bin_edges[1:] - bin_edges[:-1])  # Msun/Gyr

        # Calculate bin centers
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Check if n_bins is odd for Simpson's rule
        if n_bins % 2 == 0:
            print("n_bins is even, increasing by 1 for Simpson's rule")
            n_bins += 1  # Make it odd
            hist, bin_edges = np.histogram(formation_times, bins=n_bins, weights=masses)
            sfr_values = hist / (bin_edges[1:] - bin_edges[:-1])  # Msun/Gyr
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if norm is False:
            return bin_centers, sfr_values
        else:
            # Total mass formed
            mass_form = simpson(sfr_values, bin_centers)
            # The normalized SFR, normalized to 1 solar mass formed
            norm_sfr = sfr_values / mass_form if mass_form > 0 else None

            return bin_centers, norm_sfr

    # Padding the SFH array with zeros at the beginning and the end if necessary
    def pad_sfh(self, time_bins, sfr_values):
        t_1 = (
            0.0  # Gyr, assuming star formation starts at the beginning of the universe
        )
        t_start = time_bins[0]
        t_2 = self.age_snapshot  # Gyr, the age of the universe at the snapshot
        t_end = time_bins[-1]

        start_pedding = False
        end_pedding = False
        if t_start > t_1:
            start_pedding = True
        if t_end < t_2:
            end_pedding = True

        if start_pedding and not end_pedding:
            # Pad the beginning with zeros
            binwidth = time_bins[1] - time_bins[0]

            t_fit_new = np.arange(t_1, t_2, binwidth)
            t_fit_start_index = np.where(t_fit_new < t_start)[0]
            t_fit_start = t_fit_new[t_fit_start_index]
            if t_fit_start[-1] + binwidth / 2.0 > t_start:
                t_fit_start = t_fit_start[:-1]
            t_fit = np.concatenate((t_fit_start, time_bins), axis=0)
            sfr_fit_start = np.zeros_like(t_fit_start)
            sfr_fit = np.concatenate((sfr_fit_start, sfr_values), axis=0)
            # t_fit = np.arange(t_1, t_2, binwidth)
            # sfr_fit = np.zeros_like(t_fit)
            # sfr_fit[n_pad : n_pad + len(sfr_values)] = sfr_values

        elif end_pedding and not start_pedding:
            # Pad the end with zeros
            binwidth = time_bins[1] - time_bins[0]
            t_fit_new = np.arange(t_1, t_2, binwidth)
            t_fit_end_index = np.where(t_fit_new > t_end)[0]
            t_fit_end = t_fit_new[t_fit_end_index]
            if t_fit_end[0] - binwidth / 2.0 < t_end:
                t_fit_end = t_fit_end[1:]
            t_fit = np.concatenate((time_bins, t_fit_end), axis=0)
            sfr_fit_end = np.zeros_like(t_fit_end)
            sfr_fit = np.concatenate((sfr_values, sfr_fit_end), axis=0)

            # sfr_fit = np.zeros_like(t_fit)
            # sfr_fit[: len(sfr_values)] = sfr_values
        elif start_pedding and end_pedding:
            # Pad both the beginning and the end with zeros
            binwidth = time_bins[1] - time_bins[0]
            t_fit_new = np.arange(t_1, t_2, binwidth)

            t_fit_start_index = np.where(t_fit_new < t_start)[0]
            t_fit_end_index = np.where(t_fit_new > t_end)[0]

            t_fit_start = t_fit_new[t_fit_start_index]
            t_fit_end = t_fit_new[t_fit_end_index]

            if t_fit_start[-1] + binwidth / 2.0 > t_start:
                t_fit_start = t_fit_start[:-1]
            if t_fit_end[0] - binwidth / 2.0 < t_end:
                t_fit_end = t_fit_end[1:]

            t_fit = np.concatenate((t_fit_start, time_bins, t_fit_end), axis=0)

            sfr_fit_start = np.zeros_like(t_fit_start)
            sfr_fit_end = np.zeros_like(t_fit_end)

            sfr_fit = np.concatenate((sfr_fit_start, sfr_values, sfr_fit_end), axis=0)

            # sfr_fit = np.zeros_like(t_fit)
            # n_pad_start = int((t_start - t_1) / binwidth)
            # sfr_fit[n_pad_start : n_pad_start + len(sfr_values)] = sfr_values

        return t_fit, sfr_fit

    # Fit the SFH to find the star formation timescale tau with the exponential or the delayed tau model
    def fit_sfh_tau(self, time_bins, sfr_values, model="tau"):
        """
        Fit the star formation history (SFH) to find the star formation timescale tau

        Parameters:
        -----------
        time_bins : array
            Array of time bin centers in Gyr
        sfr_values : array
            Normalized SFR values (sfr divided by total mass formed) in Gyr^-1
        model : str
            SFH model to fit ('exponential' or 'delayed')

        Returns:
        --------
        float : Fitted tau value in Gyr
        """

        # The extra factors of 1/tau are absorbed into the amplitude A during fitting. The input t is in Gyr.
        def exp_sfh(t, tau, t_1, t_2):
            # For the normalized sfh, the amplitude
            A = 1.0 / (np.exp(-t_1 / tau) - np.exp(-t_2 / tau))
            return A * np.exp(-t / tau) / tau

        def delayed_sfh(t, tau, t_1, t_2):
            # For the normalized sfh, the amplitude
            A = 1.0 / (
                (1.0 + t_1 / tau) * np.exp(-t_1 / tau)
                - (1.0 + t_2 / tau) * np.exp(-t_2 / tau)
            )
            return A * t * np.exp(-t / tau) / tau**2

        # t_1 = time_bins[0]
        # t_2 = time_bins[-1]
        t_1 = (
            0.0  # Gyr, assuming star formation starts at the beginning of the universe
        )
        t_2 = self.age_snapshot  # Gyr, the age of the universe at the snapshot

        time_bins, sfr_values = self.pad_sfh(time_bins, sfr_values)

        # Choose model. Use partial to fix t_1 and t_2 based on input data
        if model == "tau":
            sfh_model = partial(exp_sfh, t_1=t_1, t_2=t_2)
        elif model == "delayed":
            sfh_model = partial(delayed_sfh, t_1=t_1, t_2=t_2)
        else:
            raise ValueError("Invalid model. Choose 'tau' (exponential) or 'delayed'.")

        # Initial guess for tau and amplitude A
        initial_guess = [1.0]  # tau in Gyr

        # Fit the model to the data
        try:
            popt, _ = curve_fit(
                sfh_model,
                time_bins,
                sfr_values,
                p0=initial_guess,
                # The boundary is fixed to the input tau grid limits
                bounds=(
                    float(self.pardict["tau_grid_lim"][0]),
                    float(self.pardict["tau_grid_lim"][1]),
                ),
            )
            fitted_tau = popt[0]
            print(f"Fitted tau: {fitted_tau} Gyr")

            plt.plot(time_bins, sfr_values, "o", label="Data")
            plt.plot(time_bins, sfh_model(time_bins, *popt), label="Fit")
            plt.legend()
            plt.savefig("sfh_fit_tau_" + str(fitted_tau) + ".png", dpi=300)
            plt.close()

            return fitted_tau
        except RuntimeError:
            print("Fit did not converge.")
            return None

    # Reading in the grids of absolute magnitudes from galaxev models. This is for modelling sfh with a fitting function.
    def _load_galaxev_models(self):
        # Loading in the galaxev model grids
        work_dir = self.pardict["work_dir"]
        redshift = np.linspace(
            np.float64(self.pardict["z_grid_lim"][0]),
            np.float64(self.pardict["z_grid_lim"][1]),
            np.int32(self.pardict["nz_bin"]),
        )  # redshift grid
        bands = self.pardict["bands"]  # photometric bands

        for i in range(len(redshift)):
            grid_file = work_dir + "/vst_mags_grid_z=%6.4f_sfh_%s_fast.hdf5" % (
                redshift[i],
                self.pardict["sfh_model"],
            )
            with h5py.File(grid_file, "r") as f:
                # The metallicity grid
                Z_grid = f["Z_grid"][:]
                # The age grid
                age_grid = f["age_grid"][:]
                # The grid for the number of different charateristic scale for the SFH models
                tau_grid = f["tau_grid"][:]

                # Loading magnitude grids for different bands
                for band in bands:
                    mag_grid = f["%s_mag" % (band)][:]
                    lum_grid = f["%s_lum" % (band)][:]
                    nu_filter = f["%s_nu_filter" % (band)][:]
                    response = f["%s_response" % (band)][:]

                    setattr(
                        self,
                        "mag_grid_" + band + "_redshift_bin_" + str(i),
                        np.zeros(
                            (
                                len(redshift),
                                Z_grid.shape[0],
                                tau_grid.shape[0],
                                age_grid.shape[0],
                            )
                        ),
                    )

                    setattr(
                        self,
                        "lum_grid_" + band + "_redshift_bin_" + str(i),
                        np.zeros(
                            (
                                len(redshift),
                                Z_grid.shape[0],
                                tau_grid.shape[0],
                                age_grid.shape[0],
                                len(nu_filter),
                            )
                        ),
                    )

                    setattr(
                        self,
                        "nu_filter_grid_" + band + "_redshift_bin_" + str(i),
                        np.zeros((len(redshift), len(nu_filter))),
                    )
                    setattr(
                        self,
                        "response_grid_" + band + "_redshift_bin_" + str(i),
                        np.zeros((len(redshift), len(response))),
                    )
                    getattr(self, "mag_grid_" + band + "_redshift_bin_" + str(i))[
                        i, :, :, :
                    ] = mag_grid
                    getattr(self, "lum_grid_" + band + "_redshift_bin_" + str(i))[
                        i, :, :, :, :
                    ] = lum_grid
                    getattr(self, "nu_filter_grid_" + band + "_redshift_bin_" + str(i))[
                        i, :
                    ] = nu_filter
                    getattr(self, "response_grid_" + band + "_redshift_bin_" + str(i))[
                        i, :
                    ] = response

        # Building a grid interpolator for each band
        self.mag_interpolators = {}
        self.lum_interpolators = {}
        self.nu_filter_interpolators = {}
        self.response_interpolators = {}
        for band in bands:
            self.mag_interpolators[band] = RegularGridInterpolator(
                (redshift, Z_grid, tau_grid, age_grid),
                getattr(self, "mag_grid_" + band + "_redshift_bin_" + str(i)),
                bounds_error=True,
                fill_value=None,
            )
            self.lum_interpolators[band] = RegularGridInterpolator(
                (redshift, Z_grid, tau_grid, age_grid),
                getattr(self, "lum_grid_" + band + "_redshift_bin_" + str(i)),
                bounds_error=True,
                fill_value=None,
            )
            self.nu_filter_interpolators[band] = interp1d(
                redshift,
                getattr(self, "nu_filter_grid_" + band + "_redshift_bin_" + str(i)),
                kind="linear",
                axis=0,
                bounds_error=True,
                fill_value=None,
            )
            self.response_interpolators[band] = interp1d(
                redshift,
                getattr(self, "response_grid_" + band + "_redshift_bin_" + str(i)),
                kind="linear",
                axis=0,
                bounds_error=True,
                fill_value=None,
            )

    # Reading in grids of of luminosities with a custom SFH
    def _load_galaxev_models_custom(self):
        # Loading in the galaxev model grids
        work_dir = self.pardict["work_dir"]
        bands = self.pardict["bands"]  # photometric bands

        self.nu_filter_interpolators = {}
        self.response_interpolators = {}
        self.mag_interpolators = {}
        self.lum_interpolators = {}

        for i in range(len(self.galaxy_data)):
            grid_file = (
                work_dir
                + "/vst_mags_grid_z=%6.4f_sfh_%s_galaxy_%d_fast.hdf5"
                % (
                    self.galaxy_data[i]["Redshift"],
                    self.pardict["sfh_model"],
                    self.galaxy_data[i]["SubhaloID"],
                )
            )

            print(
                "Loading custom SFH grid from file: "
                + grid_file
                + " for galaxy "
                + str(i)
            )
            with h5py.File(grid_file, "r") as f:
                # The metallicity grid
                Z_grid = f["Z_grid"][:]
                # The age grid
                age_grid = f["age_grid"][:]

                for band in bands:
                    nu_filter = f["%s_nu_filter" % (band)][:]
                    response = f["%s_response" % (band)][:]
                    mag_grid = f["%s_mag" % (band)][:]
                    lum_grid = f["%s_lum" % (band)][:]

                    print("Analyzing band:", band)

                    setattr(
                        self,
                        "mag_grid_" + band + "_custom_sfh_" + str(i),
                        np.zeros(
                            (
                                Z_grid.shape[0],
                                age_grid.shape[0],
                            )
                        ),
                    )

                    setattr(
                        self,
                        "lum_grid_" + band + "_custom_sfh_" + str(i),
                        np.zeros(
                            (
                                Z_grid.shape[0],
                                age_grid.shape[0],
                                len(nu_filter),
                            )
                        ),
                    )

                    getattr(self, "mag_grid_" + band + "_custom_sfh_" + str(i))[
                        :, :
                    ] = mag_grid
                    getattr(self, "lum_grid_" + band + "_custom_sfh_" + str(i))[
                        :, :, :
                    ] = lum_grid

                    self.nu_filter_interpolators[band + "_custom_sfh_" + str(i)] = (
                        nu_filter
                    )
                    self.response_interpolators[band + "_custom_sfh_" + str(i)] = (
                        response
                    )

                    self.mag_interpolators[band + "_custom_sfh_" + str(i)] = (
                        RegularGridInterpolator(
                            (Z_grid, age_grid),
                            getattr(self, "mag_grid_" + band + "_custom_sfh_" + str(i)),
                            bounds_error=True,
                            fill_value=None,
                            method="cubic",
                        )
                    )
                    self.lum_interpolators[band + "_custom_sfh_" + str(i)] = (
                        RegularGridInterpolator(
                            (Z_grid, age_grid),
                            getattr(self, "lum_grid_" + band + "_custom_sfh_" + str(i)),
                            bounds_error=True,
                            fill_value=None,
                            method="cubic",
                        )
                    )

    # Find all particles within the same subhalo
    def find_particles_in_galaxy(self, index, method="stars"):
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
            particles = self.gas_data[index]
        elif method == "stars":
            particles = self.stellar_data[index]
        else:
            # Combine both gas and stellar particles
            particles = pd.concat([self.gas_data[index], self.stellar_data[index]])

        if len(particles) == 0:
            return np.nan

        # galaxy_id = self.SubhaloID_to_index[index]
        # print(
        #     f"Found {len(particles['ParticleIDs'])} {method} particles in galaxy ID {galaxy_id}"
        # )

        return particles

    # Calculate star formation histroy and store them
    def calculate_and_store_sfh(self, galaxy_id):
        """Calculate and store the star formation history for a galaxy
        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        """

        # Get stellar particles for this galaxy
        stellar_particles = self.find_particles_in_galaxy(galaxy_id, method="stars")

        age_bins, sfr_values = self.star_formation_history(
            stellar_particles, n_bins=int(self.pardict["n_sfh_bins"]), norm=True
        )

        age_bins, sfr_values = self.pad_sfh(age_bins, sfr_values)

        # if len(age_bins) >= 255:
        #     age_bins_new = np.linspace(age_bins[0], age_bins[-1], 254)
        #     sfr_values_new = np.interp(age_bins_new, age_bins, sfr_values)
        #     age_bins = age_bins_new
        #     sfr_values = sfr_values_new
        #     print(
        #         "Age bins and SFR values have been resampled to 254 bins for galaxy id {}. Otherwise galaxev will complain the sed array is too long.".format(
        #             self.galaxy_data[galaxy_id]["SubhaloID"]
        #         )
        #     )

        # if age_bins is None or sfr_values is None:
        #     self.galaxy_data[stellar_particles_index]["SFH_age_bins"] = np.nan
        #     self.galaxy_data[stellar_particles_index]["SFH_sfr"] = np.nan
        # else:
        #     self.galaxy_data[stellar_particles_index]["SFH_age_bins"] = age_bins
        #     self.galaxy_data[stellar_particles_index]["SFH_sfr"] = sfr_values

        # Convert age bins to years and SFR to Msun/yr
        age_bins_yr = age_bins * 1e9  # years
        sfr_values_msun_yr = sfr_values / 1e9  # Msun/yr

        # Output SFH as an ASCII file for galaxev input
        sfh_output_file = (
            self.out_dir
            + "/galaxy_"
            + str(self.galaxy_data[galaxy_id]["SubhaloID"])
            + "_sfh.txt"
        )

        with open(sfh_output_file, "w") as f:
            f.write("# Age (yr)    SFR (Msun/yr)\n")
            for age, sfr in zip(age_bins_yr, sfr_values_msun_yr):
                f.write(f"{age:.6e}    {sfr:.6e}\n")

    # Using galaxev models to estimate the absolute magnitude from stellar mass and age
    def _estimate_magnitude_luminosity(self, galaxy_id):
        """Estimate the absolute magnitude and luminosity for a galaxy using galaxev models
        Parameters:
        -----------
        galaxy_id : int
            Galaxy identifier
        """

        # TNG does not return the characteristic star formation timescale tau directly. We will fit it to a particular model

        # Get stellar particles for this galaxy
        stellar_particles = self.find_particles_in_galaxy(galaxy_id, method="stars")

        # age_bins, norm_sfr_values = self.star_formation_history(
        #     stellar_particles, n_bins=int(self.pardict["n_sfh_bins"])
        # )
        # if age_bins is None or norm_sfr_values is None:
        #     return np.nan

        # Calculate dust attenuation based on Xu et al 2017 model
        N_H_cm2, x_edges, y_edges = self.hydrogen_column_density(galaxy_id)  # cm^-2

        masses = stellar_particles["Masses"]
        stellar_ages = stellar_particles["Stellar_age"]
        metallicities_org = stellar_particles["GFM_Metallicity"]
        ages = (
            self.age_snapshot - stellar_ages
        )  # Gyr. The time when the stars formed. Consistent with the age grid in galaxev

        galaxev_metalicity_limit = [1e-04, 0.1]
        n_out_of_bounds = np.where(
            (metallicities_org < galaxev_metalicity_limit[0])
            | (metallicities_org > galaxev_metalicity_limit[1])
        )[0]
        if len(n_out_of_bounds) > 0:
            print(
                f"Warning: {len(n_out_of_bounds)} stellar particles out of {len(metallicities_org)} have metallicities out of the galaxev model bounds. They will be set to the nearest bound."
            )
            metallicities = np.clip(
                metallicities_org,
                np.nextafter(galaxev_metalicity_limit[0], np.inf),
                np.nextafter(galaxev_metalicity_limit[1], -np.inf),
            )  # The nextafter is to avoid hitting the exact bound which may cause issues with the interpolator

        def find_magnitude_luminosity_in_band(
            nu_filter,
            response,
            apparent_magnitudes_particles,
            N_H_cm2,
            galaxy_id,
            x_edges,
            y_edges,
            stellar_particles,
            luminosities_density_particles,
            band,
        ):
            wavelength_microns = self.csol * 1e8 / nu_filter / 1e4  # microns

            # Calculating the optical depth without scattering using eq.2 of Xu et al. 2017
            tau_lambda_a = self.DG2000_optical_depth(
                wavelength_microns,
                N_H_cm2,
                self.galaxy_data[galaxy_id]["SubhaloGasMetallicity"],
                self.galaxy_data[galaxy_id]["Redshift"],
            )

            tau_lambda = self.CSK1994_scatter(wavelength_microns, tau_lambda_a)

            # --- project coordinates based on line-of-sight axis ---
            # if self.los_axis == 0:
            #     stellar_coords = stellar_particles["Coordinates"][:, [1, 2]]
            # elif self.los_axis == 1:
            #     stellar_coords = stellar_particles["Coordinates"][:, [0, 2]]
            # else:  # self.los_axis == 2
            #     stellar_coords = stellar_particles["Coordinates"][:, [0, 1]]
            stellar_coords = self.project_coordinates(particles=stellar_particles)

            # Finding which bin each stellar particle falls into
            x_indices = (
                np.digitize(stellar_coords[:, 0], x_edges) - 1
            )  # -1 to convert to 0-based index
            y_indices = (
                np.digitize(stellar_coords[:, 1], y_edges) - 1
            )  # -1 to convert to 0-based index

            # Ensure indices are within bounds
            valid_mask = (
                (x_indices >= 0)
                & (x_indices < N_H_cm2.shape[0])
                & (y_indices >= 0)
                & (y_indices < N_H_cm2.shape[1])
            )

            # All particles outside the bounds will have zero attenuation (tau_lambda = 0)
            tau_lambda_particles = np.zeros((len(valid_mask), len(wavelength_microns)))
            # tau_lambda_particles[valid_mask] = tau_lambda[x_indices[valid_mask]][
            #     y_indices[valid_mask]
            # ]
            tmp = tau_lambda[x_indices[valid_mask], y_indices[valid_mask]]
            tau_lambda_particles[valid_mask, :] = tmp

            dusted_luminosity_density_particles = (
                luminosities_density_particles
                * np.divide(
                    -np.expm1(-tau_lambda_particles),
                    tau_lambda_particles,
                    out=np.ones_like(tau_lambda_particles),
                    where=tau_lambda_particles != 0,
                )
            )  # eq. 1 of Xu et al. 2017

            def luminosity_in_band(nu_filter, response, luminosity_density):
                # Integrate luminosity density over the filter response to get luminosity in the band

                # integrand = splrep(nu_filter, luminosity_density * response)
                # L_band = splint(integrand, nu_filter[0], nu_filter[-1])

                L_band = simpson(luminosity_density * response, nu_filter, axis=-1)
                return L_band

            self.stellar_data[galaxy_id]["AB_apparent_magnitude_" + band] = (
                apparent_magnitudes_particles
            )

            self.stellar_data[galaxy_id]["Raw_Luminosity_" + band] = luminosity_in_band(
                nu_filter, response, luminosities_density_particles
            )

            self.stellar_data[galaxy_id]["Dusted_Luminosity_" + band] = (
                luminosity_in_band(
                    nu_filter, response, dusted_luminosity_density_particles
                )
            )

            # Total luminosity and magnitude for the galaxy
            total_luminosity_dusted = np.nansum(
                self.stellar_data[galaxy_id]["Dusted_Luminosity_" + band]
            )
            total_luminosity_raw = np.nansum(
                self.stellar_data[galaxy_id]["Raw_Luminosity_" + band]
            )
            self.galaxy_data[galaxy_id]["Raw_Luminosity_" + band] = total_luminosity_raw
            self.galaxy_data[galaxy_id]["Dusted_Luminosity_" + band] = (
                total_luminosity_dusted
            )

            print(
                "Total raw luminosity in band", band, ":", total_luminosity_raw, "erg/s"
            )
            print(
                "Total dusted luminosity in band",
                band,
                ":",
                total_luminosity_dusted,
                "erg/s",
            )

        if self.pardict["sfh_model"] == "custom":
            for band in self.pardict["bands"]:
                apparent_magnitudes_particles = (
                    self.mag_interpolators[band + "_custom_sfh_" + str(galaxy_id)](
                        np.array(
                            [
                                metallicities,
                                ages,
                            ]
                        ).T
                    )
                    - 2.5 * np.log10(masses)
                )  # The output from galaxev is per solar mass, so we convert to the correct magnitude after normalized by the mass

                luminosities_density_particles = (
                    self.lum_interpolators[band + "_custom_sfh_" + str(galaxy_id)](
                        np.array(
                            [
                                metallicities,
                                ages,
                            ]
                        ).T
                    )
                    * masses[:, None]
                )  # The output from galaxev is per solar mass, so we convert to the correct luminosity density after multiplying by the mass

                nu_filter = self.nu_filter_interpolators[
                    band + "_custom_sfh_" + str(galaxy_id)
                ]
                response = self.response_interpolators[
                    band + "_custom_sfh_" + str(galaxy_id)
                ]

                find_magnitude_luminosity_in_band(
                    nu_filter,
                    response,
                    apparent_magnitudes_particles,
                    N_H_cm2,
                    galaxy_id,
                    x_edges,
                    y_edges,
                    stellar_particles,
                    luminosities_density_particles,
                    band,
                )

        else:
            age_bins, norm_sfr_values = np.loadtxt(
                self.out_dir + "/galaxy_" + str(galaxy_id) + "_sfh.txt"
            ).T

            if age_bins is None or norm_sfr_values is None:
                return np.nan

            # Fit the SFH to find the characteristic timescale tau

            fitted_tau = self.fit_sfh_tau(
                age_bins, norm_sfr_values, model=self.pardict["sfh_model"]
            )

            if fitted_tau is None:
                return np.nan

            # Same SFH for all particles within the same galaxy
            tau_all = np.ones_like(masses) * fitted_tau
            # Assuming all particles are at the same redshift since the size of the galaxy is negligible compared to cosmological scales
            redshifts = self.galaxy_data[self.galaxy_data["SubhaloID"] == galaxy_id][
                "Redshift"
            ].values[0] * np.ones_like(masses)

            for band in self.pardict["bands"]:
                apparent_magnitudes_particles = (
                    self.mag_interpolators[band](
                        np.array(
                            [
                                redshifts,
                                metallicities,
                                tau_all,
                                ages,
                            ]
                        ).T
                    )
                    - 2.5 * np.log10(masses)
                )  # The output from galaxev is per solar mass, so we convert to the correct magnitude after normalized by the mass

                luminosities_density_particles = (
                    self.lum_interpolators[band](
                        np.array(
                            [
                                redshifts,
                                metallicities,
                                tau_all,
                                ages,
                            ]
                        ).T
                    )
                    * masses
                )  # The output from galaxev is per solar mass, so we convert to the correct luminosity density after multiplying by the mass

                nu_filter = self.nu_filter_interpolators[band](redshifts)
                response = self.response_interpolators[band](redshifts)

                find_magnitude_luminosity_in_band(
                    nu_filter,
                    response,
                    apparent_magnitudes_particles,
                    N_H_cm2,
                    galaxy_id,
                    x_edges,
                    y_edges,
                    stellar_particles,
                    luminosities_density_particles,
                    band,
                )

    def hydrogen_column_density(self, galaxy_id):
        """
        Calculate hydrogen column density N_H (cm⁻²) within a given number of grid cells that cover the galaxy.
        Assuming the line-of-sight is along the z-axis. Will update it later to abitrary direction

        Parameters
        ----------
        galaxy_id : int
            Galaxy identifier
        Returns
        -------
        N_H_cm2 : float
            Hydrogen column density in cm⁻².
        """
        # Getting the stellar particles within the same galaxy
        gas_particles = self.find_particles_in_galaxy(galaxy_id, method="gas")

        # galaxy_data_single = self.galaxy_data[
        #     self.galaxy_data["SubhaloID"] == galaxy_id
        # ]

        galaxy_data_single = self.galaxy_data[galaxy_id]

        R_eff = galaxy_data_single["SubhaloHalfmassRadStars"]  # kpc
        max_r = float(self.pardict["r_grid"])  # in units of R_eff
        Ngrid = int(
            self.pardict["n_grid_project"]
        )  # number of grid cells along each projected axis
        galaxy_pos = galaxy_data_single["SubhaloPos"]  # kpc

        # --- define grid boundaries ---
        grid_size = max_r * R_eff  # kpc

        x_edges = np.linspace(
            galaxy_pos[0] - grid_size, galaxy_pos[0] + grid_size, Ngrid + 1
        )
        y_edges = np.linspace(
            galaxy_pos[1] - grid_size, galaxy_pos[1] + grid_size, Ngrid + 1
        )

        if (
            gas_particles["Coordinates"] is None
            and self.pardict["galaxy_type"] == "elliptical"
        ):
            print(f"Galaxy ID {galaxy_id} has no gas particles.")
            N_H_cm2 = np.zeros((Ngrid, Ngrid))
        else:
            gas_coords = gas_particles["Coordinates"]  # kpc
            gas_masses = gas_particles["Masses"]  # Msun
            GFM_Metals = gas_particles["GFM_Metals"][
                :, 0
            ]  # total hydrogen mass fraction, hydrogen is the first element
            NeutroHydrogenAbundance = gas_particles[
                "NeutralHydrogenAbundance"
            ]  # fraction of neutral hydrogen
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

            # --- project hydrogen mass onto the 2D grid ---
            # weighted by neutral hydrogen fraction
            gas_mass_hist, _, _ = np.histogram2d(
                gas_coords[:, 0],
                gas_coords[:, 1],
                bins=[x_edges, y_edges],
                weights=gas_masses * GFM_Metals * NeutroHydrogenAbundance,  # Msun
            )

            # --- compute per-pixel hydrogen mass in grams ---
            hydrogen_mass_g = gas_mass_hist * self.Msun_to_g  # g per pixel

            # --- pixel area in cm² ---
            dx = (x_edges[-1] - x_edges[0]) / Ngrid  # kpc
            dy = (y_edges[-1] - y_edges[0]) / Ngrid  # kpc
            area_cm2 = np.float64(dx * dy) * np.float64(self.kpc_to_cm) ** 2  # cm²

            # --- compute number column density ---
            N_H_cm2 = hydrogen_mass_g / (area_cm2 * self.H_mass)  # cm⁻²

        return N_H_cm2, x_edges, y_edges

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

        if np.any((x < 0.3) | (x > 10)):
            warnings.warn(
                "Wavelength out of range for CCM89 extinction law (0.1 - 3.3 microns)."
            )

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
        mask_uv_high = x[mask_uv] >= 5.9
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
        self,
        wavelength_microns,
        N_H_cm2,
        metallicity,
        redshift,
        Z_sun=0.02,
        beta=-0.5,
        Rv=3.1,
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
            A_Lambda_over_Av[None, None, :]
            * N_H_cm2[:, :, None]
            / (2.1e21)
            * (1.0 + redshift) ** beta
            * (metallicity / Z_sun) ** s
        )

        return lambda_tau_no_scatter

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

        # Calculating the albedo ω(λ) using the Calzetti et al. (1994) empirical fit
        omega_Lambda = np.zeros_like(wavelength_angstroms)
        mask1 = (wavelength_angstroms >= 1000) & (wavelength_angstroms <= 3460)
        y = np.log10(wavelength_angstroms[mask1])
        omega_Lambda[mask1] = 0.43 + 0.366 * (1.0 - np.exp(-((y - 3.0) ** 2) / 0.2))
        mask2 = (wavelength_angstroms > 3460) & (wavelength_angstroms <= 7000)
        y = np.log10(wavelength_angstroms[mask2])
        omega_Lambda[mask2] = -0.48 * y + 2.41

        # Table of omega_lambda from Natta & Panagia 1984 for wavelength between 0.7 to 4.48 microns

        if np.any((wavelength_angstroms > 7000) & (wavelength_angstroms <= 44800)):
            warnings.warn(
                "Wavelength out of range for CSK1994 scattering model (0.1 - 0.7 microns). Using Natta & Panagia 1984 table values for longer wavelengths."
            )
            omega_table_wavelengths = np.array(
                [0.7, 0.9, 1.25, 1.65, 2.2, 3.6, 4.48]
            )  # microns

            omega_table_values = np.array([0.56, 0.50, 0.37, 0.28, 0.22, 0.054, 0.0])

            omega_interp = interp1d(
                omega_table_wavelengths,
                omega_table_values,
                kind="cubic",
                bounds_error=True,
                fill_value=None,
            )

            mask4 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 44800)
            omega_Lambda[mask4] = omega_interp(
                wavelength_angstroms[mask4] / 1e4
            )  # Convert back to microns for interpolation
        elif np.any((wavelength_angstroms > 44800) & (wavelength_angstroms < 1000)):
            warnings.warn(
                "No valid model for wavelength > 4.48 microns or < 0.1 microns in CSK1994 scattering model. Setting albedo to zero."
            )

        # Calculating the weighting factor h(λ) that accounts for the anistropy in scattering
        mask3 = (wavelength_angstroms >= 1200) & (wavelength_angstroms <= 7000)
        y = np.log10(wavelength_angstroms[mask3])
        h_Lambda[mask3] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)

        if np.any((wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)):
            warnings.warn(
                "Applying extrapolation for h(λ) beyond 0.7 microns. Bruzual et al 1988 has a table for g(λ) up to 1.8 microns, the functional form seems to hold approximately."
            )

            mask5 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)
            y = np.log10(wavelength_angstroms[mask5])
            h_Lambda[mask5] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)
        elif np.any(wavelength_angstroms > 18000):
            warnings.warn(
                "No valid model for h(λ) beyond 1.8 microns in CSK1994 scattering model. Setting h(λ) to one."
            )
            mask6 = wavelength_angstroms > 18000
            h_Lambda[mask6] = 1.0
        elif np.any(wavelength_angstroms < 1200):
            warnings.warn(
                "No valid model for h(λ) below 0.12 microns in CSK1994 scattering model. Setting h(λ) to zero."
            )

        # Correcting the optical depth for scattering effects
        lambda_tau_scatter = (
            h_Lambda * np.sqrt(1 - omega_Lambda) + (1 - h_Lambda) * (1 - omega_Lambda)
        )[None, None, :] * lambda_tau_no_scatter

        return lambda_tau_scatter

    def direct_effective_radius_surface_brightness_sersic_index(
        self,
        galaxy_id,
        band,
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

        print(
            np.min(particles["Coordinates"][:, 0]),
            np.max(particles["Coordinates"][:, 0]),
        )
        print(
            np.min(particles["Coordinates"][:, 1]),
            np.max(particles["Coordinates"][:, 1]),
        )
        print(
            np.min(particles["Coordinates"][:, 2]),
            np.max(particles["Coordinates"][:, 2]),
        )

        # # --- project coordinates based on line-of-sight axis ---
        # if self.los_axis == 0:
        #     particle_coords = particles["Coordinates"][:, [1, 2]]
        # elif self.los_axis == 1:
        #     particle_coords = particles["Coordinates"][:, [0, 2]]
        # else:  # self.los_axis == 2
        #     particle_coords = particles["Coordinates"][:, [0, 1]]
        particle_coords = self.project_coordinates(particles=particles)

        axial_ratios, orientation_angles, rel_pos = (
            self.find_axial_ratio_and_orientation(
                particle_coords, weight=particles["Dusted_Luminosity_" + band]
            )
        )

        dx, dy = rel_pos[:, 0], rel_pos[:, 1]

        # Rotate coordinates to align with major/minor axes. python expect radians, so convert degrees to radians
        cos_phi, sin_phi = (
            np.cos(orientation_angles * np.pi / 180.0),
            np.sin(orientation_angles * np.pi / 180.0),
        )

        x_rot = dx * cos_phi + dy * sin_phi  # along major axis
        y_rot = -dx * sin_phi + dy * cos_phi  # along minor axis

        # Compute elliptical radius for each particle
        q = axial_ratios
        if q <= 0:
            raise ValueError("Invalid axis ratio (b/a <= 0).")
        # Elliptical radius
        r_ell = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

        r_geo = r_ell * np.sqrt(q)  # approximate circularized radius

        # Apply radial limit
        within_limit = r_geo <= float(self.pardict["r_lim"])
        r_geo = r_geo[within_limit]
        observed_luminosity = particles["Dusted_Luminosity_" + band][within_limit]

        # Sort by elliptical radius
        idx = np.argsort(r_geo)
        r_sorted = r_geo[idx]
        L_sorted = observed_luminosity[idx]

        # Compute cumulative luminosity profile
        Lcum = np.cumsum(L_sorted)
        Lhalf = 0.5 * np.sum(observed_luminosity)

        # Find radius enclosing half the light (semi-major axis a_e)

        if q <= 0.01:
            print("This galaxy has a very elongated shape, cannot compute a_e.")
            print(
                "The axial ratio (b/a), orientation angle, min, max of x_rot, min, max of y_rot, min, max of r_ell:",
                q,
                orientation_angles,
                np.min(x_rot),
                np.max(x_rot),
                np.min(y_rot),
                np.max(y_rot),
                np.min(r_ell),
                np.max(r_ell),
            )
            return galaxy_id
        else:
            a_e = np.interp(Lhalf, Lcum, r_sorted)

        b_e = q * a_e

        # Geometric mean, the direct effective radius in Xu+2017
        R_e = np.sqrt(a_e * b_e)

        # using the Sersic profile fitting method from Xu+2017 to estimate Sersic index
        R = r_sorted  # approximate circularized radius

        # Calculate surface brightness profile
        r_bins = np.linspace(
            0,
            float(self.pardict["r_lim"]),
            int(self.pardict["n_grid_project"]) * 10 + 1,
        )
        surface_brightness = np.zeros(int(self.pardict["n_grid_project"]) * 10)

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

        print("Fitting Sersic profile to surface brightness data...")

        # Fit Sersic profile to surface brightness data to estimate Sersic index
        def sersic_profile(r, I_e_log, R_e, n, y_scale=None):
            """
            Sersic profile function

            r : radius
            I_e_log : log10 of surface brightness at effective radius
            R_e : effective radius
            n : Sersic index
            Returns: surface brightness at radius r"""
            b_n = find_bn(n)

            I_e = 10**I_e_log  # Convert log10(I_e) back to I_e
            # Using log scale to improve numerical stability
            return I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1)) / y_scale

        # Fit the de Vaucouleurs profile (n=4) to estimate R_e and I_e
        def de_vaucouleurs_profile(r, I_e_log, R_e, y_scale=None):
            """
            de Vaucouleurs profile function (Sersic n=4)

            r : radius
            I_e_log : log10 of surface brightness at effective radius
            R_e : effective radius
            Returns: surface brightness at radius r"""

            I_e = 10**I_e_log  # Convert log10(I_e) back to I_e

            # Using log scale to improve numerical stability
            return I_e * np.exp(-7.669 * ((r / R_e) ** (1 / 4) - 1)) / y_scale

        # Fit the exponential profile (n=1) to estimate R_e and I_e
        def exponential_profile(r, I_e_log, R_e, y_scale=None):
            """
            Exponential profile function (Sersic n=1)

            r : radius
            I_e_log : log10 of surface brightness at effective radius
            R_e : effective radius
            Returns: surface brightness at radius r"""

            I_e = 10**I_e_log  # Convert log10(I_e) back to I_e

            # Using log scale to improve numerical stability
            return I_e * np.exp(-1.678 * ((r / R_e) - 1)) / y_scale

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

        def plot_surface_brightness_profile(
            r_fit,
            surface_brightness,
            popt_sersic,
            popt_deV,
            popt_exp,
            galaxy_id,
            band,
        ):
            """Plot the surface brightness profile and fitted models for visual inspection"""

            plt.figure(figsize=(8, 6))
            plt.scatter(
                r_fit / popt_sersic[1],
                surface_brightness,
                label="Data",
                color="black",
                s=10,
                alpha=0.7,
            )
            r_plot = np.linspace(0, np.max(r_fit), 500)
            plt.plot(
                r_plot / popt_sersic[1],
                sersic_profile(r_plot, *popt_sersic, y_scale=1.0),
                label="Sersic Fit, n = {:.2f}".format(popt_sersic[2]),
                color="red",
            )
            plt.plot(
                r_plot / popt_sersic[1],
                de_vaucouleurs_profile(r_plot, *popt_deV, y_scale=1.0),
                label="de Vaucouleurs Fit",
                color="blue",
                linestyle="--",
            )
            plt.plot(
                r_plot / popt_sersic[1],
                exponential_profile(r_plot, *popt_exp, y_scale=1.0),
                label="Exponential Fit",
                color="green",
                linestyle=":",
            )
            plt.xlabel("Radius (kpc)")
            plt.ylabel("Surface Brightness (erg/s/kpc^2)")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(0.1, 2.0)
            plt.title(f"Galaxy ID {galaxy_id} - Band {band}")
            plt.legend()
            plt.savefig(
                f"{self.out_dir}/galaxy_{galaxy_id}_surface_brightness_profile_{band}.png"
            )
            plt.close()

        # # Compare the chi-squared values between the de Vaucouleurs and the exponential fits
        # def compare_chi_squared(
        #     popt_deV, popt_exp, r_fit, surface_brightness, valid_fit
        # ):
        #     """Compare chi-squared values of different profile fits
        #     popt_deV : array
        #         Best-fit parameters from de Vaucouleurs fit
        #     popt_exp : array
        #         Best-fit parameters from exponential fit
        #     r_fit : array
        #         Radii for fitting
        #     surface_brightness : array
        #         Surface brightness data
        #     valid_fit : array
        #         Boolean array indicating valid fit points
        #     Returns: bool
        #         True if de Vaucouleurs fit is better, False otherwise"""

        #     elliptical_galaxy = False
        #     y_fit_de_vaucouleurs = de_vaucouleurs_profile(r_fit[valid_fit], *popt_deV)
        #     y_fit_exponential = exponential_profile(r_fit[valid_fit], *popt_exp)
        #     chi_squared_de_vaucouleurs = np.sum(
        #         (surface_brightness[valid_fit] - y_fit_de_vaucouleurs) ** 2
        #     )
        #     chi_squared_exponential = np.sum(
        #         (surface_brightness[valid_fit] - y_fit_exponential) ** 2
        #     )

        #     if chi_squared_de_vaucouleurs < chi_squared_exponential:
        #         elliptical_galaxy = True
        #     return elliptical_galaxy

        # Prepare data for fitting
        r_fit = 0.5 * (r_bins[:-1] + r_bins[1:])  # bin centers

        # Follow Xu+2017 and fit only the binned radial profile between 0.05*R_e and 3.0*R_e
        fit_min = 0.05 * R_e
        fit_max = 3.0 * R_e
        valid_fit = (r_fit >= fit_min) & (r_fit <= fit_max) & (surface_brightness > 0)

        print(
            f"Fitting range: {fit_min:.3f} to {fit_max:.3f} kpc, with length {len(r_fit[valid_fit])} with orignal length {len(r_fit)}"
        )

        # Find the sersic index n, effective radius R_e, and surface brightness I_e by fitting the Sersic profile

        # Use the median surface brightness as the scaling factor to improve numerical stability
        y_scale = np.median(surface_brightness[valid_fit])

        sersic_profile_partial = partial(sersic_profile, y_scale=y_scale)
        bestfit, cov = curve_fit(
            sersic_profile_partial,
            r_fit[valid_fit],
            surface_brightness[valid_fit] / y_scale,
            p0=[np.log10(np.max(surface_brightness) * 0.5), R_e, 2.0],
            bounds=(
                [0, 0, 0.1],
                [
                    np.log10(np.max(surface_brightness)),
                    float(self.pardict["r_lim"]),
                    10.0,
                ],
                # [np.inf, np.inf, np.inf],
            ),  # Setting the bounds for Sersic index between 0.1 and 10, effective radius between 0 and r_lim, surface brightness positive
        )

        sersic_residuals = (
            surface_brightness[valid_fit]
            - sersic_profile_partial(r_fit[valid_fit], *bestfit) * y_scale
        )
        ssr_sersic = np.sum(sersic_residuals**2)

        # Find the best-fit parameters for de Vaucouleurs (n=4) and exponential (n=1) profiles
        de_vaucouleurs_profile_partial = partial(
            de_vaucouleurs_profile, y_scale=y_scale
        )
        bestfit_deV, _ = curve_fit(
            de_vaucouleurs_profile_partial,
            r_fit[valid_fit],
            surface_brightness[valid_fit] / y_scale,
            p0=[np.log10(np.max(surface_brightness) * 0.5), R_e],
            bounds=(
                [0, 0],
                [np.log10(np.max(surface_brightness)), float(self.pardict["r_lim"])],
            ),
        )

        de_vaucouleurs_residuals = (
            surface_brightness[valid_fit]
            - de_vaucouleurs_profile_partial(r_fit[valid_fit], *bestfit_deV) * y_scale
        )
        ssr_deV = np.sum(de_vaucouleurs_residuals**2)

        exponential_profile_partial = partial(exponential_profile, y_scale=y_scale)
        bestfit_exp, _ = curve_fit(
            exponential_profile_partial,
            r_fit[valid_fit],
            surface_brightness[valid_fit] / y_scale,
            p0=[np.log10(np.max(surface_brightness) * 0.5), R_e],
            bounds=(
                [0, 0],
                [np.log10(np.max(surface_brightness)), float(self.pardict["r_lim"])],
            ),
        )

        exponential_residuals = (
            surface_brightness[valid_fit]
            - exponential_profile_partial(r_fit[valid_fit], *bestfit_exp) * y_scale
        )
        ssr_exp = np.sum(exponential_residuals**2)

        # # Determine if the galaxy is elliptical based on chi-squared comparison
        # is_elliptical = compare_chi_squared(
        #     bestfit_deV, bestfit_exp, r_fit, surface_brightness, valid_fit
        # )

        # The best-fit parameters of surface brightness at the effective radius, effective radius, and Sersic index
        I_e_log_fit, R_e_fit, n_fit = bestfit
        I_e_fit = 10**I_e_log_fit  # Convert log10(I_e) back to I_e

        # Plot the surface brightness profile and the Sersic fit for visual inspection
        # plt.figure()
        # plt.scatter(r_fit, surface_brightness, label="Data", color="blue")
        # r_plot = np.linspace(0, float(self.pardict["r_lim"]), 100)
        # plt.plot(
        #     r_plot,
        #     sersic_profile(r_plot, *bestfit),
        #     label=f"Sersic Fit (n={n_fit:.2f})",
        #     color="red",
        # )
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.xlabel("Radius")
        # plt.ylabel("Surface Brightness")
        # plt.title(f"Galaxy {galaxy_id} Surface Brightness Profile in {band} band")
        # plt.legend()
        # plt.savefig(f"galaxy_{galaxy_id}_surface_brightness_{band}.png")
        # plt.close()

        surface_brightness_mean = np.sum(L_sorted[np.where(r_sorted < R_e_fit)[0]]) / (
            np.pi * q * R_e_fit**2
        )

        # Calculate the "model" luminosity with the analytical formula in Xu+2017. The model is calculated within 7 times the effective radius.
        # b_n_fit = find_bn(n_fit)
        # L_model = (
        #     I_e_fit
        #     * np.exp(b_n_fit)
        #     * R_e_fit**2
        #     * 2
        #     * np.pi
        #     * n_fit
        #     / (b_n_fit ** (2 * n_fit))
        #     * gamma(2 * n_fit)
        #     * gammainc(2 * n_fit, b_n_fit * (7.0 * R_e_fit / R_e_fit) ** (1 / n_fit))
        # )

        # Find index of stellar particles within 0.5 R_e_fit
        indices_within_half_Re = np.where(r_geo <= 0.5 * R_e_fit)[0]
        velocities_within_half_Re = (
            particles["Velocities"][indices_within_half_Re]
            - self.galaxy_data[galaxy_id]["SubhaloVel"]
        )
        # Find the velocities along the line-of-sight axis
        if self.los_axis == 0:
            los_velocities = velocities_within_half_Re[:, 0]
        elif self.los_axis == 1:
            los_velocities = velocities_within_half_Re[:, 1]
        else:  # self.los_axis == 2
            los_velocities = velocities_within_half_Re[:, 2]

        # Find the velocity dispersion within 0.5 R_e_fit
        sigma_los = np.std(los_velocities)

        print(
            f"Galaxy {self.galaxy_data[galaxy_id]['SubhaloID']}, Band {band}: Fitted Sersic index n = {n_fit:.2f} with uncertainty {cov[2, 2] ** 0.5 if cov is not None else 'N/A'}, Effective radius R_e = {R_e_fit:.2f} kpc, Surface brightness I_e = {I_e_fit:.2e} erg/s/kpc^2, velocity dispersion σ_los = {sigma_los:.2f} km/s. The ssr for the de Vaucouleurs fit is {ssr_deV:.2e}, for the exponential fit is {ssr_exp:.2e}, and for the Sersic fit is {ssr_sersic:.2e}."
        )

        # plot_surface_brightness_profile(
        #     r_fit[valid_fit],
        #     surface_brightness[valid_fit],
        #     bestfit,
        #     bestfit_deV,
        #     bestfit_exp,
        #     self.galaxy_data[galaxy_id]["SubhaloID"],
        #     band,
        # )
        # raise ValueError("Stopping after first galaxy for testing purposes.")

        self.galaxy_data[galaxy_id]["R_e_fit_" + band] = R_e_fit
        self.galaxy_data[galaxy_id]["I_e_fit_" + band] = I_e_fit
        self.galaxy_data[galaxy_id]["n_fit_" + band] = n_fit
        # self.galaxy_data[galaxy_id]["L_model_" + band] = L_model
        self.galaxy_data[galaxy_id]["R_e_direct_" + band] = R_e
        # self.galaxy_data[galaxy_id]["is_elliptical"] = is_elliptical
        # self.galaxy_data[galaxy_id]["L_cumulative_" + band] = Lcum
        self.galaxy_data[galaxy_id]["MSB_" + band] = surface_brightness_mean
        self.galaxy_data[galaxy_id]["sigma_los_" + band] = sigma_los
        self.galaxy_data[galaxy_id]["fit_cov_matrix_" + band] = cov

        # For diagnostic purposes, also returns the input radius and surface brightness that used to fit the sersic profile
        self.galaxy_data[galaxy_id]["r_fit_profile_" + band] = r_fit[
            valid_fit
        ]  # radii used for fitting
        self.galaxy_data[galaxy_id]["surface_brightness_profile_" + band] = (
            surface_brightness[valid_fit]
        )  # surface brightness data used for fitting

        pass_cutout_selection, message = self.cutout_selection(
            data_subhalo=self.galaxy_data[galaxy_id],
            stage=2,
            band=band,
            ssr_deV=ssr_deV,
            ssr_exp=ssr_exp,
            sersic_unc=cov[2, 2] ** 0.5 if cov is not None else np.inf,
        )

        if pass_cutout_selection is False:
            print(
                f"Skipping galaxy index {galaxy_id} due to cutout selection criteria: {message}"
            )
            return galaxy_id
        else:
            return None

    # Calculate the luminosity and surface brightness profile for all galaxies in bands
    def calculate_luminosities_and_surface_brightness_profiles(
        self, keyword="analysis"
    ):
        """Calculate luminosities and surface brightness profiles for all galaxies in specified bands."""
        # import time

        del_index = []
        for i in range(len(self.galaxy_data)):
            print(
                f"Calculating luminosities and surface brightness profiles for galaxy {i}..."
            )
            # start = time.time()
            self._estimate_magnitude_luminosity(i)
            for band in self.pardict["bands"]:
                index = self.direct_effective_radius_surface_brightness_sersic_index(
                    i, band, method="stars"
                )
                if index is not None:
                    del_index.append(index)

            # end = time.time()
            # print(f"Time taken for galaxy {i}: {end - start:.2f} seconds")
            # raise ValueError("Stopping after first galaxy for testing purposes.")

        # Remove galaxies that did not pass the selection

        del_index_unique = np.unique(np.array(del_index))

        if len(del_index_unique) > 0:
            print(
                f"Removing {len(del_index_unique)} galaxies that did not pass the selection criteria."
            )
            self.galaxy_data = [
                g for j, g in enumerate(self.galaxy_data) if j not in del_index_unique
            ]
            self.stellar_data = [
                s for j, s in enumerate(self.stellar_data) if j not in del_index_unique
            ]
            self.gas_data = [
                g for j, g in enumerate(self.gas_data) if j not in del_index_unique
            ]

        np.savez(
            f"{self.out_dir}/stellar_data_" + keyword + "_" + str(self.job_id) + ".npz",
            np.array(self.stellar_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{self.out_dir}/gas_data_" + keyword + "_" + str(self.job_id) + ".npz",
            np.array(self.gas_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{self.out_dir}/subhalo_data_" + keyword + "_" + str(self.job_id) + ".npz",
            np.array(self.galaxy_data, dtype=object),
            allow_pickle=True,
        )

    def combined_sfh_output_files(
        self,
        keyword="kinematics_only",
        job_id=None,
        index_to_hdf5_num_table=None,
        out_dir="./",
    ):
        """Combine individual SFH output files into a single file."""

        self.galaxy_data = []
        self.stellar_data = []
        self.gas_data = []

        for i in range(self.start_index, self.end_index):
            hdf5_num = np.where(
                (index_to_hdf5_num_table[:, 2] == i)
                & (index_to_hdf5_num_table[:, 1] > 0)
            )[0][0]
            keyword_index = keyword + "_" + str(hdf5_num)

            try:
                stellar_data_single = np.load(
                    f"{out_dir}/stellar_data_" + keyword_index + ".npz",
                    allow_pickle=True,
                )["arr_0"]
                gas_data_single = np.load(
                    f"{out_dir}/gas_data_" + keyword_index + ".npz", allow_pickle=True
                )["arr_0"]
                galaxy_data_single = np.load(
                    f"{out_dir}/subhalo_data_" + keyword_index + ".npz",
                    allow_pickle=True,
                )["arr_0"]

                for j in range(len(stellar_data_single)):
                    self.stellar_data.append(stellar_data_single[j])
                    self.gas_data.append(gas_data_single[j])
                    self.galaxy_data.append(galaxy_data_single[j])

            except FileNotFoundError:
                print(f"No galaxy pass the selection criteria in hdf5 file: {hdf5_num}")

            # os.remove(f"{out_dir}/stellar_data_" + keyword_index + ".npz")
            # os.remove(f"{out_dir}/gas_data_" + keyword_index + ".npz")
            # os.remove(f"{out_dir}/subhalo_data_" + keyword_index + ".npz")
        if len(self.stellar_data) == 0:
            print("No galaxies satisfy the selection criteria. Exiting.")
            sys.exit(0)

        np.savez(
            f"{out_dir}/stellar_data_" + keyword + ".npz",
            np.array(self.stellar_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{out_dir}/gas_data_" + keyword + ".npz",
            np.array(self.gas_data, dtype=object),
            allow_pickle=True,
        )
        np.savez(
            f"{out_dir}/subhalo_data_" + keyword + ".npz",
            np.array(self.galaxy_data, dtype=object),
            allow_pickle=True,
        )
        print(f"Combined SFH data saved to {out_dir}/combined_sfh_data.npz")


def main():
    """Main function for TNG Tully-Fisher analysis"""
    import sys

    print("TNG Tully-Fisher Generator")
    print("=" * 40)

    # Get number of galaxies from command line argument or use default

    # n_galaxies = int(sys.argv[1])  # The number of subhalos to process
    # snapshot = int(sys.argv[2])  # Snapshot number (z=0)
    # out_dir = sys.argv[3]  # The output directory
    # supplementary_dir = sys.argv[4]  # The supplementary data directory

    config = sys.argv[1]  # Reading in the path to the config file.
    pardict = ConfigObj(config)
    print(pardict["mode_load"])

    if bool(int(pardict["Combined_hdf5"])):
        job_num = 0
        while True:
            # Check if all the folders are present
            if not os.path.exists(
                pardict["out_dir"] + f"/job_id_{job_num}_{pardict['galaxy_type']}/"
            ):
                break
            job_num += 1
        print(f"Combining data from {job_num} jobs...")

        galaxy_data_all = []
        for i in range(job_num):
            out_dir = pardict["out_dir"] + f"/job_id_{i}_{pardict['galaxy_type']}/"

            try:
                galaxy_data = np.load(
                    out_dir + "/subhalo_data_analysis" + "_" + str(i) + ".npz",
                    allow_pickle=True,
                )["arr_0"]
                for j in range(len(galaxy_data)):
                    galaxy_data_all.append(galaxy_data[j])

                if i == 0:
                    # Download the first 5 galaxies as test set
                    np.savez(
                        pardict["out_dir"] + "/subhalo_data_analysis_test_set.npz",
                        np.array(galaxy_data[:5], dtype=object),
                        allow_pickle=True,
                    )
            except FileNotFoundError:
                print(
                    f"No data found in {out_dir}, no galaxy satisfy the selection criteria."
                )
        np.savez(
            pardict["out_dir"] + "/subhalo_data_analysis.npz",
            np.array(galaxy_data_all, dtype=object),
            allow_pickle=True,
        )

        if bool(int(pardict["save_particles"])):
            stellar_data_all = []
            gas_data_all = []
            for i in range(job_num):
                out_dir = pardict["out_dir"] + f"/job_id_{i}_{pardict['galaxy_type']}/"

                try:
                    stellar_data = np.load(
                        out_dir + "/stellar_data_analysis" + "_" + str(i) + ".npz",
                        allow_pickle=True,
                    )["arr_0"]
                    gas_data = np.load(
                        out_dir + "/gas_data_analysis" + "_" + str(i) + ".npz",
                        allow_pickle=True,
                    )["arr_0"]
                    for j in range(len(stellar_data)):
                        stellar_data_all.append(stellar_data[j])
                        gas_data_all.append(gas_data[j])

                    if i == 0:
                        # Download the first 5 galaxies as test set
                        np.savez(
                            pardict["out_dir"] + "/stellar_data_analysis_test_set.npz",
                            np.array(stellar_data[:5], dtype=object),
                            allow_pickle=True,
                        )
                        np.savez(
                            pardict["out_dir"] + "/gas_data_analysis_test_set.npz",
                            np.array(gas_data[:5], dtype=object),
                            allow_pickle=True,
                        )
                except FileNotFoundError:
                    print(
                        f"No data found in {out_dir}, no galaxy satisfy the selection criteria."
                    )

            np.savez(
                pardict["out_dir"] + "/stellar_data_analysis.npz",
                np.array(stellar_data_all, dtype=object),
                allow_pickle=True,
            )
            np.savez(
                pardict["out_dir"] + "/gas_data_analysis.npz",
                np.array(gas_data_all, dtype=object),
                allow_pickle=True,
            )

        print("✅ Combined data saved successfully!")

    else:
        if pardict["mode_load"] == "local":
            job_id = int(sys.argv[2])  # Index file for local data loading
            pardict["out_dir"] = (
                pardict["out_dir"] + f"/job_id_{job_id}_{pardict['galaxy_type']}/"
            )
            os.makedirs(pardict["out_dir"], exist_ok=True)
        else:
            job_id = None

        n_galaxies = int(pardict["n_galaxies"])  # The number of subhalos to process
        snapshot = int(pardict["snapshot"])  # Snapshot number
        # out_dir = pardict["out_dir"]  # The output directory
        # supplementary_dir = pardict["supplementary_dir"]  # The supplementary data directory
        # sup_files = pardict["sup_files"]  # The supplementary data files
        mass_range = (
            float(pardict["mass_low"]),
            float(pardict["mass_high"]),
        )
        minimum_particles = int(pardict["min_particles"])

        print("Running analysis with real " + pardict["simulation"] + " data...")
        try:
            if pardict["mode_load"] == "API":
                api_key = os.environ.get("TNG_API_KEY")
                if not api_key:
                    print("TNG API key not found in environment variable TNG_API_KEY")
                    print(
                        "Please get your API key from: https://www.tng-project.org/users/profile/"
                    )
                    api_key = pardict["TNG_API_KEY"]  # Or read from config file
            elif pardict["mode_load"] == "local":
                api_key = None  # No API key needed for local loading
                print("Loading data from local files...")
            else:
                raise ValueError(
                    "❌ Invalid mode_load specified in config file. Use 'API' or 'local'."
                )

            generator = TNG50TullyFisherGenerator(
                api_key=api_key,
                snapshot=snapshot,
                pardict=pardict,
                job_id=job_id,
                simulation=pardict["simulation"],
            )

            if pardict["mode"] == "sfh":
                if pardict["mode_load"] == "API":
                    generator.load_tng50_data(
                        limit=n_galaxies,
                        stellar_mass_range=mass_range,
                        minimum_particles=minimum_particles,
                    )
                else:  # Local loading
                    try:
                        index_to_hdf5_num_table = np.loadtxt(
                            pardict["out_dir"] + "/subhalo_num_per_file.txt"
                        )
                    except Exception as e:
                        print(
                            "Cannot find subhalo_num_per_file.txt. Generating it now..."
                        )
                        index_to_hdf5_num_table, pardict["sSFR_ridge"] = (
                            generator.find_subhalo_ID()
                        )
                        print("Generated subhalo_num_per_file.txt.")

                    num_hdf5_subhalo = int(index_to_hdf5_num_table[-1, -1] + 1)

                    tot_job, reminder = divmod(
                        num_hdf5_subhalo, int(pardict["index_per_job"])
                    )
                    if reminder > 0 and job_id == tot_job:
                        start_index = job_id * int(pardict["index_per_job"])
                        end_index = num_hdf5_subhalo
                    else:
                        start_index = job_id * int(pardict["index_per_job"])
                        end_index = (job_id + 1) * int(pardict["index_per_job"])

                    print(start_index, end_index, num_hdf5_subhalo)

                    for index in range(start_index, end_index):
                        print(f"\nLoading data for HDF5 subhalo file index {index}...")

                        generator.load_tng50_data(
                            limit=n_galaxies,
                            stellar_mass_range=mass_range,
                            minimum_particles=minimum_particles,
                            index_local=index,
                        )
                print("\n✅ Data successfully downloaded!")
            elif pardict["mode"] == "analysis":
                generator.calculate_luminosities_and_surface_brightness_profiles()
                print("\n✅ Analysis completed successfully!")
            else:
                raise ValueError(
                    "❌ Invalid mode specified in config file. Use 'sfh' or 'analysis'."
                )

        except Exception as e:
            print(f"❌ Failed to download data: {e}")


if __name__ == "__main__":
    main()
