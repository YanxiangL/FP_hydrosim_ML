import numpy as np
import matplotlib.pyplot as plt
import kinematics
from scipy.spatial.transform import Rotation
from configobj import ConfigObj
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


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
    # return I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1)) / y_scale
    return np.log10(I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1)) / y_scale)


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
    # return I_e * np.exp(-7.669 * ((r / R_e) ** (1 / 4) - 1)) / y_scale
    return np.log10(I_e * np.exp(-7.669 * ((r / R_e) ** (1 / 4) - 1)) / y_scale)


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
    # return I_e * np.exp(-1.678 * ((r / R_e) - 1)) / y_scale
    return np.log10(I_e * np.exp(-1.678 * ((r / R_e) - 1)) / y_scale)


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


def get_params(catalogue, band):
    """
    Extract Sersic profile parameters from a catalogue for a given band.

    catalogue : structured array or pandas DataFrame containing galaxy parameters
    band : string, the photometric band to extract parameters for
    Returns: tuple of arrays (I_e_log, R_e, n)
    """
    SB_profile = catalogue[f"surface_brightness_profile_{band}"]
    r_profile = catalogue["r_fit_profile_" + band]
    uncertainty = catalogue["surface_brightness_uncertainty_" + band]
    chi2_red = catalogue["reduced_chi_squared_" + band]

    R_e_fit = catalogue["R_e_fit_" + band]
    I_e_fit = catalogue["I_e_fit_" + band]
    n_fit = catalogue["n_fit_" + band]

    return SB_profile, r_profile, uncertainty, chi2_red, R_e_fit, I_e_fit, n_fit


def plot_vmap(subhalo_id, stat, xedges, yedges, method="star"):
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
    plt.savefig(f"../../plots/galaxy_{subhalo_id}_vmap_edgeon_{method}.png", dpi=300)
    plt.show()


def plot_edgeon_faceon_view(pos_norm, pos_lim, subhalo_id, method="star"):
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

    plt.savefig(f"../../plots/galaxy_{subhalo_id}_edgeon_faceon_{method}.png", dpi=300)
    plt.show()


def make_vmap_hist(pos_obs, v_los, x_range, y_range, nx=200, ny=200):
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


# Find the projected coordinates given the line-of-sight axis
def project_coordinates(los_axis=2, particles=None, pos=None):
    """
    Project particle coordinates based on the line-of-sight axis
    Parameters:
    -----------
    los_axis : int
        Line-of-sight axis (0, 1, or 2)
    particles : dict
        Particle data containing 'Coordinates' key
    Returns:
    --------
    numpy.ndarray
        Projected particle coordinates
    """
    # --- project coordinates based on line-of-sight axis ---
    if particles is not None:
        if los_axis == 0:
            particle_coords = particles["Coordinates"][:, [1, 2]]
        elif los_axis == 1:
            particle_coords = particles["Coordinates"][:, [0, 2]]
        else:  # los_axis == 2
            particle_coords = particles["Coordinates"][:, [0, 1]]
    elif pos is not None:
        if los_axis == 0:
            particle_coords = pos[:, [1, 2]]
        elif los_axis == 1:
            particle_coords = pos[:, [0, 2]]
        else:  # los_axis == 2
            particle_coords = pos[:, [0, 1]]

    else:
        raise ValueError(
            "Either provide a dictionary of particles or a position array."
        )

    return particle_coords


def find_inclination_ellipticity(
    pardict, data_subhalo, data_stellar=None, data_gas=None, method="star", los_axis=2
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
        pos_lim = float(pardict["pos_lim_star"])

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
        pos_lim = float(pardict["pos_lim_gas"])

    # Using the user defined line-of-sight (LOS) direction
    if los_axis == 2:
        los_vector = np.array([0, 0, 1])
        vector_y = np.array([0, 1, 0])
        vector_x = np.array([1, 0, 0])
        los_direction = "z"
        inclination_direction = "x"
    elif los_axis == 1:
        los_vector = np.array([0, 1, 0])
        vector_y = np.array([0, 0, 1])
        vector_x = np.array([1, 0, 0])
        los_direction = "y"
        inclination_direction = "x"
    elif los_axis == 0:
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

    print("Inclination angle between angular momentum axis and z-axis:", inclination)
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

    pos_norm_los = project_coordinates(pos=pos_norm)

    stat, _, xedges, yedges = make_vmap_hist(
        pos_norm_los,
        v_los,
        x_range=[-pos_lim, pos_lim],
        y_range=[-pos_lim, pos_lim],
        nx=200,
        ny=200,
    )

    plot_edgeon_faceon_view(pos_norm, pos_lim, data_subhalo["SubhaloID"], method=method)

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

    pos_rot_los = project_coordinates(pos=pos_rot)

    stat, _, xedges, yedges = make_vmap_hist(
        pos_rot_los,
        v_los,
        x_range=[-pos_lim, pos_lim],
        y_range=[-pos_lim, pos_lim],
        nx=200,
        ny=200,
    )

    plot_vmap(data_subhalo["SubhaloID"], stat, xedges, yedges, method=method)


def rotation_curve_fit(rr, vv, vv_err, p0=[200.0, 5.0, 1.0]):
    """
    Fit the rotation curve using an arctangent function.

    rr : radius array
    vv : velocity array
    vv_err : velocity error array
    p0 : initial guess for the parameters [V_max, R_turn, offset]
    Returns: fitted parameters and covariance matrix
    """

    def arctan_func(r, V_max, R_turn, offset):
        return (2 / np.pi) * V_max * np.arctan(r / R_turn) + offset

    popt, pcov = curve_fit(
        arctan_func,
        rr,
        vv,
        sigma=vv_err,
        absolute_sigma=True,
        p0=p0,
        maxfev=10000,
        bounds=([0.0, 0.01, -1000.0], [1000, 10.0, 1000.0]),
    )

    model = arctan_func(rr, *popt)
    return popt, pcov, model


def make_synthetic_image(xyz, weight, box_size=50.0, npix=512, sigma=1.0):
    """
    Generate a 2D synthetic image from particle positions and luminosities.

    Parameters
    ----------
    xyz : ndarray
        Nx3 array of particle positions (e.g., in kpc)
    weight : ndarray
        Nx1 array of weights (e.g., dust-attenuated luminosities)
    box_size : float
        Size of the 2D projected box (kpc)
    npix : int
        Number of pixels along each axis
    sigma : float
        Gaussian smoothing (in pixels)

    Returns
    -------
    img : 2D ndarray
        Smoothed luminosity map
    """
    # project onto xy-plane
    x, y = xyz[:, 0], xyz[:, 1]

    # make 2D histogram weighted by luminosity
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=npix,
        range=[[-box_size / 2, box_size / 2], [-box_size / 2, box_size / 2]],
        weights=weight,
    )

    # # optional Gaussian smoothing
    # img = gaussian_filter(H, sigma=sigma)

    img = H

    return img


# data = np.load(
#     "/home/ylai1998/KL_pipeline/kl_roman_pipe/data/tng50/subhalo_data_analysis.npz",
#     allow_pickle=True,
# )["arr_0"]
# data_gas = np.load(
#     "/home/ylai1998/KL_pipeline/kl_roman_pipe/data/tng50/gas_data_analysis.npz",
#     allow_pickle=True,
# )["arr_0"]
# data_stellar = np.load(
#     "/home/ylai1998/KL_pipeline/kl_roman_pipe/data/tng50/stellar_data_analysis.npz",
#     allow_pickle=True,
# )["arr_0"]

# galaxy_0 = data[0]
# gas_0 = data_gas[0]
# stellar_0 = data_stellar[0]

# pos0 = galaxy_0["SubhaloPos"]
# vel0 = galaxy_0["SubhaloVel"]
# radius = galaxy_0["SubhaloHalfmassRadStars"]

# pos = gas_0["Coordinates"] - pos0
# vel = gas_0["Velocities"] - vel0
# mass = gas_0["Masses"]
# sfr = gas_0["StarFormationRate"]

# ez = kinematics.AngularMomentum(
#     mass,
#     pos,
#     vel,
#     return_ji=False,
#     # Restrict to twice the stellar half-mass radius
#     range=2.0 * galaxy_0["SubhaloHalfmassRadStars"],
# )

# pos_star = stellar_0["Coordinates"] - pos0
# vel_star = stellar_0["Velocities"] - vel0
# mass_star = stellar_0["Masses"]

# axis = kinematics.AngularMomentum(mass_star, pos_star, vel_star, return_ji=False)

# # vrot, etheta = kinematics.RotationVelocity(vel, pos, ez)
# vrot, etheta = kinematics.RotationVelocity(vel, pos, axis)

# rr_sample = np.linspace(
#     0.0, 2.5 * galaxy_0["SubhaloHalfmassRadStars"], 100, endpoint=False
# )

# mean_vrot = []
# std_vrot = []
# for r in rr_sample:
#     mean, std = kinematics.ValueAtRadius(
#         pos,
#         vrot,
#         r + 0.05,
#         rwidth=0.05,
#         rlim=2.5 * galaxy_0["SubhaloHalfmassRadStars"],
#         ez=axis,
#         zi_weight=False,
#     )
#     mean_vrot.append(mean)
#     std_vrot.append(std)

# mean_vrot = np.array(mean_vrot)
# std_vrot = np.array(std_vrot)
# std_vrot[0] = std_vrot[1]  # avoid zero error at r=0

# plt.errorbar(rr_sample + 0.05, mean_vrot, yerr=std_vrot, fmt="o", ls="none", capsize=3)
# plt.xlabel("Radius [kpc]")
# plt.ylabel("Mean Rotation Velocity [km/s]")

# index_keep = np.where(rr_sample + 0.05 > 0.3)[0]


# popt, pcov, model = rotation_curve_fit(
#     rr_sample[index_keep] + 0.05, mean_vrot[index_keep], std_vrot[index_keep]
# )

# plt.plot(rr_sample[index_keep] + 0.05, model, "-", label="Fitted Rotation Curve")
# plt.legend()
# plt.show()
# print("Fitted rotation curve parameters (V_max, R_turn, offset):", popt)
# print(np.sqrt(np.diag(pcov)))
# print(galaxy_0["SubhaloVmax"], galaxy_0["SubhaloVmaxRad"])
# raise ValueError("Debugging stop")

galaxy_index = 0

data = np.load("../../TNG50/subhalo_data_analysis_gold.npz", allow_pickle=True)["arr_0"]
data_gas = np.load("../../TNG50/gas_data_analysis_gold.npz", allow_pickle=True)["arr_0"]
data_stellar = np.load("../../TNG50/stellar_data_analysis_gold.npz", allow_pickle=True)[
    "arr_0"
]

print("Number of galaxies in the sample:", len(data))

# angles_all = []
# for i in range(len(data)):
#     sfr_gas = data_gas[i]["StarFormationRate"]
#     pos_rel_gas = data_gas[i]["Coordinates"] - data[i]["SubhaloPos"]
#     mass_gas = data_gas[i]["Masses"]
#     index = np.where(
#         (pos_rel_gas[:, 0] ** 2 + pos_rel_gas[:, 1] ** 2 + pos_rel_gas[:, 2] ** 2)
#         ** 0.5
#         < 2.0 * data[i]["SubhaloHalfmassRadStars"]
#     )[0]
#     # Find gas within 2 times the stellar half-mass radius
#     sfr_gas_within = sfr_gas[index]
#     mass_gas_within = mass_gas[index]
#     # Find gas particles within the radius with non-zero star formation rate
#     sfr_gas_sf = np.where(sfr_gas_within > 0)[0]
#     mass_gas_sf = mass_gas_within[sfr_gas_sf]
#     f_sf_gas = (
#         np.sum(mass_gas_sf) / np.sum(mass_gas_within)
#         if np.sum(mass_gas_within) > 0
#         else 0.0
#     )

#     mass_star = data_stellar[i]["Masses"]
#     pos_rel_star = data_stellar[i]["Coordinates"] - data[i]["SubhaloPos"]
#     vel_rel_star = data_stellar[i]["Velocities"] - data[i]["SubhaloVel"]
#     vel_rel_gas = data_gas[i]["Velocities"] - data[i]["SubhaloVel"]

#     vrot_vsigma_star = kinematics.find_v_rot_v_sigma(
#         mass_star,
#         vel_rel_star,
#         pos_rel_star,
#         range=2.0 * data[i]["SubhaloHalfmassRadStars"],
#     )[0]
#     vrot_vsigma_gas = kinematics.find_v_rot_v_sigma(
#         mass_gas,
#         vel_rel_gas,
#         pos_rel_gas,
#         range=2.0 * data[i]["SubhaloHalfmassRadStars"],
#     )[0]

#     angles_all.append(
#         [
#             data[i]["SubhaloID"],
#             data[i]["Inclination_star"],
#             data[i]["Inclination_gas"],
#             data[i]["Position_Angle_star"],
#             data[i]["Position_Angle_gas"],
#             data[i]["ThinDisc"][0],
#             f_sf_gas,
#             vrot_vsigma_star,
#             vrot_vsigma_gas,
#             data[i]["chi2_red_sersic"],
#         ],
#     )

# angles_all = np.array(angles_all)
# np.savetxt("../../plots/angles_all_0.txt", angles_all, fmt="%3.3f")

# raise ValueError("Debugging stop")

configfile = "../../pygalaxev/config.ini"
pardict = ConfigObj(configfile)
galaxy_0 = data[galaxy_index]
gas_0 = data_gas[galaxy_index]
stellar_0 = data_stellar[galaxy_index]

(
    galaxy_0_r_SB_profile,
    galaxy_0_r_profile,
    galaxy_0_r_uncertainty,
    galaxy_0_r_chi2_red,
    galaxy_0_r_R_e_fit,
    galaxy_0_r_I_e_fit,
    galaxy_0_r_n_fit,
) = get_params(galaxy_0, "r")
(
    galaxy_0_g_SB_profile,
    galaxy_0_g_profile,
    galaxy_0_g_uncertainty,
    galaxy_0_g_chi2_red,
    galaxy_0_g_R_e_fit,
    galaxy_0_g_I_e_fit,
    galaxy_0_g_n_fit,
) = get_params(galaxy_0, "g")
(
    galaxy_0_i_SB_profile,
    galaxy_0_i_profile,
    galaxy_0_i_uncertainty,
    galaxy_0_i_chi2_red,
    galaxy_0_i_R_e_fit,
    galaxy_0_i_I_e_fit,
    galaxy_0_i_n_fit,
) = get_params(galaxy_0, "i")

sersic_fit_r = sersic_profile(
    galaxy_0_r_profile,
    np.log10(galaxy_0_r_I_e_fit),
    galaxy_0_r_R_e_fit,
    galaxy_0_r_n_fit,
    y_scale=1.0,
)
sersic_fit_g = sersic_profile(
    galaxy_0_g_profile,
    np.log10(galaxy_0_g_I_e_fit),
    galaxy_0_g_R_e_fit,
    galaxy_0_g_n_fit,
    y_scale=1.0,
)
sersic_fit_i = sersic_profile(
    galaxy_0_i_profile,
    np.log10(galaxy_0_i_I_e_fit),
    galaxy_0_i_R_e_fit,
    galaxy_0_i_n_fit,
    y_scale=1.0,
)

print(
    "galaxy_0 SubhaloID:",
    galaxy_0["SubhaloID"],
    " with reduced chi-squared:",
    galaxy_0_r_chi2_red,
    " in r band",
    galaxy_0_g_chi2_red,
    " in g band",
    galaxy_0_i_chi2_red,
    " in i band",
)
print(
    "Inclination (star, gas):",
    galaxy_0["Inclination_star"],
    galaxy_0["Inclination_gas"],
)
print(
    "Position Angle (star, gas):",
    galaxy_0["Position_Angle_star"],
    galaxy_0["Position_Angle_gas"],
)

plt.figure(figsize=(10, 6))
plt.errorbar(
    galaxy_0_r_profile / galaxy_0_r_R_e_fit,
    np.log10(galaxy_0_r_SB_profile),
    yerr=galaxy_0_r_uncertainty,
    fmt=".",
    label="r-band",
    c="r",
)
plt.errorbar(
    galaxy_0_g_profile / galaxy_0_g_R_e_fit,
    np.log10(galaxy_0_g_SB_profile),
    yerr=galaxy_0_g_uncertainty,
    fmt=".",
    label="g-band",
    c="g",
)
plt.errorbar(
    galaxy_0_i_profile / galaxy_0_i_R_e_fit,
    np.log10(galaxy_0_i_SB_profile),
    yerr=galaxy_0_i_uncertainty,
    fmt=".",
    label="i-band",
    c="b",
)
plt.plot(
    galaxy_0_r_profile / galaxy_0_r_R_e_fit,
    sersic_fit_r,
    "--",
    label=f"r-band Sersic fit, n = {round(galaxy_0_r_n_fit, 3)}",
    c="r",
)
plt.plot(
    galaxy_0_g_profile / galaxy_0_g_R_e_fit,
    sersic_fit_g,
    "--",
    label=f"g-band Sersic fit, n = {round(galaxy_0_g_n_fit, 3)}",
    c="g",
)
plt.plot(
    galaxy_0_i_profile / galaxy_0_i_R_e_fit,
    sersic_fit_i,
    "--",
    label=f"i-band Sersic fit, n = {round(galaxy_0_i_n_fit, 3)}",
    c="b",
)
plt.xlabel("Radius / R_e")
plt.ylabel("log10(Surface Brightness [erg/s/kpc^2])")
plt.legend()
plt.xscale("log")
plt.savefig(f"../../plots/sersic_fit_galaxy_{galaxy_0['SubhaloID']}.png", dpi=300)
plt.show()

find_inclination_ellipticity(pardict, galaxy_0, data_stellar=stellar_0, method="star")
find_inclination_ellipticity(pardict, galaxy_0, data_gas=gas_0, method="gas")
