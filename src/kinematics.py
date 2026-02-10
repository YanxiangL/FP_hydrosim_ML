import numpy as np
import collections.abc as abc
from scipy.spatial.transform import Rotation
from scipy.stats import binned_statistic
import warnings

warnings.filterwarnings("error")


def std(x, xmean, weights):
    num = x.size
    return np.sqrt(
        np.sum((x - xmean) ** 2 * weights) / np.sum(weights) * num / (num - 1)
    )


def AngularMomentum(mass, pos, vel, return_ji=True, range=None):
    """
    ji: specific angular momentum
    ez: direcction along the total angular momentum

    return_ji: whether or nor return the ji
    range: the sphere where we should include in the calculation
    """
    if isinstance(range, float):
        if range > 0:
            idx = np.linalg.norm(pos, axis=1) < range
            if np.sum(idx) != 0:
                ji = np.cross(pos[idx], vel[idx])
                Jtot = np.sum(ji * mass[idx, np.newaxis], axis=0)
            else:
                raise ZeroDivisionError(f"zero denominator")
        else:
            raise ValueError(f"range should be positive")
    else:
        ji = np.cross(pos, vel)
        Jtot = np.sum(ji * mass[:, np.newaxis], axis=0)

    ez = Jtot / np.sqrt(np.sum(Jtot**2))

    if return_ji and range is None:
        return ji, ez
    else:
        return ez


def fraction_in_disk(mass, pos, vel, ez, range=None):
    if isinstance(range, float):
        if range > 0:
            idx = np.linalg.norm(pos, axis=1) < range
            if np.sum(idx) != 0:
                j = np.cross(pos[idx], vel[idx])
                jz = np.dot(ez[np.newaxis, :], j.T).flatten()
                mass_weight_j = np.sum(
                    jz / np.linalg.norm(j, axis=1) * mass[idx], axis=0
                )
                mass_sum = np.sum(mass[idx])
                frac = mass_weight_j / mass_sum
                return frac
            else:
                raise ZeroDivisionError(f"zero denominator")
        else:
            raise ValueError(f"range should be positive")
    else:
        j = np.cross(pos, vel)
        jz = np.dot(ez[np.newaxis, :], j.T).flatten()
        mass_weight_j = np.sum(jz / np.linalg.norm(j, axis=1) * mass, axis=0)
        mass_sum = np.sum(mass)
        frac = mass_weight_j / mass_sum
        return frac


def RotationMatrix(z, zprime):
    """
    z: original unit vector
    zprime: new direction in unit vector

    return scipy Rotation class.
    Use RotationMatrix.apply(v) for transforming v to the new coordinate
    """
    theta = np.arccos(np.dot(zprime, z))
    u = np.cross(zprime, z)
    u /= np.linalg.norm(u)
    return Rotation.from_rotvec(u * theta)


def RotationVelocity(vel, pos, ez, return_etheta=True, eps=1e-8):
    """
    return
        vrot: rotational velocity component
        etheta: direction of the rotation
    """
    pos_perp = pos - np.dot(pos, ez)[:, np.newaxis] * ez[np.newaxis, :]
    etheta = np.cross(ez, pos_perp)

    # etheta = np.cross(ez, pos)
    norm = np.linalg.norm(etheta, axis=1)
    mask = norm > eps
    if np.sum(mask) != len(norm):
        print("Some particles lie very close to the rotation axis or at the center")

    etheta_new = np.zeros(etheta.shape)
    etheta_new[mask] = etheta[mask] / norm[mask, np.newaxis]
    vrot = np.zeros(len(vel))
    vrot[mask] = np.einsum("ij,ij->i", vel[mask], etheta_new[mask])
    if return_etheta:
        return vrot, etheta_new
    else:
        return vrot


def find_v_rot_v_sigma(mass, vel, pos, range=None):
    ez = AngularMomentum(mass, pos, vel, range=range, return_ji=False)

    if range is not None and range > 0:
        idx = np.linalg.norm(pos, axis=1) < range
    else:
        idx = np.arange(pos.shape[0])

    pos = pos[idx]
    vel = vel[idx]
    mass = mass[idx]

    vrot, etheta = RotationVelocity(vel, pos, ez)
    vrot_mass = np.sum(mass * vrot) / np.sum(mass)

    # Velocity dispersion (3D, relative to mean rotation)
    vel_mean = vrot[:, None] * etheta
    vel_residual = vel - vel_mean

    # Mass weighted velocity dispersion
    sigma2 = np.sum(mass * np.sum(vel_residual**2, axis=1)) / np.sum(mass)
    sig_tot = np.sqrt(sigma2)

    vrot_v_sigma = vrot_mass / sig_tot

    return vrot_v_sigma, vrot, sig_tot


def RadiusDecomp(pos, ez):
    """
    return
        ri: cylindrical r
        zi: cylindrical z
        er: cylindrical r direction
    """
    zi = np.dot(pos, ez)
    er = pos - zi[:, np.newaxis] * ez[np.newaxis, :]
    ri = np.linalg.norm(er, axis=1)
    er = er / ri[:, np.newaxis]
    return ri, zi, er


def ValueAtRadius(
    pos, value, radius, rwidth, rlim, ez, zi_weight=True, line_weight=None
):
    ri, zi, _ = RadiusDecomp(pos, ez)
    idx = (np.abs(ri - radius) < rwidth) & (np.linalg.norm(pos, axis=1) < rlim)
    n = np.sum(idx)
    if n > 1:
        # use inverse of distance to the disk as weighting
        if zi_weight:
            weights = 1.0 / (np.abs(zi[idx]) + 1e-9)
        else:
            weights = np.ones_like(zi[idx])

        # consider line strength
        if isinstance(line_weight, (abc.Sequence, np.ndarray)):
            weights *= line_weight[idx]

        # mean and standard deviation
        if np.sum(weights) != 0:
            mean_vrot = np.average(value[idx], weights=weights)
            std_vrot = np.sqrt(
                n
                / (n - 1)
                * np.sum(weights * (value[idx] - mean_vrot) ** 2)
                / np.sum(weights)
            )
        else:
            mean_vrot = 0.0
            std_vrot = 0.0
    elif n == 1:
        mean_vrot = value[idx][0]
        std_vrot = 0.0
    else:
        mean_vrot = 0.0
        std_vrot = 0.0
    return mean_vrot, std_vrot


def RotationCurve(r, vrot, bins, rhalf, weights=None):
    rbins = 0.5 * (bins[:-1] + bins[1:])
    if isinstance(weights, (abc.Sequence, np.ndarray)):
        vrot_mean = np.zeros(rbins.shape)
        vrot_std = np.zeros(rbins.shape)
        binnumber = np.digitize(r, rhalf * bins)
        for i in range(rbins.size):
            idx = binnumber == i + 1
            if np.sum(idx) > 1:
                vrot_mean[i] = np.average(vrot[idx], weights=weights[idx])
                vrot_std[i] = std(vrot[idx], vrot_mean[i], weights[idx])
            elif np.sum(idx) == 1:
                vrot_mean[i] = vrot[idx]
                vrot_std[i] = np.nan
            else:
                vrot_mean[i] = np.nan
                vrot_std[i] = np.nan
    else:
        ret = binned_statistic(r, vrot, statistic="mean", bins=bins * rhalf)
        vrot_mean = ret.statistic
        ret = binned_statistic(r, vrot, statistic="std", bins=bins * rhalf)
        vrot_std = ret.statistic
    return rbins, vrot_mean, vrot_std


def Vmax(vrot_mean, vrot_std=None, rbins=None):
    if isinstance(rbins, (abc.Sequence, np.ndarray)) and isinstance(
        vrot_std, (abc.Sequence, np.ndarray)
    ):
        # only return the value has representive value
        idx = ~np.isnan(vrot_std)
        valid_mean = vrot_mean[idx]
        valid_std = vrot_std[idx]
        valid_rbins = rbins[idx]

        idx = np.nanargmax(valid_mean)
        return valid_mean[idx], valid_std[idx], valid_rbins[idx]
    else:
        idx = np.nanargmax(vrot_mean)
        return vrot_mean[idx]


def RotationEnergy(mass, pos, ji, ez, line_weight=None):
    """
    return rotational energy
    """
    ri, _, _ = RadiusDecomp(pos, ez)
    jzi = np.dot(ji, ez)

    if isinstance(line_weight, (abc.Sequence, np.ndarray)):
        return 0.5 * np.sum(mass * (jzi / ri) ** 2 * line_weight) / np.sum(line_weight)
    else:
        return 0.5 * np.sum(mass * (jzi / ri) ** 2)


def KineticEnergy(mass, vel, line_weight=None):
    """
    return kinematic energy
    """
    vtot = np.linalg.norm(vel, axis=1)
    if isinstance(line_weight, (abc.Sequence, np.ndarray)):
        return 0.5 * np.sum(mass * vtot**2 * line_weight) / np.sum(line_weight)
    else:
        return 0.5 * np.sum(mass * vtot**2)


def KappaRotation(mass, pos, vel, line_weight=None, ez_range=None):
    if ez_range is None:
        ji, ez = AngularMomentum(mass, pos, vel)
    else:
        ez = AngularMomentum(mass, pos, vel, range=ez_range)
        ji = np.cross(pos, vel)
    Erot = RotationEnergy(mass, pos, ji, ez, line_weight=line_weight)
    Ekin = KineticEnergy(mass, vel, line_weight=line_weight)
    return Erot / Ekin
