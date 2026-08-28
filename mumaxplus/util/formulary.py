"""Functions to calculate common micromagnetic quantities."""

import numpy as _np
from .constants import MU0


def magnetostatic_energy_density(msat) -> float:
    r"""Magnetostatic energy density (J/m³).
    
    1/2 μ0 msat²
    
    Parameters
    ----------
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetostatic energy density (J/m³).
    """
    return 0.5 * MU0 * msat**2

def Km(msat) -> float:
    r"""Magnetostatic energy density (J/m³).
    
    1/2 μ0 msat²

    This is a short alias for `magnetostatic_energy_density(msat)`.

    Parameters
    ----------
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetostatic energy density (J/m³).
    """
    return magnetostatic_energy_density(msat)


def exchange_length(aex, msat) -> float:
    r"""Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.
    
    .. math:: l_{ex} = \sqrt{a_\text{ex}/K_\text{m}} = \sqrt{2a_\text{ex}/(\mu_0 m_\text{sat}^2)}
    
    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Exchange length (m).
    """
    return _np.sqrt(aex/Km(msat))

def l_ex(aex, msat) -> float:
    r"""Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.

    .. math:: l_{ex} = \sqrt{a_\text{ex}/K_\text{m}} = \sqrt{2a_\text{ex}/(\mu_0 m_\text{sat}^2)}

    This is a short alias for `exchange_length(aex, msat)`.
    
    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Exchange length (m).
    """
    return exchange_length(aex, msat)


def wall_width(aex, K_eff) -> float:
    r"""Bloch wall width (m). Mind the lack of any prefactor! Different
    definitions exist, for example with a prefactor π.
    
    .. math:: \sqrt{a_\text{ex}/K_{\text{eff}}}

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    K_eff : float
        Effective anisotropy constant (J/m³), for example Ku1.

    Returns
    -------
    float
        Wall width (m).
    """
    return _np.sqrt(aex/K_eff)

def wall_energy(aex, K_eff) -> float:
    r"""Bloch wall energy (J/m²): the energy per unit of Bloch domain wall area.
    Mind the lack of any prefactor, different definitions exist!

    .. math:: \sqrt{a_\text{ex} K_{\text{eff}}}

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    K_eff : float
        Effective anisotropy constant (J/m³), for example Ku1.

    Returns
    -------
    float
        Wall energy (J/m²).
    """
    return _np.sqrt(aex*K_eff)


def helical_length(aex, D) -> float:
    r"""Characteristic length scale of a DMI dominated system, such as with
    skyrmions. Mind the lack of any prefactor! Different definitions exist,
    with prefactors like 1, 2 or 4π.
    
    aex/D

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    D : float
        Strength of DMI (J/m²), for example interfacial DMI strength.

    Returns
    -------
    float
        Helical length (m).
    """
    return aex/D


def magnetic_hardness(K1, msat) -> float:
    r"""Magnetic hardness parameter κ (dimensionless). It should be greater than
    1 for a permanent magnet and much less than 1 for a good temporary magnet.
    
    .. math:: \sqrt{\frac{|K_1|}{\mu_0 m_\text{sat}^2}}
    
    https://doi.org/10.1016/j.scriptamat.2015.09.021

    Parameters
    ----------
    K1 : float
        Anisotropy energy density (J/m³), for example ku1.
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetic hardness parameter (dimensionless).
    """
    return _np.sqrt(abs(K1)/(MU0 * msat**2))


def bulk_modulus(C11, C44):
    """For isotropic materials, the bulk modulus is given by
    K = C11 - 4/3 C44 = C12 + 2/3 C44.
    https://en.wikipedia.org/wiki/Hooke%27s_law
    
    Parameters
    ----------
    C11 : float
        Stiffness constant C11 (Pa).
    C44 : float
        Stiffness constant C44 (Pa).

    Returns
    -------
    float
        Bulk modulus (Pa).
    """
    return C11 - 4/3 * C44


def Rayleigh_damping_coefficients(frequency_1, damping_ratio_1, frequency_2, damping_ratio_2):
    """Rayleigh damping mass coefficient α and stiffness coefficient β
    calculated by providing damping ratios ζ₁,₂ at specified frequencies f₁,₂.

    α = 4π f₁ f₂ * (ζ₁f₂ - ζ₂f₁) / (f₂² - f₁²)

    β = 1/π * (ζ₂f₂ - ζ₁f₁) / (f₂² - f₁²)

    with f₁/f₂ <= ζ₂/ζ₁ <= f₂/f₁

    Based on https://www.comsol.com/blogs/how-to-model-different-types-of-damping-in-comsol-multiphysics
    and https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.126.html.
    
    Parameters
    ----------
    frequency_1 : float
        Low frequency f₁ (Hz) at which damping_ratio_1 is expected.
        This should be smaller than frequency_2.
    damping_ratio_1 : float
        Positive damping ratio ζ₁ (dimensionless) expected at frequency_1.
    frequency_2 : float
        High frequency f₂ (Hz) at which damping_ratio_2 is expected.
        This should be larger than frequency_1.
    damping_ratio_2 : float
        Positive damping ratio ζ₂ (dimensionless) expected at frequency_2.

    Returns
    -------
    tuple[float] of size 2
        Rayleigh damping mass coefficient α (1/s) and stiffness coefficient β (s).

    See Also
    --------
    Rayleigh_damping_stiffness_coefficient
    """
    assert damping_ratio_1 > 0 and damping_ratio_2 >= 0
    assert frequency_1 >= 0 and frequency_2 >= 0 and frequency_2 > frequency_1
    assert frequency_1 / frequency_2 <= damping_ratio_2 / damping_ratio_1 and \
           damping_ratio_2 / damping_ratio_1 <= frequency_2 / frequency_1

    denom = (frequency_2*frequency_2 - frequency_1*frequency_1)

    mass_coef =  4 * _np.pi * frequency_1 * frequency_2 \
        * (damping_ratio_1 * frequency_2 - damping_ratio_2 * frequency_1) \
        / denom
    stiffness_coef = 1 / _np.pi \
        * (damping_ratio_2 * frequency_2 - damping_ratio_1 * frequency_1) \
        / denom

    return (mass_coef, stiffness_coef)

def Rayleigh_damping_stiffness_coefficient(frequency, damping_ratio):
    """Rayleigh damping stiffness coefficient β, assuming a mass coefficient of
    zero, calculated by providing a damping ratio ζ at a specified frequency f.

    β = ζ/(πf)

    This is useful to obtain a damping that increases linearly with frequency,
    by only setting the viscosity tensor, but not the phenomenological elastic
    damping constant.

    Based on https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.126.html.
    
    Parameters
    ----------
    frequency : float
        Frequency f (Hz) at which the damping_ratio is expected.
    damping_ratio : float
        Positive damping ratio ζ (dimensionless) expected at the frequency.

    Returns
    -------
    float
        Rayleigh damping stiffness coefficient β (s).

    See Also
    --------
    Rayleigh_damping_coefficients
    """
    assert damping_ratio >= 0 and frequency > 0

    return damping_ratio / (_np.pi * frequency)
