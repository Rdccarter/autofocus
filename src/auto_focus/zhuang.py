"""Zhuang/Huang astigmatic defocus model.

Implements the standard defocus model from:
  Huang, Wang, Bates & Zhuang, Science 319, 810 (2008)

Each lateral PSF width (sigma_x, sigma_y) follows:
  sigma(z) = sigma0 * sqrt(1 + ((z-c)/d)^2 + A*((z-c)/d)^3 + B*((z-c)/d)^4)

The ellipticity ratio e = sigma_x / sigma_y varies monotonically near focus
and can be inverted to recover z.

Ported and cleaned up from the cylindrical-lens reference (Vliet lab).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special


# ---------------------------------------------------------------------------
# Core Zhuang defocus curve
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZhuangParams:
    """Parameters for one lateral axis of the Zhuang defocus model.

    sigma(z) = sigma0 * sqrt(1 + X^2 + A*X^3 + B*X^4)
    where X = (z - c) / d
    """
    sigma0: float   # minimum PSF width (um or px)
    A: float        # 3rd-order coefficient
    B: float        # 4th-order coefficient
    c: float        # z offset of this axis's focus (um)
    d: float        # focal depth parameter (um)


@dataclass(slots=True)
class ZhuangEllipticityParams:
    """Combined 9-parameter model: e(z) = e0 * sqrt(sigma_x^2(z) / sigma_y^2(z)).

    q = [e0, z0, c, Ax, Bx, dx, Ay, By, dy]
    where:
      e0    = sigma_x0 / sigma_y0 at focus
      z0    = center of focus
      c     = half-separation between x and y foci
      Ax,Bx = 3rd/4th order for x axis
      dx    = focal depth for x axis
      Ay,By,dy = same for y axis
    """
    e0: float
    z0: float
    c: float
    Ax: float
    Bx: float
    dx: float
    Ay: float
    By: float
    dy: float

    def to_array(self) -> np.ndarray:
        return np.array([self.e0, self.z0, self.c,
                         self.Ax, self.Bx, self.dx,
                         self.Ay, self.By, self.dy])

    @classmethod
    def from_array(cls, q: np.ndarray) -> "ZhuangEllipticityParams":
        return cls(e0=q[0], z0=q[1], c=q[2],
                   Ax=q[3], Bx=q[4], dx=q[5],
                   Ay=q[6], By=q[7], dy=q[8])

    @classmethod
    def from_axis_params(cls, px: ZhuangParams, py: ZhuangParams) -> "ZhuangEllipticityParams":
        """Construct from independently fitted x and y axis parameters."""
        e0 = math.sqrt(px.sigma0 / py.sigma0) if py.sigma0 > 0 else 1.0
        z0 = (px.c + py.c) / 2.0
        c = (px.c - py.c) / 2.0
        return cls(e0=e0, z0=z0, c=c,
                   Ax=px.A, Bx=px.B, dx=px.d,
                   Ay=py.A, By=py.B, dy=py.d)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def zhuang_sigma(z: np.ndarray | float, p: ZhuangParams) -> np.ndarray | float:
    """Evaluate sigma(z) for one axis."""
    z = np.asarray(z, dtype=float)
    X = (z - p.c) / p.d if p.d != 0 else np.full_like(z, np.inf)
    arg = 1.0 + X**2 + p.A * X**3 + p.B * X**4
    with np.errstate(invalid="ignore"):
        return p.sigma0 * np.sqrt(np.maximum(arg, 0.0))


def zhuang_ellipticity(z: np.ndarray | float, q: ZhuangEllipticityParams | np.ndarray) -> np.ndarray | float:
    """Ellipticity e(z) = e0 * sqrt(sigma_x^2 / sigma_y^2).

    q: [e0, z0, c, Ax, Bx, dx, Ay, By, dy]
    """
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    z = np.asarray(z, dtype=float)
    X = (z - q[2] - q[1]) / q[5] if q[5] != 0 else np.full_like(z, np.inf)
    Y = (z + q[2] - q[1]) / q[8] if q[8] != 0 else np.full_like(z, np.inf)
    num = 1.0 + X**2 + q[3] * X**3 + q[4] * X**4
    den = 1.0 + Y**2 + q[6] * Y**3 + q[7] * Y**4
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[0] * np.sqrt(np.maximum(num, 0.0) / np.maximum(den, 1e-30))


# ---------------------------------------------------------------------------
# Fitting: single-axis Zhuang curve
# ---------------------------------------------------------------------------

def _zhuang_initial_guess(z: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Moment-based initial guess for [sigma0, A, B, c, d]."""
    p = np.zeros(5)
    p[0] = float(np.nanmin(sigma))
    p[3] = float(z[np.nanargmin(sigma)])
    with np.errstate(divide="ignore", invalid="ignore"):
        d_est = np.sqrt((z - p[3])**2 / ((sigma / p[0])**2 - 1.0))
    d_est = d_est[np.isfinite(d_est)]
    p[4] = float(np.nanmean(d_est)) if len(d_est) > 0 else 1.0
    if not np.isfinite(p[4]) or p[4] <= 0:
        p[4] = 1.0
    return p


def fit_zhuang_axis(
    z: np.ndarray,
    sigma: np.ndarray,
    dsigma: np.ndarray | None = None,
) -> tuple[ZhuangParams, np.ndarray, tuple[float, float]]:
    """Fit Zhuang defocus curve to one axis.

    Returns (params, d_params, (chi2_reduced, R2)).
    """
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Remove NaN
    mask = np.isfinite(z) & np.isfinite(sigma)
    if dsigma is not None:
        dsigma = np.asarray(dsigma, dtype=float)
        mask &= np.isfinite(dsigma)
        dsigma = dsigma[mask]
    else:
        dsigma = np.ones(mask.sum())
    z = z[mask]
    sigma = sigma[mask]

    nan5 = np.full(5, np.nan)
    if len(z) < 5:
        return ZhuangParams(np.nan, np.nan, np.nan, np.nan, np.nan), nan5, (np.nan, np.nan)

    # Clamp weights away from zero
    dsigma = np.clip(dsigma, 1e-10, None)

    p0 = _zhuang_initial_guess(z, sigma)

    def cost(p):
        model = zhuang_sigma(z, ZhuangParams(*p))
        return float(np.nansum(((sigma - model) / dsigma)**2))

    result = scipy.optimize.minimize(cost, p0, options={"disp": False, "maxiter": 100000})
    q = result.x

    if q.size >= z.size:
        return ZhuangParams(*q), nan5, (np.nan, np.nan)

    # Error estimates from Hessian
    try:
        dq = np.sqrt(result.fun / (z.size - q.size) * np.diag(result.hess_inv))
    except (AttributeError, np.linalg.LinAlgError):
        dq = nan5

    chi2 = result.fun / (z.size - q.size)
    model = zhuang_sigma(z, ZhuangParams(*q))
    ss_res = np.nansum((sigma - model)**2)
    ss_tot = np.nansum((sigma - np.nanmean(sigma))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Retry with unit weights if fit is degenerate
    if np.isnan(zhuang_sigma(0.0, ZhuangParams(*q))):
        return fit_zhuang_axis(z, sigma, None)

    return ZhuangParams(*q), dq, (chi2, r2)


# ---------------------------------------------------------------------------
# Fitting: combined ellipticity model
# ---------------------------------------------------------------------------

def fit_zhuang_ellipticity(
    z: np.ndarray,
    ell: np.ndarray,
    initial: ZhuangEllipticityParams | np.ndarray | None = None,
) -> tuple[ZhuangEllipticityParams, np.ndarray, float]:
    """Fit 9-parameter ellipticity model.

    Returns (params, d_params, chi2_reduced).
    """
    z = np.asarray(z, dtype=float)
    ell = np.asarray(ell, dtype=float)

    mask = np.isfinite(z) & np.isfinite(ell)
    z = z[mask]
    ell = ell[mask]

    nan9 = np.full(9, np.nan)
    if len(z) < 9:
        return ZhuangEllipticityParams(*nan9), nan9, np.nan

    if initial is None:
        p0 = np.array([1.0, float(np.mean(z)), 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4])
    elif isinstance(initial, ZhuangEllipticityParams):
        p0 = initial.to_array()
    else:
        p0 = np.asarray(initial)

    def cost(p):
        return float(np.nansum((ell - zhuang_ellipticity(z, p))**2))

    result = scipy.optimize.minimize(cost, p0, options={"disp": False, "maxiter": 100000})
    q = result.x

    if q.size >= z.size:
        dq = nan9
    else:
        try:
            dq = np.sqrt(result.fun / (z.size - q.size) * np.diag(result.hess_inv))
        except (AttributeError, np.linalg.LinAlgError):
            dq = nan9

    chi2 = result.fun / (z.size - q.size) if z.size > q.size else np.nan
    return ZhuangEllipticityParams.from_array(q), dq, chi2


# ---------------------------------------------------------------------------
# Z lookup from ellipticity (inverse mapping)
# ---------------------------------------------------------------------------

def find_z_from_ellipticity(
    ell: float,
    q: ZhuangEllipticityParams | np.ndarray,
    z_range: tuple[float, float] | None = None,
) -> float:
    """Invert e(z) to find z given measured ellipticity.

    Uses polynomial root finding (exact Zhuang inversion) with fallback to
    numerical minimization.  Returns NaN if no valid root is found.
    """
    if isinstance(q, ZhuangEllipticityParams):
        q_arr = q.to_array()
    else:
        q_arr = np.asarray(q)

    if z_range is None:
        z_range = _find_zhuang_usable_range(q_arr)
        if z_range is None:
            return np.nan

    # Numerical fallback: minimize |e(z) - ell|^2 over the usable range
    z_grid = np.linspace(z_range[0], z_range[1], 200)
    e_grid = zhuang_ellipticity(z_grid, q_arr)
    residuals = np.abs(e_grid - ell)
    i_best = np.nanargmin(residuals)
    z0 = z_grid[i_best]

    # Refine with bounded minimization
    result = scipy.optimize.minimize_scalar(
        lambda z: (zhuang_ellipticity(float(z), q_arr) - ell)**2,
        bounds=(z_range[0], z_range[1]),
        method="bounded",
    )
    z = float(result.x) if result.success else float(z0)

    if z < z_range[0] or z > z_range[1]:
        return np.nan
    return z


def _find_zhuang_usable_range(q: np.ndarray) -> tuple[float, float] | None:
    """Find the monotonic range of the ellipticity curve near focus.

    Returns (z_min, z_max) or None if range cannot be determined.
    """
    z0 = q[1]  # center of focus
    z_test = np.linspace(z0 - 2.0, z0 + 2.0, 2000)
    e_test = zhuang_ellipticity(z_test, q)

    valid = np.isfinite(e_test) & (e_test > 0)
    if not np.any(valid):
        return None

    # Find where de/dz changes sign (extrema of ellipticity)
    de = np.diff(e_test)
    sign_changes = np.where(np.diff(np.sign(de)))[0]

    if len(sign_changes) == 0:
        # Monotonic over entire range
        z_vals = z_test[valid]
        return (float(z_vals[0]), float(z_vals[-1]))

    # Find the extrema bracketing z0
    extrema_z = z_test[sign_changes + 1]
    below = extrema_z[extrema_z < z0]
    above = extrema_z[extrema_z > z0]

    z_lo = float(below[-1]) if len(below) > 0 else float(z_test[valid][0])
    z_hi = float(above[0]) if len(above) > 0 else float(z_test[valid][-1])

    return (z_lo, z_hi)


def zhuang_usable_range(q: ZhuangEllipticityParams | np.ndarray) -> tuple[float, float] | None:
    """Public interface: get the usable Z range for a fitted model."""
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    return _find_zhuang_usable_range(q)


# ---------------------------------------------------------------------------
# Second-moment metric ↔ Zhuang model bridge
# ---------------------------------------------------------------------------

def second_moment_error_from_ellipticity(ell: float) -> float:
    """Convert sigma_x/sigma_y ellipticity ratio to the second-moment error signal.

    The second-moment metric is (var_x - var_y)/(var_x + var_y).
    Since var ~ sigma^2, and ell = sigma_x/sigma_y:
      error = (ell^2 - 1) / (ell^2 + 1)
    """
    e2 = ell**2
    return (e2 - 1.0) / (e2 + 1.0)


def ellipticity_from_second_moment_error(error: float) -> float:
    """Inverse: convert second-moment error signal to ellipticity ratio.

    error = (ell^2 - 1)/(ell^2 + 1)  →  ell^2 = (1 + error)/(1 - error)
    """
    if error >= 1.0:
        return float("inf")
    if error <= -1.0:
        return 0.0
    return math.sqrt((1.0 + error) / (1.0 - error))


# ---------------------------------------------------------------------------
# Build a lookup table for fast runtime Z estimation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZhuangLookupTable:
    """Precomputed lookup table mapping second-moment error signal → z offset.

    Built from a fitted Zhuang ellipticity model for fast runtime evaluation
    without iterative root-finding.
    """
    error_values: np.ndarray  # sorted ascending
    z_offsets: np.ndarray     # corresponding z values
    z_range: tuple[float, float]
    z_center: float           # z at error=0 (focus)

    def error_to_z_offset_um(self, error: float) -> float:
        """Fast interpolated lookup: error signal → z offset in um."""
        if len(self.error_values) < 2:
            return 0.0
        z = float(np.interp(error, self.error_values, self.z_offsets))
        return z - self.z_center


def build_lookup_table(
    q: ZhuangEllipticityParams | np.ndarray,
    n_points: int = 1000,
) -> ZhuangLookupTable:
    """Build a lookup table from a fitted Zhuang model.

    Maps second-moment error signal to z offset for fast runtime use.
    """
    if isinstance(q, ZhuangEllipticityParams):
        q_arr = q.to_array()
    else:
        q_arr = np.asarray(q)

    z_range = _find_zhuang_usable_range(q_arr)
    if z_range is None:
        raise ValueError("Cannot determine usable Z range from Zhuang parameters")

    z_vals = np.linspace(z_range[0], z_range[1], n_points)
    ell_vals = zhuang_ellipticity(z_vals, q_arr)
    error_vals = np.array([second_moment_error_from_ellipticity(float(e)) for e in ell_vals])

    # Ensure monotonic: the error signal should be monotonic in the usable range.
    # If not perfectly monotonic due to numerics, enforce it.
    valid = np.isfinite(error_vals) & np.isfinite(z_vals)
    error_vals = error_vals[valid]
    z_vals = z_vals[valid]

    if len(error_vals) < 2:
        raise ValueError("Not enough valid points for lookup table")

    # Sort by error for np.interp
    order = np.argsort(error_vals)
    error_sorted = error_vals[order]
    z_sorted = z_vals[order]

    # Find z at error=0 (focus)
    z_center = float(np.interp(0.0, error_sorted, z_sorted))

    return ZhuangLookupTable(
        error_values=error_sorted,
        z_offsets=z_sorted,
        z_range=z_range,
        z_center=z_center,
    )
