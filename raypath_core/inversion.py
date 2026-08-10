"""Regularised velocity inversion and uncertainty calculations."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from .constants import (
    DEFAULT_UNCERTAINTY_SEED,
    QUICK_ENSEMBLE_SIZE,
    REPORT_QUALITY_ENSEMBLE_MINIMUM,
    VELOCITY_MAX,
    VELOCITY_MIN,
)
from .models import (
    InversionResult,
    RayPathError,
    RegularizationSelectionResult,
    VelocityUncertaintyResult,
)
from .rays import forward_model, pseudo_interval_velocities
from .vs30 import calculate_ts1170_5_method1_vs30

def _initial_velocity_estimate(
    depths_m: np.ndarray,
    times_s: np.ndarray,
    receiver_offsets_m: np.ndarray,
) -> np.ndarray:
    """Build a stable bounded starting profile from apparent and interval speeds."""

    raw = pseudo_interval_velocities(depths_m, times_s)
    path_speed = np.sqrt(depths_m * depths_m + receiver_offsets_m * receiver_offsets_m) / times_s
    finite_path = path_speed[np.isfinite(path_speed) & (path_speed > 0.0)]
    fallback = float(np.median(finite_path)) if finite_path.size else 250.0
    initial = np.where(np.isfinite(raw) & (raw > 0.0), raw, path_speed)
    initial = np.where(np.isfinite(initial) & (initial > 0.0), initial, fallback)
    # A mild median limiter prevents one bad interval from destabilising the first solve.
    if initial.size >= 3:
        limited = initial.copy()
        for i in range(initial.size):
            lo = max(0, i - 1)
            hi = min(initial.size, i + 2)
            local = float(np.median(initial[lo:hi]))
            limited[i] = np.clip(initial[i], 0.4 * local, 2.5 * local)
        initial = limited
    return np.clip(initial, VELOCITY_MIN, VELOCITY_MAX)


def depth_aware_regularization_operator(depths_m: Sequence[float]) -> np.ndarray:
    """Return a dimensionless finite-difference operator for layered log(Vs).

    For three or more layers the operator approximates the second derivative
    at layer centres, integrates its squared magnitude over depth, and scales
    it by total profile depth.  Consequently its magnitude is substantially
    independent of receiver count and irregular interval spacing.  A two-layer
    profile uses the corresponding dimensionless first derivative.
    """

    depths = np.asarray(depths_m, dtype=float)
    if depths.ndim != 1 or depths.size < 2:
        raise ValueError("Depth-aware regularisation requires at least two layer-bottom depths.")
    if not np.all(np.isfinite(depths)) or not np.all(depths > 0.0) or np.any(np.diff(depths) <= 0.0):
        raise ValueError("Regularisation depths must be finite, positive, and strictly increasing.")
    thicknesses = np.diff(np.r_[0.0, depths])
    centres = depths - 0.5 * thicknesses
    profile_depth = float(depths[-1])
    if depths.size == 2:
        spacing = float(centres[1] - centres[0])
        return np.asarray([[-profile_depth / spacing, profile_depth / spacing]], dtype=float)

    left_spacing = centres[1:-1] - centres[:-2]
    right_spacing = centres[2:] - centres[1:-1]
    support = 0.5 * (left_spacing + right_spacing)
    support_weights = support / float(np.sum(support))
    operator = np.zeros((depths.size - 2, depths.size), dtype=float)
    for row, (left, right, weight) in enumerate(zip(left_spacing, right_spacing, support_weights)):
        scale = profile_depth * profile_depth * math.sqrt(float(weight))
        operator[row, row] = scale * 2.0 / (left * (left + right))
        operator[row, row + 1] = -scale * 2.0 / (left * right)
        operator[row, row + 2] = scale * 2.0 / (right * (left + right))
    return operator


def _travel_time_jacobian_ms(
    thicknesses_m: np.ndarray, velocities_mps: np.ndarray, rays: Sequence[RaySolution]
) -> np.ndarray:
    """Return d(travel time in ms)/d(log Vs) using Fermat's principle."""

    count = velocities_mps.size
    jacobian = np.zeros((len(rays), count), dtype=float)
    for index, ray in enumerate(rays):
        path_lengths = thicknesses_m[: index + 1] / np.cos(ray.angles_rad)
        jacobian[index, : index + 1] = -1000.0 * path_lengths / velocities_mps[: index + 1]
    return jacobian


def invert_velocity_profile(
    depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    source_offset_m: float,
    regularization: float = 0.35,
    progress_callback: Any | None = None,
    receiver_offsets_m: Sequence[float] | None = None,
    observation_std_s: Sequence[float] | None = None,
    robust_loss: str = "linear",
    huber_threshold_sigma: float = 1.5,
) -> InversionResult:
    """Invert arrival times for one velocity per depth interval.

    Velocities are parameterised logarithmically and solved with L-BFGS-B.
    The data term uses one observation standard deviation per arrival and is
    expressed on an equivalent millisecond-squared scale.  ``robust_loss`` may
    be ``"linear"`` or ``"huber"``; neither mode deletes observations.  The
    regularisation term applies a depth-aware, normalised finite-difference
    operator to log velocity.  The UI maps its 0..1 control directly to
    ``regularization``.
    """

    z = np.asarray(depths_m, dtype=float)
    t_obs = np.asarray(observed_times_s, dtype=float)
    if z.ndim != 1 or t_obs.ndim != 1 or z.size != t_obs.size or z.size < 2:
        raise ValueError("At least two matching receiver depths and arrival times are required.")
    if not np.all(np.isfinite(z)) or not np.all(z > 0.0) or not np.all(np.diff(z) > 0.0):
        raise ValueError("Receiver depths must be finite, positive, and strictly increasing.")
    if not np.all(np.isfinite(t_obs)) or not np.all(t_obs > 0.0):
        raise ValueError("Measured arrival times must be finite and greater than zero.")
    if not math.isfinite(source_offset_m) or source_offset_m < 0.0:
        raise ValueError("Source offset must be finite and non-negative.")
    if receiver_offsets_m is None:
        receiver_offsets = np.full(z.size, float(source_offset_m), dtype=float)
    else:
        receiver_offsets = np.asarray(receiver_offsets_m, dtype=float)
        if receiver_offsets.ndim != 1 or receiver_offsets.size != z.size:
            raise ValueError("Receiver offsets must contain one value for every receiver depth.")
        if not np.all(np.isfinite(receiver_offsets)) or not np.all(receiver_offsets >= 0.0):
            raise ValueError("Receiver offsets must be finite and greater than or equal to zero.")
    if observation_std_s is None:
        observation_std = np.full(z.size, 0.001, dtype=float)
    else:
        observation_std = np.asarray(observation_std_s, dtype=float)
        if observation_std.ndim != 1 or observation_std.size != z.size:
            raise ValueError("Observation uncertainty must contain one standard deviation per arrival.")
        if not np.all(np.isfinite(observation_std)) or not np.all(observation_std > 0.0):
            raise ValueError("Observation standard deviations must be finite and greater than zero.")
    loss_name = str(robust_loss).strip().lower()
    if loss_name not in {"linear", "huber"}:
        raise ValueError("Robust loss must be 'linear' or 'huber'.")
    huber_delta = float(huber_threshold_sigma)
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("The Huber threshold must be finite and greater than zero.")
    lam = float(np.clip(regularization, 0.0, 1.0))
    h = np.diff(np.r_[0.0, z])
    smoothing_operator = depth_aware_regularization_operator(z)
    initial = _initial_velocity_estimate(z, t_obs, receiver_offsets)
    lower = math.log(VELOCITY_MIN)
    upper = math.log(VELOCITY_MAX)
    eval_count = 0
    sigma_ms = observation_std * 1000.0
    inverse_variance = 1.0 / (sigma_ms * sigma_ms)
    sigma_reference_ms = math.sqrt(z.size / float(np.sum(inverse_variance)))

    def objective(log_vs: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eval_count
        eval_count += 1
        vs = np.exp(log_vs)
        try:
            t_calc, rays = forward_model(h, vs, source_offset_m, receiver_offsets)
        except (ValueError, RayPathError, FloatingPointError):
            return 1.0e30, np.zeros_like(log_vs)
        residual_ms = (t_calc - t_obs) * 1000.0
        standardized = residual_ms / sigma_ms
        if loss_name == "huber":
            absolute = np.abs(standardized)
            rho = np.where(
                absolute <= huber_delta,
                standardized * standardized,
                2.0 * huber_delta * absolute - huber_delta * huber_delta,
            )
            psi = np.where(
                absolute <= huber_delta,
                2.0 * standardized,
                2.0 * huber_delta * np.sign(standardized),
            )
        else:
            rho = standardized * standardized
            psi = 2.0 * standardized
        data_cost = float(sigma_reference_ms * sigma_reference_ms * np.mean(rho))

        # Fermat's principle removes the derivative of the optimally bent path
        # itself.  For receiver i and crossed layer j, the derivative with
        # respect to log(Vs_j) is simply -path_length_ij / Vs_j.  Supplying
        # this lower-triangular Jacobian avoids O(number_of_layers) numerical
        # forward models for every optimiser step.
        travel_jacobian_ms = _travel_time_jacobian_ms(h, vs, rays)
        gradient = (
            sigma_reference_ms
            * sigma_reference_ms
            / z.size
            * travel_jacobian_ms.T.dot(psi / sigma_ms)
        )

        smooth_vector = smoothing_operator.dot(log_vs)
        smooth_cost = float(np.dot(smooth_vector, smooth_vector))
        smooth_gradient = 2.0 * smoothing_operator.T.dot(smooth_vector)
        if progress_callback is not None and eval_count % 10 == 0:
            progress_callback(eval_count, math.sqrt(max(data_cost, 0.0)))
        # At lambda=1 a factor of 100 makes unit normalised roughness equivalent
        # to a 10 ms RMS data error; lambda=0 is an unregularised fit.
        return data_cost + lam * 100.0 * smooth_cost, gradient + lam * 100.0 * smooth_gradient

    result = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * z.size,
        options={"maxiter": 600, "ftol": 1.0e-12, "gtol": 1.0e-7, "maxls": 40},
    )
    velocities = np.exp(result.x)
    calculated, rays = forward_model(h, velocities, source_offset_m, receiver_offsets)
    residuals = calculated - t_obs
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    residual_ms = residuals * 1000.0
    standardized_residuals = residuals / observation_std
    if loss_name == "huber":
        absolute_standardized = np.abs(standardized_residuals)
        robust_weight = np.where(
            absolute_standardized <= huber_delta,
            1.0,
            huber_delta / np.maximum(absolute_standardized, np.finfo(float).eps),
        )
    else:
        robust_weight = np.ones(z.size, dtype=float)
    normalised_weight = inverse_variance / float(np.mean(inverse_variance))
    weighted_rmse_ms = float(np.sqrt(np.mean(normalised_weight * residual_ms * residual_ms)))
    final_jacobian_ms = _travel_time_jacobian_ms(h, velocities, rays)
    weighted_jacobian = (
        np.sqrt(normalised_weight * robust_weight / z.size)[:, np.newaxis] * final_jacobian_ms
    )
    data_hessian = weighted_jacobian.T.dot(weighted_jacobian)
    regularization_hessian = lam * 100.0 * smoothing_operator.T.dot(smoothing_operator)
    information = data_hessian + regularization_hessian
    information += np.eye(z.size) * max(1.0e-12, float(np.trace(information)) * 1.0e-12 / z.size)
    inverse_information = np.linalg.pinv(information, rcond=1.0e-10)
    resolution = np.clip(np.diag(inverse_information.dot(data_hessian)), 0.0, 1.0)
    hat_matrix = weighted_jacobian.dot(inverse_information).dot(weighted_jacobian.T)
    leverage = np.clip(np.diag(hat_matrix), 0.0, 1.0 - 1.0e-9)
    influence = np.abs(standardized_residuals) * np.sqrt(leverage / np.maximum(1.0 - leverage, 1.0e-9))
    outlier_flags = np.abs(standardized_residuals) > 3.0
    influential_flags = influence > 2.0
    bound_tolerance = 1.0e-3
    bound_active = (result.x <= lower + bound_tolerance) | (result.x >= upper - bound_tolerance)
    smooth_vector = smoothing_operator.dot(result.x)
    roughness_norm = float(np.linalg.norm(smooth_vector))
    regularization_contribution = float(lam * 100.0 * np.dot(smooth_vector, smooth_vector))
    if loss_name == "huber":
        absolute = np.abs(standardized_residuals)
        rho = np.where(
            absolute <= huber_delta,
            standardized_residuals * standardized_residuals,
            2.0 * huber_delta * absolute - huber_delta * huber_delta,
        )
    else:
        rho = standardized_residuals * standardized_residuals
    final_data_cost = float(sigma_reference_ms * sigma_reference_ms * np.mean(rho))
    warnings: list[str] = []
    if np.any(bound_active):
        warnings.append(f"{int(np.count_nonzero(bound_active))} velocity parameter(s) are active at solver bounds.")
    if np.any(resolution < 0.20):
        warnings.append(f"{int(np.count_nonzero(resolution < 0.20))} layer(s) have low linearised resolution (<0.20).")
    if np.any(outlier_flags):
        warnings.append(f"{int(np.count_nonzero(outlier_flags))} observation(s) exceed 3 standard deviations.")
    if np.any(influential_flags):
        warnings.append(f"{int(np.count_nonzero(influential_flags))} observation(s) have influence score above 2.")

    # L-BFGS-B can report a line-search warning after already reaching an
    # excellent finite solution.  Preserve the optimiser status, but only
    # reject a result whose values are not usable.
    usable = bool(np.all(np.isfinite(velocities)) and np.all(np.isfinite(calculated)))
    if not usable:
        raise RayPathError(f"Inversion did not produce a finite model: {result.message}")
    return InversionResult(
        depths_m=z,
        thicknesses_m=h,
        velocities_mps=velocities,
        observed_times_s=t_obs,
        calculated_times_s=calculated,
        residuals_s=residuals,
        ray_parameters=np.asarray([ray.ray_parameter for ray in rays]),
        ray_x_segments=[ray.horizontal_segments_m.copy() for ray in rays],
        receiver_offsets_m=receiver_offsets.copy(),
        rmse_s=rmse,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        weighted_rmse_s=weighted_rmse_ms / 1000.0,
        data_cost_ms2=final_data_cost,
        regularization_cost=regularization_contribution,
        roughness_norm=roughness_norm,
        objective_value=final_data_cost + regularization_contribution,
        observation_std_s=observation_std.copy(),
        standardized_residuals=standardized_residuals,
        resolution_diagonal=resolution,
        observation_leverage=leverage,
        influence_scores=influence,
        outlier_flags=outlier_flags,
        influential_flags=influential_flags,
        bound_active_flags=bound_active,
        robust_loss=loss_name,
        warnings=tuple(warnings),
    )


def select_regularization_lcurve(
    depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    source_offset_m: float,
    receiver_offsets_m: Sequence[float] | None = None,
    observation_std_s: Sequence[float] | None = None,
    robust_loss: str = "linear",
    candidate_factors: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 1.0),
) -> tuple[RegularizationSelectionResult, InversionResult]:
    """Select regularisation using maximum distance from the log-log L-curve chord.

    All candidate models use identical observations, uncertainty weights, loss,
    bounds, and geometry.  The end points define a chord after both log axes are
    normalised to 0..1; the interior candidate farthest toward the L-curve
    corner is selected.  This deterministic criterion supplements, rather than
    removes, the manual slider.
    """

    factors = np.asarray(candidate_factors, dtype=float)
    if factors.ndim != 1 or factors.size < 3 or not np.all(np.isfinite(factors)):
        raise ValueError("L-curve selection requires at least three finite candidate factors.")
    if np.any(factors < 0.0) or np.any(factors > 1.0) or np.any(np.diff(factors) <= 0.0):
        raise ValueError("L-curve candidate factors must be strictly increasing within 0..1.")
    models = [
        invert_velocity_profile(
            depths_m,
            observed_times_s,
            source_offset_m,
            regularization=float(factor),
            receiver_offsets_m=receiver_offsets_m,
            observation_std_s=observation_std_s,
            robust_loss=robust_loss,
        )
        for factor in factors
    ]
    misfit = np.asarray([max(model.weighted_rmse_s * 1000.0, 1.0e-12) for model in models])
    roughness = np.asarray([max(model.roughness_norm, 1.0e-12) for model in models])
    x = np.log10(roughness)
    y = np.log10(misfit)

    def normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.ptp(values))
        return np.zeros_like(values) if span <= 1.0e-12 else (values - float(np.min(values))) / span

    x_norm = normalise(x)
    y_norm = normalise(y)
    start = np.asarray([x_norm[0], y_norm[0]])
    end = np.asarray([x_norm[-1], y_norm[-1]])
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    distances = np.zeros(factors.size, dtype=float)
    if chord_length > 1.0e-12:
        for index in range(1, factors.size - 1):
            point = np.asarray([x_norm[index], y_norm[index]])
            offset = point - start
            distances[index] = abs(float(chord[0] * offset[1] - chord[1] * offset[0])) / chord_length
    selected_index = int(np.argmax(distances[1:-1])) + 1
    selection = RegularizationSelectionResult(
        selected_factor=float(factors[selected_index]),
        candidate_factors=factors,
        weighted_rmse_ms=misfit,
        roughness_norm=roughness,
        chord_distances=distances,
    )
    return selection, models[selected_index]


def generate_velocity_uncertainty_ensemble(
    central_result: InversionResult,
    source_offset_m: float,
    regularization: float,
    ensemble_size: int = QUICK_ENSEMBLE_SIZE,
    random_seed: int = DEFAULT_UNCERTAINTY_SEED,
    robust_loss: str | None = None,
    progress_callback: Any | None = None,
) -> VelocityUncertaintyResult:
    """Perturb arrivals by their 1-sigma uncertainties and reinvert repeatedly."""

    requested = int(ensemble_size)
    if requested < 1:
        raise ValueError("The uncertainty ensemble must request at least one model.")
    seed = int(random_seed)
    if central_result.observation_std_s.size != central_result.depths_m.size:
        raise ValueError("The central model does not contain one uncertainty for every observation.")
    rng = np.random.default_rng(seed)
    velocity_models: list[np.ndarray] = []
    vs30_values: list[float] = []
    failure_count = 0
    selected_loss = robust_loss or central_result.robust_loss
    for index in range(requested):
        perturbed = central_result.observed_times_s + rng.normal(
            0.0, central_result.observation_std_s, central_result.depths_m.size
        )
        if np.any(perturbed <= 0.0):
            failure_count += 1
            continue
        try:
            model = invert_velocity_profile(
                central_result.depths_m,
                perturbed,
                source_offset_m,
                regularization=regularization,
                receiver_offsets_m=central_result.receiver_offsets_m,
                observation_std_s=central_result.observation_std_s,
                robust_loss=selected_loss,
            )
            velocity_models.append(model.velocities_mps)
            try:
                vs30_values.append(
                    calculate_ts1170_5_method1_vs30(model.depths_m, model.velocities_mps).value_mps
                )
            except ValueError:
                pass
        except (ValueError, RayPathError, FloatingPointError):
            failure_count += 1
        if progress_callback is not None:
            progress_callback(index + 1, requested)
    minimum_success = max(3, int(math.ceil(0.80 * requested)))
    if len(velocity_models) < minimum_success:
        raise RayPathError(
            f"Only {len(velocity_models)} of {requested} uncertainty models succeeded; "
            f"at least {minimum_success} are required."
        )
    levels = (2.5, 50.0, 97.5)
    velocity_array = np.asarray(velocity_models, dtype=float)
    velocity_percentiles = np.percentile(velocity_array, levels, axis=0)
    vs30_array = np.asarray(vs30_values, dtype=float)
    if vs30_array.size:
        vs30_percentiles = np.percentile(vs30_array, levels)
        vs30_lower, vs30_median, vs30_upper = (float(value) for value in vs30_percentiles)
    else:
        vs30_lower = vs30_median = vs30_upper = None
    warnings = []
    if requested < REPORT_QUALITY_ENSEMBLE_MINIMUM:
        warnings.append(
            f"{requested} models provide a preliminary sensitivity preview; use at least "
            f"{REPORT_QUALITY_ENSEMBLE_MINIMUM} models for report-quality percentile intervals."
        )
    if failure_count:
        warnings.append(f"{failure_count} perturbed model(s) failed and were excluded from the percentiles.")
    return VelocityUncertaintyResult(
        random_seed=seed,
        requested_models=requested,
        successful_models=len(velocity_models),
        percentile_levels=levels,
        velocity_lower_mps=velocity_percentiles[0],
        velocity_median_mps=velocity_percentiles[1],
        velocity_upper_mps=velocity_percentiles[2],
        vs30_samples_mps=vs30_array,
        vs30_lower_mps=vs30_lower,
        vs30_median_mps=vs30_median,
        vs30_upper_mps=vs30_upper,
        warnings=tuple(warnings),
    )
