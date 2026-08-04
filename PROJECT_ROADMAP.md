# RayPath SCPT — Project State and Development Roadmap

**Document status:** Living project specification  
**Snapshot date:** 5 August 2026  
**Application status:** Development alpha; suitable for internal development and evaluation, but not yet approved for unreviewed engineering use  
**Primary platform:** Windows desktop, Python 3.12, PySide6  
**Units:** SI only — metres (m), milliseconds (ms), and metres per second (m/s)

## 1. Purpose

RayPath SCPT is a desktop application for processing Seismic Cone Penetration Test (SCPT) records. Its principal objective is to reduce false or unphysical interval-velocity spikes by fitting measured shear-wave arrival times with a one-dimensional, refracted ray-path model governed by Snell's Law.

The application is intended to provide a transparent workflow from raw waveform review through arrival-time picking, velocity interpretation, Vs30 calculation, visualisation, and reporting. It is not intended to replace engineering judgement, geological interpretation, field quality control, or independent technical review.

This document records:

- the current implemented state of the application;
- the technical limitations that remain;
- the recommended sequence of development;
- the implementation pathway and acceptance criteria for each work package; and
- the release gates that should be satisfied before producing a portable application.

## 2. Technical reference framework

Development should be informed by the current editions or status of the following sources. The application must identify the method and edition used in every standards-facing output rather than making a generic claim of compliance.

- [MBIE/NZGS Earthquake Geotechnical Engineering Practice, Module 2 — Geotechnical investigations for earthquake engineering, Revision 1](https://www.building.govt.nz/assets/Uploads/building-code-compliance/b-stability/b1-structure/geotechnical-guidelines/module-2-geotech-investigations-earthquake-engineering-version-1.pdf)
- [NZ Ground Investigation Specification](https://www.nzgs.org/libraries/nz-ground-investigation-specification/)
- [ASTM D7400/D7400M-26 — Standard Test Methods for Downhole Seismic Testing](https://store.astm.org/d7400_d7400m-26.html)
- [NZSEE — Site classification methodology for TS 1170.5 design spectra](https://bulletin.nzsee.org.nz/index.php/bnzsee/article/view/1686)
- [Engineering New Zealand — status of TS 1170.5:2025 and NZS 1170.5:2004](https://www.engineeringnz.org/news-insights/new-technical-specifications-for-seismic-design/)
- [NZGS — The use of SCPT and HVSR for site period and subsoil class estimation](https://www.nzgs.org/libraries/the-use-of-scpt-and-hvsr-for-site-period-and-subsoil-class-estimation-2/)

At this snapshot date, NZS 1170.5:2004 remains referenced by the New Zealand Building Code. TS 1170.5:2025 is available as a voluntary Technical Specification and may be used through an Alternative Solution pathway. The software should support both contexts without treating them as interchangeable.

## 3. Current application state

### 3.1 Architecture and dependencies

The application is implemented as a single Python file, `raypath_scpt.py`, with the following dependencies recorded in `requirements.txt`:

- PySide6 6.8 or later;
- NumPy 2.1 or later;
- SciPy 1.14 or later;
- Matplotlib 3.9 or later; and
- ReportLab 4.2 or later.

The current single-file architecture was useful during rapid prototyping. It now makes isolated testing, controlled changes, versioned data migration, and packaging more difficult. Refactoring is recommended after the technical calculation interfaces have been stabilised.

### 3.2 Data import and project state

Implemented:

- Direct import of GOnsite/GORILLA `.GRU` records.
- Interpretation of channels 17 and 18 as the left and right source-direction traces.
- A 50 ms pre-trigger correction applied during GRU import.
- Validation that imported time arrays increase monotonically and span the configured trigger time.
- Editable manual input table for depth and the three currently supported arrival definitions.
- Excel/CSV-style grid paste support.
- Project creation, opening, saving, and Save As using the `.rpscpt` project format.
- CSV export.
- No demonstration or synthetic observations are loaded into a new project.

Current limitations:

- The 50 ms pre-trigger period is a global software constant rather than a visible, project-level import setting.
- Raw recorded time and corrected trigger-relative time are not presented as separate auditable quantities.
- There is no raw-file hash, imported-file manifest, application version, or project change history.
- Instrument identity, calibration, operator, test date, site coordinates, source orientation, inclination, and depth datum are not captured in a structured project metadata model.

### 3.3 Waveform picker

Implemented:

- Paired display of channel 17 in blue and channel 18 in red.
- A maximisable and minimisable picker window.
- Automatic zoom to 25 ms either side of the maximum peak region.
- Guided six-pick workflow: left then right for first peak, first cross, and maximum peak.
- Automatic movement to the next required pick.
- Save-and-next or re-pick prompt after completing an interval.
- Pick mode can be restored after using Matplotlib navigation without losing the current zoom extent.
- Bold, enlarged live pick-value display.
- Suggested picks based on pre-trigger baseline noise, a smoothed trace, amplitude thresholds, local extrema, and individual trace zero crossings.
- A waveform waterfall plot containing the reviewed picks.

Current limitations:

- The implemented "first cross" is an individual zero-axis crossing on each trace. MBIE/NZGS defines the first crossover as the single time at which the reversed-polarity traces cross each other after first arrival.
- Peak/trough terminology and polarity validation are not explicitly enforced.
- Maximum peak is useful for sensitivity comparison but is not a primary arrival definition in the referenced downhole guidance.
- The two independently selected left/right times are arithmetically averaged without first requiring a polarity, phase, timing, or quality-consistency check.
- There is no formal signal-to-noise ratio, clipping check, polarity-reversal score, correlation score, pick uncertainty, acceptance flag, or operator comment.
- Repeated impacts at the same depth cannot yet be retained individually and stacked.

### 3.4 Numerical backend

Implemented:

- A one-dimensional, horizontally layered direct-ray model.
- A common ray parameter, `p = sin(theta) / Vs`, across crossed layers.
- Brent root solving below the critical ray-parameter limit.
- Theoretical travel time calculated along refracted layer segments.
- One calculated travel time and ray geometry for each successively deeper receiver.
- Logarithmic velocity parameterisation bounded between 50 m/s and 2,000 m/s.
- L-BFGS-B minimisation of arrival-time misfit.
- An analytical travel-time gradient based on Fermat's principle.
- A user-adjustable regularisation term that penalises curvature in log velocity.
- Background-thread execution so the GUI remains responsive.
- Comparison models for first peak, first cross, and maximum peak picks.

Current limitations:

- The model assumes horizontal, laterally homogeneous layers and a direct arrival. It does not model dipping layers, lateral variation, anisotropy, head waves, converted waves, or complex non-direct first arrivals.
- Source and receiver geometry uses horizontal offset and nominal receiver depth only.
- One velocity parameter is assigned to every receiver interval, which can imply more subsurface resolution than the observations support.
- The regularisation curvature uses unscaled array differences. Its physical effect therefore depends on receiver spacing when intervals are non-uniform.
- The regularisation slider is an empirical trade-off control, not a physical soil property, and is not selected by an objective criterion.
- All observations receive equal weight. Pick uncertainty and waveform quality do not enter the inversion.
- There is no robust loss, outlier diagnostic, parameter-resolution estimate, covariance result, bootstrap ensemble, or confidence interval.
- No alternative slope-based or geological-layer interpretation is calculated for independent comparison.

### 3.5 Visualisation and results

Implemented:

- Velocity-profile comparison, including raw pseudo-interval and regularised RayPath profiles.
- Ray-path cross-section with a horizontally stretchable plot.
- Observed and calculated arrival-time fit.
- Waveform waterfall with picks.
- Comparison of all available pick-derived velocity models.
- Results table with layer depths, velocity, and fitting error.
- Convergence and RMSE reporting in the status bar.

Current limitations:

- The model plot has no uncertainty envelope or depth-resolution indicator.
- There is no corrected vertical travel-time plot or slope-method interpretation.
- CPT stratigraphy and user-defined geological boundaries cannot be overlaid or used as model constraints.
- Records cannot be visually excluded and immediately re-inverted from the plots.

### 3.6 Vs30 analysis

Implemented:

- `Vs30 = 30 / sum(h_i / Vs_i)` using vertical shear-wave travel time.
- Truncation of layers that cross 30 m.
- Comparison of Vs30 from the three existing pick definitions.
- Recalculation at different smoothing settings.
- Extrapolation from profiles reaching at least 25 m.
- An experimental 0.25-to-4.0 slider that changes the shallower/deeper weighting used to estimate the missing 25-to-30 m interval.

Current limitations:

- The weighted harmonic-mean extrapolation is not the TS 1170.5:2025 Method 1 procedure.
- TS Method 1 extends the last measured layer to 30 m when its depth requirements are met.
- The required or recommended treatment of the 0-to-3 m portion of an SCPT/downhole profile is not implemented in a standards-facing calculation.
- The prescribed method-dependent Vs30 uncertainty bounds are not calculated.
- No site-class boundary crossing or multi-class envelope warning is provided.
- The sensitivity slider could be mistaken for a standards-defined weighting factor unless it is explicitly separated from the primary result.

### 3.7 PDF reporting and branding

Implemented:

- Branded PDF report using the selected Cone & Ray identity and Field Teal theme.
- Full-page engineering plots.
- Velocity, arrival-fit, ray-path, and waveform-waterfall figures.
- Tables of reviewed picks, model results, RMSE, and Vs30 comparisons.
- Omission of the former Vs30 smoothing-sensitivity graph from the PDF while retaining the in-application analysis.

Current limitations:

- The report does not yet provide a complete data-quality schedule, exclusions log, uncertainty statement, standards method/edition declaration, reviewer sign-off, or model-applicability statement.
- Experimental and standards-facing Vs30 results are not yet clearly separated.
- The report cannot establish reproducibility through a raw-file hash, software build identifier, calculation configuration, or complete audit trail.

## 4. Governing development principles

The following decisions should remain in force unless they are deliberately superseded and recorded:

1. All public engineering inputs and outputs remain SI only.
2. Raw imported waveform samples are immutable; corrections and processing are stored as derived data.
3. The 50 ms GRU pre-trigger value remains the default for the known equipment, but becomes visible and configurable per import/project.
4. Standards-facing results and experimental sensitivity results are clearly separated.
5. The existing weighting slider is retained only as experimental sensitivity unless a recognised method explicitly requires it.
6. A result is never labelled simply "NZ compliant". The relevant document, edition, method, assumptions, and uncertainty treatment are named.
7. Maximum peak and individual zero-crossing results may remain for comparison but are not mislabelled as standard peak/trough or crossover methods.
8. Velocity reversals are not automatically suppressed; real reversals are possible and should be assessed against waveform quality and geology.
9. Vs alone is not used as the sole basis for liquefaction assessment.
10. Automated suggestions assist the analyst but do not silently replace reviewed picks.
11. Every engineering result must be reproducible from the saved project, source data, application version, and calculation settings.

## 5. Roadmap overview

| Milestone | Outcome | Release meaning |
|---|---|---|
| M0 — Baseline freeze | Existing behaviour documented and protected by tests | Current internal alpha can be changed safely |
| M1 — Technically corrected picker and Vs30 | Picking terminology, QC, trigger handling, and TS Method 1 corrected | Suitable for controlled internal technical evaluation |
| M2 — Quantified inversion | Depth-aware smoothing, observation weighting, uncertainty, and comparator method implemented | Engineering beta suitable for formal validation |
| M3 — NZ reporting workflow | Dual NZS/TS context, metadata, audit trail, site-period support, and defensible report completed | Release candidate for expert review |
| M4 — Validated portable release | Independent review, reference datasets, regression suite, refactor, installer/portable build, and signed release | Controlled production deployment |

Packaging begins only after the M3 release gates are satisfied and the M4 validation evidence is substantially complete.

## 6. Detailed work packages

### WP-00 — Establish the verified baseline

**Objective:** Protect the current working behaviour before changing calculation definitions or file structures.

**Implementation pathway:**

1. Assign an internal semantic version and expose it in the About dialog, project file, CSV, and PDF.
2. Create a `tests/` directory and separate numerical tests from the GUI entry point.
3. Add deterministic tests for:
   - homogeneous direct rays;
   - two- and multi-layer refracted rays;
   - zero and non-zero offsets;
   - near-critical ray parameters;
   - synthetic inversion recovery;
   - GRU pre-trigger conversion;
   - project save/open round-tripping;
   - all existing Vs30 calculations; and
   - legacy project migration.
4. Preserve at least one de-identified GRU reference dataset and its reviewed project as a regression fixture, subject to data permissions.
5. Record current screenshots and a reference PDF for visual regression review.

**Acceptance criteria:**

- Numerical tests run without starting Qt.
- Known synthetic travel times reproduce to documented tolerances.
- Existing project files load without loss of picks or settings.
- A failed regression test blocks a release build.

**Dependencies:** None. This is the first work package.

### WP-01 — Correct arrival definitions and add waveform QC

**Objective:** Align the picker with recognised downhole seismic interpretation while preserving useful experimental comparisons.

**Implementation pathway:**

1. Introduce distinct pick identifiers:
   - first arrival (FA), optional/manual;
   - first peak/trough (PT), pair-aware;
   - first waveform crossover (CO), one time per reversed pair;
   - cross-correlation interval time (CC), later/optional;
   - individual trace zero crossings, experimental; and
   - maximum absolute peak, experimental.
2. Rename the current `first_cross` data during project migration to avoid silently changing its meaning.
3. Plot a sign-reversed overlay and automatically evaluate whether the two traces butterfly around the selected arrival.
4. Calculate per-depth QC metrics:
   - pre-trigger noise level;
   - signal-to-noise ratio;
   - clipping or constant-value detection;
   - left/right polarity consistency;
   - sign-reversed waveform correlation;
   - left/right PT or experimental-pick disagreement; and
   - sample interval consistency.
5. Add an analyst quality state: accepted, accepted with comment, rejected, or not reviewed.
6. Add an uncertainty field or draggable time band for each accepted arrival.
7. Allow repeated impacts at the same depth to be retained, compared, and stacked rather than overwritten.
8. Provide keyboard shortcuts, undo/redo, and a clear re-pick history for high-volume review.

**Acceptance criteria:**

- CO is stored as one pairwise crossover time, not the average of two zero crossings.
- PT respects expected reversed polarity.
- Experimental picks are visually and textually distinguished.
- Rejected observations are omitted from inversion with a visible audit entry.
- All accepted picks have a review state and uncertainty value or documented default.
- The PDF displays the method definitions, QC result, picks, and exclusions.

**Dependencies:** WP-00.

### WP-02 — Make trigger handling and geometry auditable

**Objective:** Prevent timing and geometry assumptions from becoming hidden sources of velocity error.

**Implementation pathway:**

1. Add a GRU import dialog with a default 50 ms pre-trigger value.
2. Retain both recorded time and trigger-relative time in the in-memory/project data model.
3. Store the applied correction per imported record or import batch.
4. Add optional trigger-channel or trigger-calibration data when future formats provide it.
5. Capture:
   - source offset and its measurement uncertainty;
   - source elevation and ground datum;
   - receiver depth reference;
   - source-block orientation and strike directions;
   - receiver orientation;
   - cone inclination/deviation with depth; and
   - whether depth is measured along rods or corrected vertically.
6. Correct receiver coordinates when inclination data are supplied.
7. Add an applicability warning when source/receiver geometry is incomplete or inconsistent.

**Acceptance criteria:**

- The analyst can reconstruct raw recorded time from saved project data.
- Every report states the applied pre-trigger correction and time reference.
- The ray model uses corrected geometry where supplied.
- Changing the pre-trigger or geometry invalidates existing inversion results and requires a rerun.

**Dependencies:** WP-00. Coordinate with WP-01 project-schema changes.

### WP-03 — Implement standards-facing Vs30 methods

**Objective:** Provide a transparent TS 1170.5:2025 Method 1 calculation without removing exploratory sensitivity tools.

**Implementation pathway:**

1. Create an explicit `Vs30Method` calculation interface rather than embedding all extrapolation logic in one function.
2. Implement TS 1170.5:2025 Method 1:
   - use direct measured Vs over the required profile depth;
   - extend the last measured layer to 30 m when the Method 1 depth requirement is met;
   - apply the specified shallow 0-to-3 m treatment for SCPT/downhole profiles;
   - calculate central, lower-bound, and upper-bound Vs30 using the applicable 5% uncertainty treatment; and
   - record measured, adjusted, and extrapolated thicknesses.
3. Show the raw RayPath profile and standards-adjusted Vs30 calculation together.
4. Move the existing weighting control into an "Experimental sensitivity" section with a persistent warning.
5. Exclude experimental weighting from the primary standards result and standard summary table by default.
6. Add Method 2 as a later sub-feature for profiles with at least 15 m of measured Vs, using an approved Vsz-to-Vs30 relationship, CPT/SPT-derived lower profile, or authoritative geological model with the required uncertainty treatment.
7. Do not implement Method 3 until its data inputs, correlations, provenance, and uncertainty rules can be represented correctly.

**Acceptance criteria:**

- Worked examples from the published TS methodology reproduce within rounding tolerance.
- The report identifies method, measured depth, shallow adjustment, extrapolated interval, uncertainty factor, and result bounds.
- If uncertainty spans more than one TS site class, the result is visibly flagged.
- Experimental weighting cannot be mistaken for the Method 1 result.
- Full-depth profiles are clipped exactly at 30 m.

**Dependencies:** WP-00. Benefits from WP-02 geometry and trigger metadata.

### WP-04 — Improve inversion regularisation and uncertainty

**Objective:** Make the smooth model less dependent on sampling geometry and quantify the confidence justified by the observations.

**Implementation pathway:**

1. Replace unscaled second differences with a depth-aware derivative or finite-difference operator based on layer-centre spacing.
2. Normalise the regularisation term so its behaviour remains comparable across different receiver intervals and numbers of layers.
3. Accept an observation standard deviation for every arrival and minimise weighted residuals.
4. Convert QC metrics into transparent default uncertainty suggestions, while allowing analyst override.
5. Add a robust loss option and identify influential or outlying observations without silently deleting them.
6. Add regularisation selection support using an L-curve, cross-validation, or another documented objective method. Preserve manual override.
7. Generate uncertainty ensembles by perturbing arrivals within their pick uncertainties and rerunning the inversion.
8. Plot velocity confidence envelopes and propagate each ensemble to Vs30.
9. Add sensitivity or resolution indicators for each layer/node.
10. Detect parameters at velocity bounds and report them as possible model inadequacy rather than ordinary convergence.

**Acceptance criteria:**

- Changing from uniform to irregular depth spacing does not materially change an equivalent synthetic solution solely because of the regularisation formulation.
- Low-quality or uncertain picks exert less influence in a documented manner.
- Every model reports both data RMSE and regularisation contribution.
- Uncertainty bands are repeatable when a random seed is stored.
- Bound-active, poorly resolved, or outlier-sensitive intervals are visibly flagged.

**Dependencies:** WP-01 for pick uncertainties and QC. WP-00 for numerical regression coverage.

### WP-05 — Add comparator interpretations and geological constraints

**Objective:** Validate RayPath results against recognised alternative processing and prevent over-parameterised profiles.

**Implementation pathway:**

1. Implement corrected vertical travel-time conversion.
2. Add a depth-versus-corrected-time plot and interactive linear-segment fitting.
3. Calculate slope-method layer velocities.
4. Let the analyst define or import layer boundaries from CPT, borelogs, or an interpreted ground model.
5. Add a piecewise-constant geological-layer RayPath inversion with fewer velocity parameters than observations.
6. Compare:
   - pseudo-interval velocity;
   - slope-method velocity;
   - receiver-interval regularised RayPath velocity; and
   - geological-layer RayPath velocity.
7. Add cross-correlation between successive depths as an independent interval-time interpretation.
8. Present differences as epistemic/model uncertainty, not as a competition in which the smoothest profile automatically wins.

**Acceptance criteria:**

- The slope method reproduces documented hand calculations and a reference dataset.
- User-defined boundaries are stored with source/provenance and appear in plots and reports.
- The application prevents an underdetermined geological-layer inversion.
- Vs30 and site-period comparisons can be produced for every valid interpretation method.

**Dependencies:** WP-01 and WP-04.

### WP-06 — Add New Zealand standards context and site-period analysis

**Objective:** Make the output relevant to both the current Building Code-referenced framework and the emerging TS framework without over-automating site classification.

**Implementation pathway:**

1. Add a project calculation context:
   - NZBC referenced — NZS 1170.5:2004; or
   - Alternative Solution/voluntary — TS 1170.5:2025.
2. Record the selected document edition and method in every exported result.
3. Add quarter-wavelength site-period calculation, `T0 = 4 * sum(h_i / Vs_i)`, where the profile extends to a defensible controlling rock or impedance boundary.
4. Allow the controlling boundary and its evidence to be entered explicitly.
5. Support comparison with an entered HVSR site period.
6. For NZS 1170.5:2004, provide calculated metrics and decision support but do not assign A-to-E class from Vs30 alone.
7. For TS 1170.5:2025, evaluate the Vs30 range and required additional soil criteria before suggesting applicable class or classes.
8. Require analyst confirmation and record the basis for any reported site class.

**Acceptance criteria:**

- A site class cannot be produced from an incomplete set of required inputs.
- The selected standard and edition are visible in the UI and PDF.
- Site period is not presented as the full-profile fundamental period when the controlling boundary is unknown.
- Uncertainty that spans class boundaries produces a multi-class warning/envelope rather than a single unsupported class.

**Dependencies:** WP-03, WP-04, and preferably WP-05.

### WP-07 — Complete metadata, provenance, and engineering reporting

**Objective:** Make an issued result independently reviewable and reproducible.

**Implementation pathway:**

1. Add structured project metadata for client, project, site, test ID, coordinates, date, operator, analyst, reviewer, equipment, calibration, and investigation method.
2. Store a cryptographic hash and original path/name for each imported source file.
3. Record application version, project schema version, calculation settings, random seed, and timestamps.
4. Add a non-destructive calculation history containing each inversion run and the settings that produced it.
5. Add report sections for:
   - scope and purpose;
   - source data and geometry;
   - equipment and acquisition metadata;
   - trigger correction;
   - picking definitions;
   - waveform QC and exclusions;
   - velocity analysis methods;
   - uncertainty and sensitivity;
   - Vs30 method and adjustments;
   - standards context;
   - model assumptions and limitations; and
   - analyst/reviewer sign-off.
6. Export a machine-readable calculation manifest alongside the PDF when requested.
7. Add project schema migrations so older `.rpscpt` files remain readable.

**Acceptance criteria:**

- A reviewer can identify every source record, pick, exclusion, calculation setting, and result from the PDF/project package.
- Re-running an unchanged saved calculation produces the same result within documented numerical tolerance.
- Editing source-dependent settings marks prior results as superseded rather than silently updating them.
- Older supported project schemas migrate with a logged migration record.

**Dependencies:** Schema changes from WP-01 to WP-06 should be substantially stable.

### WP-08 — Validation and independent technical review

**Objective:** Demonstrate that the application is fit for its stated purpose before production distribution.

**Implementation pathway:**

1. Build a validation matrix covering:
   - homogeneous and smoothly varying profiles;
   - sharp stiffness contrasts;
   - genuine velocity reversals;
   - irregular receiver spacing;
   - near-surface refraction;
   - offsets from zero through the expected field range;
   - trigger shifts and noisy records;
   - missing/rejected depths;
   - low-amplitude soft soils;
   - bound-active solutions; and
   - profiles terminating above, at, and below 30 m.
2. Compare forward calculations with independent hand calculations or a separately implemented reference solver.
3. Compare interpreted profiles with an established independent downhole processing workflow where available.
4. Run inter-analyst picking trials to estimate repeatability.
5. Have an experienced New Zealand SCPT practitioner and appropriately qualified geotechnical engineer review:
   - picking definitions;
   - model assumptions;
   - test cases;
   - Vs30 and site-class workflows;
   - uncertainty presentation; and
   - report wording.
6. Log review findings and close material actions before release.

**Acceptance criteria:**

- A signed validation report identifies test cases, expected results, tolerances, outcomes, limitations, and unresolved risks.
- No release-critical review actions remain open.
- The software's stated scope matches what has actually been validated.
- Known model-inapplicability cases produce warnings or are rejected.

**Dependencies:** WP-01 through WP-07.

### WP-09 — Refactor and package the portable application

**Objective:** Produce a maintainable, reproducible Windows application only after the technical workflow is stable.

**Implementation pathway:**

1. Refactor the single file into a package such as:

   ```text
   raypath_scpt/
     __init__.py
     app.py
     core/
       rays.py
       inversion.py
       uncertainty.py
       vs30.py
       site_period.py
     data/
       gru.py
       projects.py
       migrations.py
     picking/
       methods.py
       quality.py
     reporting/
       pdf_report.py
     ui/
       main_window.py
       picker.py
     resources/
   tests/
   ```

2. Add a `pyproject.toml`, locked dependency/build environment, automated formatting, static checks, and test commands.
3. Remove numerical dependencies on Qt so the engineering backend remains independently testable.
4. Package with a Windows-compatible tool such as PyInstaller or Nuitka.
5. Bundle Qt plugins, Matplotlib resources, ReportLab assets, and RayPath branding explicitly.
6. Create both:
   - a portable zipped application folder; and
   - an installer with Start Menu shortcuts and uninstall support.
7. Add code signing when the release process and certificate are available.
8. Test on clean, supported Windows systems without Python or Anaconda installed.
9. Verify opening/saving projects, GRU imports, all plots, background inversion, CSV/PDF export, high-DPI scaling, and paths containing spaces or non-ASCII characters.
10. Publish checksums, release notes, known limitations, and the validation document with each controlled release.

**Acceptance criteria:**

- The packaged application passes the full automated suite and clean-machine acceptance test.
- It does not depend on an installed Python, Spyder, PyQt5, or Anaconda environment.
- A packaged calculation matches the development-environment calculation within documented tolerance.
- Application version, project schema, build identifier, and validation status are visible.
- Upgrade and rollback procedures are documented and tested.

**Dependencies:** WP-00 through WP-08. Packaging work may be prototyped earlier, but no production release should be issued early.

## 7. Recommended immediate development sequence

The next implementation cycle should be limited to the following sequence:

1. **WP-00 baseline tests and versioning.** Freeze trusted current behaviour before data definitions change.
2. **WP-01 pick terminology migration.** Rename the current individual zero crossing and add a true pairwise crossover.
3. **WP-01 QC minimum.** Add polarity, correlation, SNR, acceptance, comment, and uncertainty fields.
4. **WP-02 pre-trigger auditability.** Make 50 ms the visible default and preserve raw/corrected times.
5. **WP-03 TS Method 1.** Implement last-layer extension, shallow adjustment, and 5% bounds.
6. **Report changes.** Clearly separate standards-facing and experimental outputs.

Completion of these six items defines Milestone M1. Depth-aware regularisation and ensemble uncertainty should begin only after the accepted arrival-time data model is stable.

## 8. Release gates

### Internal technical evaluation gate

- WP-00 complete.
- Pick definitions no longer misleading.
- Pre-trigger correction visible and saved.
- Experimental Vs30 weighting clearly labelled.

### Engineering beta gate

- WP-01 through WP-05 substantially complete.
- Automated numerical and project-migration tests passing.
- Vs30 worked examples verified.
- Uncertainty and exclusions visible in UI and report.

### Production release-candidate gate

- WP-06 and WP-07 complete.
- No generic standards-compliance claims.
- Complete reproducibility manifest and limitations statement.
- Validation matrix executed.

### Portable production release gate

- WP-08 review accepted.
- WP-09 clean-machine tests passing.
- Versioned installer/portable archive, checksums, release notes, and rollback package available.

## 9. Deferred and out-of-scope items

The following should not displace the release-critical work:

- Full two- or three-dimensional seismic tomography.
- Automatic interpretation of dipping or laterally variable stratigraphy.
- Automatic liquefaction assessment based only on Vs.
- Automatic site classification without the required geological and geotechnical inputs.
- Cloud storage, multi-user collaboration, or online licensing.
- Additional vendor formats before the GRU workflow and project schema are stable.

Potential later enhancements include CPT/CPTu import, region-appropriate Vs correlation overlays, HVSR data import, AGS-compatible exchange, batch project processing, and controlled company report templates.

## 10. Definition of project success

RayPath SCPT will be ready for controlled production deployment when it can take traceable source waveforms through reviewed arrival picks to a reproducible velocity profile and standards-contextualised Vs30 result, while quantifying uncertainty, exposing assumptions, preserving alternative interpretations, and producing sufficient evidence for independent engineering review.

The success criterion is not simply that the software produces a smooth profile. It is that the profile is technically explainable, independently checkable, appropriately qualified, and no more precise than the field data and model assumptions justify.
