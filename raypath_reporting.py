"""PDF report assembly separated from the desktop window implementation."""

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from raypath_core import *


def build_pdf_report(self: Any, target: Path) -> None:
    """Build a multi-page RayPath SCPT report with charts and tables."""

    try:
        from reportlab.graphics.shapes import Circle, Drawing, Line, Path as ShapePath, Rect
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            KeepTogether,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF reporting requires ReportLab. Install it with `python -m pip install reportlab`."
        ) from exc

    from xml.sax.saxutils import escape

    assert self.result is not None
    selected_kind = str(self.estimator_combo.currentData())
    selected_result = self.result
    report_time = datetime.now().astimezone()
    try:
        report_geometry = calculate_corrected_geometry(
            self._geometry_recorded_depths(),
            self.offset_spin.value(),
            self.survey_geometry,
        )
    except (ValueError, TypeError):
        report_geometry = self.active_geometry
    primary = colors.HexColor("#147D75")
    dark = colors.HexColor("#163F3B")
    copper = colors.HexColor("#C87941")
    mid = colors.HexColor("#5F746F")
    pale = colors.HexColor("#E8F2EF")
    grid = colors.HexColor("#C9D5D2")
    soft = colors.HexColor("#F5F8F7")
    page_width, _ = A4
    content_width = page_width - 30 * mm
    chart_buffers: list[io.BytesIO] = []

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=dark,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BrandWordmark",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=19,
            textColor=dark,
            spaceAfter=0.8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BrandTagline",
            parent=styles["Normal"],
            fontSize=7.8,
            leading=9.5,
            textColor=mid,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BrandMeta",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=mid,
            alignment=TA_RIGHT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=mid,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=primary,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=dark,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportNote",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=mid,
            borderColor=grid,
            borderWidth=0.5,
            borderPadding=6,
            backColor=soft,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.2,
            leading=7.2,
            textColor=colors.white,
        )
    )

    def p(text: Any, style: str = "SmallText") -> Any:
        return Paragraph(escape(str(text)), styles[style])

    def metric_text(value: float, decimals: int = 1) -> str:
        if math.isinf(value):
            return "Inf" if value > 0.0 else "-Inf"
        return f"{value:.{decimals}f}"

    def table_style(header: bool = True, font_size: float = 7.5) -> Any:
        commands: list[tuple[Any, ...]] = [
            ("GRID", (0, 0), (-1, -1), 0.35, grid),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, soft]),
        ]
        if header:
            commands.extend(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), dark),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        return TableStyle(commands)

    def chart_image(draw: Any, height_mm: float = 220.0) -> Any:
        """Render one portrait-oriented chart sized to fill an A4 report page."""

        figure = Figure(figsize=(7.2, 8.8), dpi=150, facecolor="white")
        axes = figure.add_subplot(111)
        draw(axes)
        axes.grid(True, color="#D6E0DD", linewidth=0.6, alpha=0.8)
        axes.tick_params(labelsize=8)
        for spine in axes.spines.values():
            spine.set_color("#819790")
        figure.tight_layout(pad=1.1)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=170, facecolor="white")
        buffer.seek(0)
        chart_buffers.append(buffer)
        return Image(buffer, width=content_width, height=height_mm * mm)

    def draw_velocity(ax: Any) -> None:
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            if comparison is None:
                continue
            ax.stairs(
                comparison.velocities_mps,
                np.r_[0.0, comparison.depths_m],
                orientation="horizontal",
                color=MODEL_COLORS[kind],
                linewidth=2.2 if kind == selected_kind else 1.6,
                label=PICK_LABELS[kind],
            )
            ensemble = self.uncertainty_results.get(kind)
            if kind == selected_kind and ensemble is not None:
                envelope_edges = np.r_[0.0, comparison.depths_m]
                ax.fill_betweenx(
                    envelope_edges,
                    np.r_[ensemble.velocity_lower_mps, ensemble.velocity_lower_mps[-1]],
                    np.r_[ensemble.velocity_upper_mps, ensemble.velocity_upper_mps[-1]],
                    step="post",
                    color=MODEL_COLORS[kind],
                    alpha=0.18,
                    label=(
                        "Preliminary 2.5-97.5% pick-time ensemble"
                        if ensemble.requested_models < REPORT_QUALITY_ENSEMBLE_MINIMUM
                        else "Report-quality 2.5-97.5% pick-time ensemble"
                    ),
                )
        ax.set_title("Optimized shear-wave velocity comparison", fontsize=11, fontweight="bold")
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Corrected vertical depth (m)")
        ax.set_ylim(float(selected_result.depths_m[-1]), 0.0)
        ax.legend(fontsize=8, frameon=True)

    def draw_fit(ax: Any) -> None:
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            if comparison is None:
                continue
            ax.plot(
                comparison.calculated_times_s * 1000.0,
                comparison.depths_m,
                "-s",
                color=MODEL_COLORS[kind],
                linewidth=2.0 if kind == selected_kind else 1.2,
                markersize=3,
                label=f"{PICK_LABELS[kind]} calculated",
            )
            ax.scatter(
                comparison.observed_times_s * 1000.0,
                comparison.depths_m,
                facecolors="none",
                edgecolors=MODEL_COLORS[kind],
                s=20,
            )
            if kind == selected_kind and comparison.observation_std_s.size == comparison.depths_m.size:
                ax.errorbar(
                    comparison.observed_times_s * 1000.0,
                    comparison.depths_m,
                    xerr=comparison.observation_std_s * 1000.0,
                    fmt="none",
                    ecolor=MODEL_COLORS[kind],
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.7,
                )
                flagged = comparison.outlier_flags | comparison.influential_flags
                if np.any(flagged):
                    ax.scatter(
                        comparison.observed_times_s[flagged] * 1000.0,
                        comparison.depths_m[flagged],
                        marker="x",
                        color="#C84A3A",
                        s=38,
                        linewidths=1.4,
                        label="Outlier/influential observation",
                    )
        ax.set_title("Observed and calculated arrival times", fontsize=11, fontweight="bold")
        ax.set_xlabel("Post-trigger travel time (ms)")
        ax.set_ylabel("Corrected vertical depth (m)")
        ax.set_ylim(float(selected_result.depths_m[-1]), 0.0)
        ax.tick_params(axis="y", labelleft=True, colors="#111111")
        ax.yaxis.label.set_color("#111111")
        ax.legend(fontsize=7.5, frameon=True)

    def draw_rays(ax: Any) -> None:
        edges = np.r_[0.0, selected_result.depths_m]
        offset = self.offset_spin.value()
        receiver_offsets = selected_result.receiver_offsets_m
        plot_min = min(0.0, offset, float(np.min(receiver_offsets)))
        plot_max = max(0.0, offset, float(np.max(receiver_offsets)))
        for boundary in edges:
            ax.hlines(boundary, plot_min, plot_max, color="#B8C4CC", linewidth=0.55)
        cmap = colormaps["viridis"]
        for index, segments in enumerate(selected_result.ray_x_segments):
            ax.plot(
                np.r_[0.0, np.cumsum(segments)],
                edges[: index + 2],
                color=cmap((index + 1) / len(selected_result.ray_x_segments)),
                linewidth=1.0,
            )
        ax.plot(
            np.r_[offset, receiver_offsets],
            np.r_[0.0, selected_result.depths_m],
            color="#163F3B",
            linewidth=1.5,
            label="Corrected receiver path",
        )
        ax.scatter([0.0], [0.0], marker="*", s=90, color="#C87941", zorder=5, label="Source")
        ax.scatter(receiver_offsets, selected_result.depths_m, marker="<", s=18, color="#147D75", zorder=5, label="Receivers")
        ax.set_title(f"Ray paths - {PICK_LABELS.get(selected_kind, selected_kind)} model", fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Corrected vertical depth (m)")
        ax.set_ylim(float(selected_result.depths_m[-1]) * 1.03, -float(selected_result.depths_m[-1]) * 0.03)
        margin = max((plot_max - plot_min) * 0.08, 0.1)
        ax.set_xlim(plot_min - margin, plot_max + margin)
        ax.set_aspect("auto")
        ax.legend(fontsize=8, frameon=True)

    def draw_waterfall(ax: Any) -> None:
        self._plot_waveform_waterfall(ax, dark_theme=False)
        ax.set_title("Paired SCPT waveform waterfall", fontsize=11, fontweight="bold")

    def cone_ray_logo(size_mm: float = 18.0) -> Any:
        """Return the Field Teal Cone & Ray mark as vector ReportLab art."""

        size = size_mm * mm
        drawing = Drawing(size, size)
        drawing.add(
            Rect(
                0.5 * mm,
                0.5 * mm,
                size - 1.0 * mm,
                size - 1.0 * mm,
                rx=3.2 * mm,
                ry=3.2 * mm,
                fillColor=pale,
                strokeColor=primary,
                strokeWidth=0.65,
            )
        )
        for y_mm in (5.0, 9.0, 13.0):
            drawing.add(
                Line(
                    2.3 * mm,
                    y_mm * mm,
                    15.8 * mm,
                    y_mm * mm,
                    strokeColor=colors.HexColor("#8FBDB7"),
                    strokeWidth=0.45,
                )
            )
        drawing.add(
            Line(15.0 * mm, 2.4 * mm, 15.0 * mm, 15.7 * mm, strokeColor=dark, strokeWidth=1.15)
        )
        ray_points = (
            ((3.0, 15.0), (6.0, 13.2), (10.1, 12.6), (15.0, 13.0)),
            ((3.0, 15.0), (5.4, 10.7), (9.7, 8.7), (15.0, 9.0)),
            ((3.0, 15.0), (5.0, 7.0), (9.2, 4.8), (15.0, 5.0)),
        )
        for points in ray_points:
            path = ShapePath()
            path.moveTo(points[0][0] * mm, points[0][1] * mm)
            for x_mm, y_mm in points[1:]:
                path.lineTo(x_mm * mm, y_mm * mm)
            path.strokeColor = primary
            path.strokeWidth = 0.9
            path.fillColor = None
            drawing.add(path)
        drawing.add(Circle(3.0 * mm, 15.0 * mm, 1.05 * mm, fillColor=copper, strokeColor=copper))
        for y_mm in (5.0, 9.0, 13.0):
            receiver = ShapePath()
            receiver.moveTo(14.0 * mm, y_mm * mm)
            receiver.lineTo(15.0 * mm, (y_mm + 0.65) * mm)
            receiver.lineTo(15.0 * mm, (y_mm - 0.65) * mm)
            receiver.closePath()
            receiver.fillColor = primary
            receiver.strokeColor = primary
            drawing.add(receiver)
        return drawing

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="RayPath SCPT Engineering Report",
        author="RayPath SCPT",
        subject="SCPT ray-path inversion and Vs30 analysis",
    )

    def page_decoration(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(grid)
        canvas.setLineWidth(0.4)
        canvas.line(15 * mm, 12 * mm, page_width - 15 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(mid)
        canvas.setFillColor(copper)
        canvas.circle(16 * mm, 8.2 * mm, 1.0 * mm, fill=1, stroke=0)
        canvas.setFillColor(mid)
        canvas.drawString(
            19 * mm,
            7.5 * mm,
            f"RayPath SCPT v{APP_VERSION} - Engineering interpretation report",
        )
        canvas.drawRightString(page_width - 15 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    story: list[Any] = []
    project_name = self.project_path.stem if self.project_path else "Untitled"
    brand_header = Table(
        [
            [
                cone_ray_logo(),
                [
                    Paragraph('RayPath <font color="#147D75">SCPT</font>', styles["BrandWordmark"]),
                    Paragraph("Forward ray-path shear-wave interpretation", styles["BrandTagline"]),
                ],
                Paragraph(
                    f"<b>PROJECT</b> {escape(project_name)}<br/>{report_time.strftime('%d %b %Y').upper()}",
                    styles["BrandMeta"],
                ),
            ]
        ],
        colWidths=[21 * mm, 104 * mm, content_width - 125 * mm],
    )
    brand_header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 3),
                ("LEFTPADDING", (1, 0), (1, 0), 2),
                ("RIGHTPADDING", (1, 0), (1, 0), 3),
                ("LEFTPADDING", (2, 0), (2, 0), 3),
                ("RIGHTPADDING", (2, 0), (2, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ("LINEBELOW", (0, 0), (-1, -1), 2.0, primary),
            ]
        )
    )
    story.append(brand_header)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("SCPT Engineering Interpretation Report", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Forward ray-path shear-wave velocity inversion, arrival-time comparison, and Vs30 summary",
            styles["ReportSubtitle"],
        )
    )

    metadata = [
        [p("Report generated"), p(report_time.strftime("%Y-%m-%d %H:%M %Z"))],
        [p("Application version"), p(APP_VERSION)],
        [p("Project schema"), p(PROJECT_SCHEMA_VERSION)],
        [p("Project"), p(self.project_path.name if self.project_path else "Untitled")],
        [p("GRU source"), p(self.gru_path.name if self.gru_path else "Manual / CSV observations")],
        [p("Source offset"), p(f"{self.offset_spin.value():.3f} m")],
        [
            p("Source-offset uncertainty"),
            p(
                "Not recorded"
                if self.survey_geometry.source_offset_uncertainty_m is None
                else f"±{self.survey_geometry.source_offset_uncertainty_m:.3f} m"
            ),
        ],
        [p("Recorded depth basis"), p(DEPTH_BASIS_LABELS[self.survey_geometry.depth_basis])],
        [p("Receiver depth reference"), p(self.survey_geometry.receiver_depth_reference or "Not recorded")],
        [p("Coordinate system"), p(self.survey_geometry.coordinate_system or "Not recorded")],
        [p("Vertical datum"), p(self.survey_geometry.vertical_datum or "Not recorded")],
        [
            p("Source / reference elevations"),
            p(
                f"{self.survey_geometry.source_elevation_m:.3f} / "
                f"{self.survey_geometry.receiver_reference_elevation_m:.3f} m"
                if self.survey_geometry.elevations_enabled
                else "Not applied"
            ),
        ],
        [p("Selected detailed model"), p(PICK_LABELS.get(selected_kind, selected_kind))],
        [
            p("Inversion objective"),
            p(
                f"{'L-curve selected' if self.auto_regularization_check.isChecked() else 'Manual'} "
                f"regularisation {self.reg_slider.value() / 100.0:.2f}; {selected_result.robust_loss} loss; "
                f"RMSE {selected_result.rmse_s * 1000.0:.3f} ms; weighted RMSE "
                f"{selected_result.weighted_rmse_s * 1000.0:.3f} ms; data cost "
                f"{selected_result.data_cost_ms2:.4g} ms^2; regularisation contribution "
                f"{selected_result.regularization_cost:.4g}"
            ),
        ],
        [
            p("Arrival uncertainty weighting"),
            p(
                f"One sigma per observation; selected-model range "
                f"{np.min(selected_result.observation_std_s) * 1000.0:.3f}-"
                f"{np.max(selected_result.observation_std_s) * 1000.0:.3f} ms; fallback "
                f"{self.manual_uncertainty_spin.value():.3f} ms"
            ),
        ],
        [
            p("Uncertainty ensemble"),
            p(
                "Off"
                if selected_kind not in self.uncertainty_results
                else (
                    f"{uncertainty_ensemble_classification(self.uncertainty_results[selected_kind].requested_models)}; "
                    f"{self.uncertainty_results[selected_kind].successful_models}/"
                    f"{self.uncertainty_results[selected_kind].requested_models} successful; seed "
                    f"{self.uncertainty_results[selected_kind].random_seed}; velocity and Vs30 percentiles "
                    "2.5/50/97.5; Vs30 median/95% interval "
                    f"{self.uncertainty_results[selected_kind].vs30_median_mps:.1f} / "
                    f"{self.uncertainty_results[selected_kind].vs30_lower_mps:.1f}-"
                    f"{self.uncertainty_results[selected_kind].vs30_upper_mps:.1f} m/s"
                )
            ),
        ],
        [
            p("Primary Vs30 method"),
            p(
                "TS 1170.5:2025 Method 1 - direct measured Vs"
                if self.current_vs30 is None
                else (
                    f"TS 1170.5:2025 Method 1; raw/adjusted "
                    f"{self.current_vs30.raw_value_mps:.1f}/{self.current_vs30.value_mps:.1f} m/s; "
                    f"5% bounds {self.current_vs30.lower_bound_mps:.1f}-"
                    f"{self.current_vs30.upper_bound_mps:.1f} m/s; 0-3 m set to "
                    f"{self.current_vs30.shallow_reference_velocity_mps:.1f} m/s; bands "
                    f"{'/'.join(self.current_vs30.indicative_vs30_bands)}"
                )
            ),
        ],
        [
            p("Experimental extrapolation weighting"),
            p(f"{self._extrapolation_weight_factor():.2f} - excluded from the primary Method 1 result"),
        ],
        [p("Deepest receiver"), p(f"{selected_result.depths_m[-1]:.2f} m")],
        [
            p("GRU pre-trigger correction"),
            p(
                f"{self.waveform_records[0].pre_trigger_ms:.3f} ms"
                if self.waveform_records
                else (
                    f"{self.gru_pre_trigger_ms:.3f} ms (saved source unavailable)"
                    if self.gru_pre_trigger_ms is not None
                    else "Not applicable — manual/CSV observations"
                )
            ),
        ],
        [
            p("Arrival-time reference"),
            p("Milliseconds relative to physical trigger; recorded-clock picks retained in project audit data"),
        ],
    ]
    if self.observation_review:
        accepted_count, rejected_count, unreviewed_count = self._review_counts()
        metadata.extend(
            [
                [p("Waveform review"), p(
                    f"{accepted_count} accepted; {rejected_count} rejected/excluded; "
                    f"{unreviewed_count} not reviewed"
                )],
                [p("QC warning thresholds"), p(
                    f"SNR < {QC_SNR_WARNING_DB:g} dB; sign-reversed correlation < "
                    f"{QC_CORRELATION_WARNING:.2f}; PT disagreement > max("
                    f"{QC_PICK_DISAGREEMENT_WARNING_MS:g} ms, two samples); sample-interval variation > 1%"
                )],
            ]
        )
    metadata_table = Table(metadata, colWidths=[58 * mm, content_width - 58 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, grid),
                ("BACKGROUND", (0, 0), (0, -1), pale),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(metadata_table)
    story.append(PageBreak())
    story.append(Paragraph("Model comparison", styles["SectionHeading"]))
    comparison_rows: list[list[Any]] = [[
        p("Pick definition", "TableHeader"),
        p("Layers", "TableHeader"),
        p("RMSE / weighted (ms)", "TableHeader"),
        p("Reg. contribution", "TableHeader"),
        p("TS M1 Vs30 (m/s)", "TableHeader"),
        p("5% range (m/s)", "TableHeader"),
        p("Vs30 bands", "TableHeader"),
        p("30 m extension", "TableHeader"),
    ]]
    for kind in PICK_KINDS:
        comparison = self.comparison_results.get(kind)
        vs30 = self.comparison_vs30.get(kind)
        if comparison is None:
            comparison_rows.append(
                [p(PICK_LABELS[kind]), p("-"), p("-"), p("-"), p("-"), p("-"), p("-"), p("Not calculated")]
            )
            continue
        if vs30 is None:
            vs30_text = "Unavailable"
            bounds_text = "-"
            bands_text = "-"
            extrapolation_text = self.comparison_vs30_reasons.get(kind, "Insufficient depth")
        else:
            vs30_text = f"{vs30.value_mps:.1f}"
            bounds_text = f"{vs30.lower_bound_mps:.1f}-{vs30.upper_bound_mps:.1f}"
            bands_text = "/".join(vs30.indicative_vs30_bands)
            extrapolation_text = (
                "None - measured to at least 30 m"
                if vs30.extrapolated_thickness_m <= 0.0
                else f"{vs30.extrapolated_thickness_m:.2f} m at last-layer {vs30.extrapolated_velocity_mps:.1f} m/s"
            )
        comparison_rows.append(
            [
                p(PICK_LABELS[kind]),
                p(comparison.depths_m.size),
                p(f"{comparison.rmse_s * 1000.0:.3f} / {comparison.weighted_rmse_s * 1000.0:.3f}"),
                p(f"{comparison.regularization_cost:.3g}"),
                p(vs30_text),
                p(bounds_text),
                p(bands_text),
                p(extrapolation_text),
            ]
        )
    comparison_table = Table(
        comparison_rows,
        colWidths=[30 * mm, 11 * mm, 25 * mm, 16 * mm, 23 * mm, 25 * mm, 15 * mm, content_width - 145 * mm],
        repeatRows=1,
    )
    comparison_table.setStyle(table_style(font_size=7.5))
    story.append(comparison_table)
    story.append(Spacer(1, 3 * mm))
    if any(
        value is not None
        for value in (self.slope_result, self.geological_result, self.cross_correlation_result)
    ):
        story.append(Paragraph("Comparator interpretations and geological constraints", styles["SectionHeading"]))
        boundaries = self._comparator_boundaries()
        provenance = self.comparator_provenance_edit.text().strip() or "Not recorded"
        story.append(
            Paragraph(
                "Internal geological boundaries: "
                + (", ".join(f"{value:.3f} m" for value in boundaries) if boundaries else "none — single layer")
                + f". Source/provenance: {escape(provenance)}. Comparator arrival definition: "
                + escape(PICK_LABELS.get(self.comparator_pick_kind, self.comparator_pick_kind or "not recorded"))
                + ".",
                styles["ReportNote"],
            )
        )
        comparator_rows: list[list[Any]] = [[
            p("Interpretation", "TableHeader"),
            p("Layer", "TableHeader"),
            p("Top (m)", "TableHeader"),
            p("Bottom (m)", "TableHeader"),
            p("Vs (m/s)", "TableHeader"),
            p("RMSE (ms)", "TableHeader"),
            p("TS M1 Vs30 (m/s)", "TableHeader"),
        ]]

        def comparator_vs30_text(kind: str) -> str:
            value = self.comparator_vs30.get(kind)
            if value is None:
                return "-"
            return f"{value.value_mps:.1f} [{value.lower_bound_mps:.1f}-{value.upper_bound_mps:.1f}]"

        if self.slope_result is not None:
            for index, layer in enumerate(self.slope_result.layers):
                comparator_rows.append([
                    p("Corrected-time slope"),
                    p(index + 1),
                    p(f"{layer.top_depth_m:.3f}"),
                    p(f"{layer.bottom_depth_m:.3f}"),
                    p(f"{layer.velocity_mps:.1f}"),
                    p(f"{layer.rmse_s * 1000.0:.3f}"),
                    p(comparator_vs30_text("slope") if index == 0 else ""),
                ])
        if self.geological_result is not None:
            for index, (top, bottom, velocity) in enumerate(zip(
                self.geological_result.layer_tops_m,
                self.geological_result.layer_bottoms_m,
                self.geological_result.velocities_mps,
            )):
                comparator_rows.append([
                    p("Geological RayPath"),
                    p(index + 1),
                    p(f"{top:.3f}"),
                    p(f"{bottom:.3f}"),
                    p(f"{velocity:.1f}"),
                    p(f"{self.geological_result.rmse_s * 1000.0:.3f}" if index == 0 else ""),
                    p(comparator_vs30_text("geological") if index == 0 else ""),
                ])
        if self.cross_correlation_result is not None:
            tops = np.r_[0.0, self.cross_correlation_result.depths_m[:-1]]
            for index, (top, bottom, velocity) in enumerate(zip(
                tops,
                self.cross_correlation_result.depths_m,
                self.cross_correlation_result.velocities_mps,
            )):
                comparator_rows.append([
                    p("Successive-depth correlation"),
                    p(index + 1),
                    p(f"{top:.3f}"),
                    p(f"{bottom:.3f}"),
                    p(f"{velocity:.1f}"),
                    p(f"{self.cross_correlation_result.rmse_s * 1000.0:.3f}" if index == 0 else ""),
                    p(comparator_vs30_text("cross_correlation") if index == 0 else ""),
                ])
        comparator_table = Table(
            comparator_rows,
            colWidths=[42 * mm, 13 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, content_width - 139 * mm],
            repeatRows=1,
        )
        comparator_table.setStyle(table_style(font_size=7.3))
        story.append(comparator_table)
        story.append(
            Paragraph(
                "The corrected-time slope method uses the conventional straight-ray cosine correction and is an "
                "independent comparator, not a refracted-ray solution. Geological RayPath uses fewer velocity "
                "parameters than arrival observations. Cross-correlation uses successive-depth waveform lag and "
                "requires analyst review of correlation quality.",
                styles["ReportNote"],
            )
        )
        story.append(Spacer(1, 3 * mm))
    selected_ensemble = self.uncertainty_results.get(selected_kind)
    if (
        selected_ensemble is not None
        and selected_ensemble.requested_models < REPORT_QUALITY_ENSEMBLE_MINIMUM
    ):
        story.append(
            Paragraph(
                f"<b>PRELIMINARY UNCERTAINTY PREVIEW:</b> This report contains only "
                f"{selected_ensemble.requested_models} requested perturbation models. The percentile envelope "
                f"is a sensitivity preview, not the final reported uncertainty. Rerun with at least "
                f"{REPORT_QUALITY_ENSEMBLE_MINIMUM} models (the Final report preset uses "
                f"{FINAL_ENSEMBLE_SIZE}) before issue.",
                styles["ReportNote"],
            )
        )
        story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "Arrival definitions: First peak/trough is the mean of the reviewed left and right extrema; "
            "Pair crossover is one reviewed time where the reversed traces intersect after arrival; "
            "Individual zero crossing is the mean of two per-trace zero-axis crossings and is experimental; "
            "Maximum peak is the mean of two maximum-amplitude times and is experimental.",
            styles["ReportNote"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "The primary Vs30 follows TS 1170.5:2025 Method 1 for direct SCPT measurements: 0-3 m is "
            "assigned the depth-average measured from 2.5-3.5 m, profiles reaching at least 25 m extend "
            "the last measured layer to 30 m, and the reported bounds are Vs30/1.05 and 1.05 x Vs30. "
            "The listed bands use Vs30 thresholds only and are not a final site classification; all additional "
            "TS soil and rock criteria require engineering review. Experimental interval weighting is excluded "
            "from these primary results.",
            styles["ReportNote"],
        )
    )

    full_page_plots = [
        ("Velocity profile comparison", draw_velocity),
        ("Arrival-time fit comparison", draw_fit),
        ("Selected-model ray paths", draw_rays),
    ]
    if self.waveform_records:
        full_page_plots.append(("Waveform waterfall and reviewed picks", draw_waterfall))
    for heading, draw_plot in full_page_plots:
        story.append(PageBreak())
        story.append(
            KeepTogether(
                [
                    Paragraph(heading, styles["SectionHeading"]),
                    chart_image(draw_plot),
                ]
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Receiver pick times", styles["SectionHeading"]))
    story.append(Spacer(1, 2 * mm))
    pick_rows: list[list[Any]] = [
        [
            p("Depth (m)", "TableHeader"),
            *[p(f"{PICK_LABELS[kind]} (ms)", "TableHeader") for kind in PICK_KINDS],
        ]
    ]
    for row in range(self.input_table.rowCount()):
        values = []
        for column in range(1 + len(PICK_KINDS)):
            item = self.input_table.item(row, column)
            values.append(item.text().strip() if item else "")
        if not any(values):
            continue
        pick_rows.append([p(value or "-") for value in values])
    pick_table = LongTable(
        pick_rows,
        colWidths=[28 * mm, *([38 * mm] * len(PICK_KINDS))],
        repeatRows=1,
    )
    pick_table.setStyle(table_style(font_size=7.2))
    story.append(pick_table)

    story.append(PageBreak())
    story.append(Paragraph("Survey geometry and corrected receiver coordinates", styles["SectionHeading"]))
    orientation_rows = [
        ["Geometry item", "Recorded value"],
        [p("Source to sounding bearing"), p(
            "Not recorded" if self.survey_geometry.source_to_receiver_bearing_deg is None
            else f"{self.survey_geometry.source_to_receiver_bearing_deg:.3f}°"
        )],
        [p("Shear-plank/block axis bearing"), p(
            "Not recorded" if self.survey_geometry.source_block_axis_bearing_deg is None
            else f"{self.survey_geometry.source_block_axis_bearing_deg:.3f}°"
        )],
        [p("Channel 17 / left strike bearing"), p(
            "Not recorded" if self.survey_geometry.channel_17_strike_bearing_deg is None
            else f"{self.survey_geometry.channel_17_strike_bearing_deg:.3f}°"
        )],
        [p("Channel 18 / right strike bearing"), p(
            "Not recorded" if self.survey_geometry.channel_18_strike_bearing_deg is None
            else f"{self.survey_geometry.channel_18_strike_bearing_deg:.3f}°"
        )],
        [p("Receiver orientation bearing"), p(
            "Not recorded" if self.survey_geometry.receiver_orientation_bearing_deg is None
            else f"{self.survey_geometry.receiver_orientation_bearing_deg:.3f}°"
        )],
        [p("Geometry notes"), p(self.survey_geometry.notes or "None")],
    ]
    orientation_table = Table(orientation_rows, colWidths=[65 * mm, content_width - 65 * mm])
    orientation_table.setStyle(table_style(font_size=7.2))
    story.append(orientation_table)
    story.append(Spacer(1, 3 * mm))
    if report_geometry is None:
        story.append(Paragraph("Corrected receiver coordinates could not be calculated.", styles["ReportNote"]))
    else:
        warning_text = "None" if not report_geometry.warnings else "; ".join(report_geometry.warnings)
        story.append(Paragraph(f"Applicability warnings: {escape(warning_text)}", styles["ReportNote"]))
        point_by_depth = {
            round(point.recorded_depth_m, 6): point
            for point in self.survey_geometry.deviation_points
        }
        geometry_rows: list[list[Any]] = [[
            "Recorded depth",
            "Inclination",
            "Azimuth",
            "Corrected vertical depth",
            "Receiver offset",
            "East",
            "North",
        ]]
        for recorded, vertical, offset, east, north in zip(
            report_geometry.recorded_depths_m,
            report_geometry.vertical_depths_m,
            report_geometry.receiver_offsets_m,
            report_geometry.receiver_east_m,
            report_geometry.receiver_north_m,
        ):
            point = point_by_depth.get(round(float(recorded), 6))
            geometry_rows.append([
                p(f"{recorded:.3f}"),
                p("-" if point is None else f"{point.inclination_deg:.3f}°"),
                p("-" if point is None or point.azimuth_deg is None else f"{point.azimuth_deg:.3f}°"),
                p(f"{vertical:.3f}"),
                p(f"{offset:.3f}"),
                p(f"{east:.3f}"),
                p(f"{north:.3f}"),
            ])
        geometry_table = LongTable(
            geometry_rows,
            colWidths=[25 * mm, 22 * mm, 22 * mm, 31 * mm, 25 * mm, 22 * mm, 22 * mm],
            repeatRows=1,
        )
        geometry_table.setStyle(table_style(font_size=6.5))
        story.append(geometry_table)

    if self.waveform_records:
        story.append(PageBreak())
        story.append(Paragraph("Waveform QC and exclusions schedule", styles["SectionHeading"]))
        story.append(
            Paragraph(
                "QC metrics are deterministic review aids. A warning does not automatically reject an "
                "observation. Only records explicitly marked Rejected are excluded from inversion. "
                "Recorded arrival uncertainties are interpreted as one standard deviation and enter the "
                "weighted inversion. A/Q denotes analyst override or automatic QC default; the full basis "
                "is retained in project and CSV audit data.",
                styles["ReportNote"],
            )
        )
        qc_rows: list[list[Any]] = [
            [
                "Depth",
                "Review state",
                "Unc. ms/source",
                "SNR L/R",
                "Noise RMS L/R",
                "Corr.",
                "Lag",
                "Delta PT/Z/M",
                "Polarity",
                "Warnings",
            ]
        ]
        for record in self.waveform_records:
            qc = calculate_waveform_qc(record)
            qc_rows.append(
                [
                    p(f"{record.depth_m:.2f}"),
                    p(REVIEW_LABELS.get(record.review_state, record.review_state)),
                    p(
                        "-"
                        if record.pick_uncertainty_ms is None
                        else f"{record.pick_uncertainty_ms:.3f} "
                        f"{'A' if record.pick_uncertainty_source == 'analyst_override' else 'Q'}"
                    ),
                    p(f"{metric_text(qc.snr_left_db)}/{metric_text(qc.snr_right_db)}"),
                    p(f"{qc.noise_rms_left:.3g}/{qc.noise_rms_right:.3g}"),
                    p(f"{qc.sign_reversed_correlation:.3f}"),
                    p(f"{qc.correlation_lag_ms:+.2f}"),
                    p(
                        "/".join(
                                "-" if value is None else f"{value:.1f}"
                            for value in (
                                qc.first_peak_disagreement_ms,
                                qc.zero_cross_disagreement_ms,
                                qc.max_peak_disagreement_ms,
                            )
                        )
                    ),
                    p("Yes" if qc.polarity_reversed else "No"),
                    p("None" if not qc.warnings else "; ".join(qc.warnings)),
                ]
            )
        qc_table = LongTable(
            qc_rows,
            colWidths=[
                12 * mm,
                25 * mm,
                12 * mm,
                18 * mm,
                20 * mm,
                13 * mm,
                13 * mm,
                24 * mm,
                15 * mm,
                28 * mm,
            ],
            repeatRows=1,
        )
        qc_table.setStyle(table_style(font_size=5.5))
        story.append(qc_table)

        commented = [record for record in self.waveform_records if record.review_comment]
        if commented:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph("Analyst comments", styles["Heading3"]))
            comment_rows = [["Depth (m)", "Review state", "Comment"]]
            comment_rows.extend(
                [
                    p(f"{record.depth_m:.2f}"),
                    p(REVIEW_LABELS.get(record.review_state, record.review_state)),
                    p(record.review_comment),
                ]
                for record in commented
            )
            comment_table = LongTable(
                comment_rows,
                colWidths=[20 * mm, 35 * mm, 125 * mm],
                repeatRows=1,
            )
            comment_table.setStyle(table_style(font_size=7.0))
            story.append(comment_table)
    elif self.observation_review:
        story.append(PageBreak())
        story.append(Paragraph("Saved waveform review and exclusions", styles["SectionHeading"]))
        story.append(
            Paragraph(
                "The referenced raw GRU file was unavailable when this report was generated. Saved analyst "
                "review state, uncertainty, comments, and exclusions are retained below; signal QC metrics "
                "could not be recalculated.",
                styles["ReportNote"],
            )
        )
        saved_rows: list[list[Any]] = [["Depth (m)", "Review state", "Uncertainty (ms)", "Comment"]]
        for depth, metadata in sorted(self.observation_review.items()):
            state = str(metadata.get("review_state", "not_reviewed"))
            uncertainty = metadata.get("pick_uncertainty_ms")
            saved_rows.append(
                [
                    p(f"{depth:.2f}"),
                    p(REVIEW_LABELS.get(state, state)),
                    p("-" if uncertainty is None else f"{float(uncertainty):.3f}"),
                    p(str(metadata.get("review_comment", "")) or "-"),
                ]
            )
        saved_table = LongTable(
            saved_rows,
            colWidths=[25 * mm, 45 * mm, 30 * mm, 80 * mm],
            repeatRows=1,
        )
        saved_table.setStyle(table_style(font_size=7.0))
        story.append(saved_table)

    story.append(PageBreak())
    story.append(Paragraph("Layer velocity results", styles["SectionHeading"]))
    story.append(Spacer(1, 2 * mm))
    layer_rows: list[list[Any]] = [
        [
            p("Layer", "TableHeader"),
            p("Top (m)", "TableHeader"),
            p("Bottom (m)", "TableHeader"),
            *[p(f"{PICK_LABELS[kind]} Vs", "TableHeader") for kind in PICK_KINDS],
            p("Selected residual / diagnostics", "TableHeader"),
        ]
    ]
    tops = np.r_[0.0, selected_result.depths_m[:-1]]
    for index in range(selected_result.depths_m.size):
        velocity_cells = []
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            velocity_cells.append("-" if comparison is None else f"{comparison.velocities_mps[index]:.1f}")
        diagnostic_parts = []
        if selected_result.resolution_diagonal.size == selected_result.depths_m.size:
            diagnostic_parts.append(f"R={selected_result.resolution_diagonal[index]:.2f}")
        if selected_result.standardized_residuals.size == selected_result.depths_m.size:
            diagnostic_parts.append(f"z={selected_result.standardized_residuals[index]:+.2f}")
        if selected_result.bound_active_flags.size == selected_result.depths_m.size and selected_result.bound_active_flags[index]:
            diagnostic_parts.append("BOUND")
        if selected_result.outlier_flags.size == selected_result.depths_m.size and selected_result.outlier_flags[index]:
            diagnostic_parts.append("OUTLIER")
        if selected_result.influential_flags.size == selected_result.depths_m.size and selected_result.influential_flags[index]:
            diagnostic_parts.append("INFLUENTIAL")
        layer_rows.append(
            [
                p(index + 1),
                p(f"{tops[index]:.2f}"),
                p(f"{selected_result.depths_m[index]:.2f}"),
                *[p(value) for value in velocity_cells],
                p(
                    f"{selected_result.residuals_s[index] * 1000.0:+.3f} ms; "
                    + (", ".join(diagnostic_parts) or "diagnostics unavailable")
                ),
            ]
        )
    layer_table = LongTable(
        layer_rows,
        colWidths=[10 * mm, 16 * mm, 18 * mm, *([24 * mm] * len(PICK_KINDS)), 40 * mm],
        repeatRows=1,
    )
    layer_table.setStyle(table_style(font_size=6.9))
    story.append(layer_table)

    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
