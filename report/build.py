"""Build the scientific report from manuscript text and publication CSVs."""
import csv
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "results" / "tables"
styles = getSampleStyleSheet()
styles["Title"].alignment = TA_LEFT
styles["BodyText"].leading = 14
styles["BodyText"].spaceAfter = 7
styles["Heading1"].textColor = colors.HexColor("#173b55")
story = []


def csv_rows(name):
    with (TABLES / name).open() as stream:
        return list(csv.DictReader(stream))


def paragraph(text, style="BodyText"):
    story.append(Paragraph(escape(text), styles[style]))


def table(headers, rows, widths=None):
    grid = Table([headers, *rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173b55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([grid, Spacer(1, 10)])


def number(value):
    return f"{float(value):,.2f}"


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#435362"))
    canvas.drawString(doc.leftMargin, 30, "LVLM-Count reproducibility study")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 30, str(doc.page))


def main():
    for line in (HERE / "manuscript.md").read_text().splitlines():
        if line.startswith("# "):
            paragraph(line[2:], "Title")
        elif line.startswith("## "):
            paragraph(line[3:], "Heading1")
        elif line:
            paragraph(line)
    story.append(PageBreak())
    paragraph("Appendix A. Quantitative results", "Heading1")
    paragraph("Table A1. Paired differences in MAE. A positive difference favors full. "
              "Predictions are averaged by record before calculating absolute error.")
    table(["Dataset", "Records", "Images", "Base MAE", "Full MAE", "Difference", "95% CI"],
          [[r["dataset"], r["records"], r["source_images"], number(r["base_mae"]),
            number(r["full_mae"]), number(r["diff"]),
            f'{number(r["ci_lo"])} to {number(r["ci_hi"])}']
           for r in csv_rows("paired_by_image.csv")])
    results = csv_rows("evaluation.csv")
    paragraph("Table A2. Ablations and coverage. MAE and RMSE use each variant’s available records. "
              "Common MAE uses all IDs shared by variants within the dataset. In Penguin this is 96.")
    for dataset in dict.fromkeys(r["dataset"] for r in results):
        paragraph(dataset.replace("_", " "), "Heading2")
        table(["Method", "Records", "Trials", "MAE", "RMSE", "Raw MAE", "Flags", "Common MAE"],
              [[r["method"], r["records"], r["trials"], number(r["mae"]), number(r["rmse"]),
                number(r["raw_mae"]), r["flagged"], number(r["common_mae"])]
               for r in results if r["dataset"] == dataset])
    paragraph("Table A3. Deterministic mask-count baseline. The validation pilot is separate from the test split.")
    table(["Dataset", "Records", "MAE", "RMSE"],
          [[r["dataset"], r["n_images"], number(r["MAE"]), number(r["RMSE"])]
           for r in csv_rows("mask_baseline.csv")])
    paragraph("Table A4. Strict-format failures for the primary methods. Full ablation counts accompany the source.")
    table(["Dataset / method", "Verified jobs", "Jobs with strict failure", "Rate", "Exhausted"],
          [[f'{r["dataset"]} / {r["method"]}', r["verified_jobs"], r["jobs_with_strict_failure"],
            f'{100 * int(r["jobs_with_strict_failure"]) / int(r["verified_jobs"]):.1f}%',
            r["jobs_exhausted"]]
           for r in csv_rows("strict_format.csv")
           if r["method"] in ("base", "full") and int(r["verified_jobs"])])
    SimpleDocTemplate(
        str(HERE / "countvision.pdf"), pagesize=(612, 792), leftMargin=60, rightMargin=60,
        topMargin=48, bottomMargin=48, title="Object-aware image division for visual counting",
        author="CountVision project",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
