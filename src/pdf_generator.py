from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from io import BytesIO
from typing import List, Dict
import logging, os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)

def generate_pdf_summary(
    results: List[Dict],
    explanations: str,
    summary_bullets: str,
    output_path: str = None
) -> bytes:
    logging.info("Generating improved PDF summary")
    buffer = BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='List', leftIndent=20, fontSize=10, spaceAfter=6))
        story = []

        story.append(Paragraph("🩺 Medical Report Summary", styles["Title"]))
        story.append(Spacer(1, 12))

        metadata = [r for r in results if "test_name" not in r]
        if metadata:
            story.append(Paragraph("👤 Patient Information", styles["Heading2"]))
            for item in metadata:
                for key, value in item.items():
                    story.append(Paragraph(f"<b>{key.capitalize()}</b>: {value}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
        test_results = [r for r in results if "test_name" in r]
        if test_results:
            story.append(Paragraph("🧪 Test Results", styles["Heading2"]))
            data = [["Test Name", "Value", "Unit", "Normal Range", "Status"]]
            for res in test_results:
                data.append([
                    res.get("test_name", "Unknown"),
                    res.get("value", "Unknown"),
                    res.get("unit", "Unknown"),
                    res.get("normal_range", ""),
                    res.get("status", "Unknown")
                ])
            table = Table(data, hAlign='LEFT', colWidths=[130, 70, 70, 130, 80])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))

        if explanations:
            story.append(Paragraph("🧾 Explanation", styles["Heading2"]))
            explanation_clean = explanations.replace("\n", " ").strip()
            story.append(Paragraph(explanation_clean, styles["Normal"]))
            story.append(Spacer(1, 12))

        if summary_bullets:
            story.append(Paragraph("📌 Summary and Recommendations", styles["Heading2"]))

            try:
                import re
                summary_match = re.search(r"\*\*Summary:\*\*(.*?)\*\*", summary_bullets, re.DOTALL)
                risks_match = re.search(r"\*\*Risks/Conditions:\*\*(.*?)\*\*", summary_bullets, re.DOTALL)
                actions_match = re.search(r"\*\*Actions/Recommendations:\*\*(.*)", summary_bullets, re.DOTALL)

                if summary_match:
                    story.append(Paragraph("📋 <b>Summary:</b>", styles["Normal"]))
                    summary_lines = summary_match.group(1).strip().split("*")
                    summary_items = [ListItem(Paragraph(line.strip(), styles["List"])) for line in summary_lines if line.strip()]
                    story.append(ListFlowable(summary_items, bulletType='bullet'))

                if risks_match:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph("⚠️ <b>Risks/Conditions:</b>", styles["Normal"]))
                    risk_lines = risks_match.group(1).strip().split("*")
                    risk_items = [ListItem(Paragraph(line.strip(), styles["List"])) for line in risk_lines if line.strip()]
                    story.append(ListFlowable(risk_items, bulletType='bullet'))

                if actions_match:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph("✅ <b>Actions/Recommendations:</b>", styles["Normal"]))
                    action_lines = actions_match.group(1).strip().split("*")
                    action_items = [ListItem(Paragraph(line.strip(), styles["List"])) for line in action_lines if line.strip()]
                    story.append(ListFlowable(action_items, bulletType='bullet'))

            except Exception as e:
                logging.warning("Failed to parse summary bullet points. Falling back to plain text.")
                story.append(Paragraph(summary_bullets.replace("\n", " "), styles["Normal"]))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        logging.info("Improved PDF summary generated successfully")
        return pdf_bytes

    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        raise
