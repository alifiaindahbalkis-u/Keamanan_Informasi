from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generate_security_report(result_df):
    """
    Membuat laporan keamanan AI-NIDS dalam format PDF.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey
    )

    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    elements = []

    total = len(result_df)

    attack_count = int(
        (result_df["Kode Prediksi"] == 1).sum()
    )

    normal_count = int(
        (result_df["Kode Prediksi"] == 0).sum()
    )

    risk_counts = (
        result_df["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    low_count = int(risk_counts["Low"])
    medium_count = int(risk_counts["Medium"])
    high_count = int(risk_counts["High"])

    attack_percentage = (
        attack_count / total * 100 if total else 0
    )

    elements.append(
        Paragraph("AI-BASED NETWORK INTRUSION DETECTION SYSTEM", title_style)
    )

    elements.append(
        Paragraph(
            "Automated Security Report",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("1. Informasi Analisis", heading_style)
    )

    info_data = [
        ["Informasi", "Keterangan"],
        ["Tanggal Laporan", datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        ["Total Data Dianalisis", f"{total:,}"],
        ["Metode", "Random Forest"],
        ["Threshold", "0.50"]
    ]

    info_table = Table(
        info_data,
        colWidths=[6 * cm, 10 * cm]
    )

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B5A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(info_table)
    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph("2. Ringkasan Hasil Deteksi", heading_style)
    )

    summary_data = [
        ["Kategori", "Jumlah", "Persentase"],
        ["Normal", f"{normal_count:,}",
         f"{normal_count / total * 100:.2f}%" if total else "0%"],
        ["Attack", f"{attack_count:,}", f"{attack_percentage:.2f}%"],
        ["Total", f"{total:,}", "100%" if total else "0%"]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[5 * cm, 5 * cm, 6 * cm]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B5A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph("3. Distribusi Risk Level", heading_style)
    )

    risk_data = [
        ["Risk Level", "Jumlah", "Persentase"],
        ["Low", f"{low_count:,}",
         f"{low_count / total * 100:.2f}%" if total else "0%"],
        ["Medium", f"{medium_count:,}",
         f"{medium_count / total * 100:.2f}%" if total else "0%"],
        ["High", f"{high_count:,}",
         f"{high_count / total * 100:.2f}%" if total else "0%"]
    ]

    risk_table = Table(
        risk_data,
        colWidths=[5 * cm, 5 * cm, 6 * cm]
    )

    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B5A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))

    elements.append(risk_table)
    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph("4. Analisis Jenis Serangan", heading_style)
    )

    if "Jenis Serangan Aktual" in result_df.columns:

        attack_data = result_df[
            result_df["Jenis Serangan Aktual"] != "normal"
        ]

        if not attack_data.empty:

            attack_types = (
                attack_data["Jenis Serangan Aktual"]
                .value_counts()
                .head(10)
            )

            type_data = [["Jenis Serangan", "Jumlah Data"]]

            for attack_type, count in attack_types.items():
                type_data.append([
                    escape(str(attack_type)),
                    f"{count:,}"
                ])

            type_table = Table(
                type_data,
                colWidths=[10 * cm, 6 * cm]
            )

            type_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#263B5A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7)
            ]))

            elements.append(type_table)

        else:
            elements.append(
                Paragraph(
                    "Tidak ditemukan jenis serangan aktual pada dataset.",
                    normal_style
                )
            )

    else:
        elements.append(
            Paragraph(
                "Dataset tidak memiliki label aktual. "
                "Analisis jenis serangan tidak dapat ditampilkan.",
                normal_style
            )
        )

    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph("5. Rekomendasi Keamanan", heading_style)
    )

    recommendations = []

    if attack_count > 0:
        recommendations.append(
            "Lakukan pemeriksaan lebih lanjut terhadap data "
            "yang diklasifikasikan sebagai Attack."
        )
    else:
        recommendations.append(
            "Tidak terdapat data yang diklasifikasikan sebagai Attack. "
            "Pemantauan jaringan tetap perlu dilakukan."
        )

    if high_count > 0:
        recommendations.append(
            "Prioritaskan pemeriksaan trafik dengan Risk Level High."
        )

    if medium_count > 0:
        recommendations.append(
            "Lakukan pemantauan tambahan terhadap trafik dengan "
            "Risk Level Medium."
        )

    recommendations.append(
        "Validasi hasil deteksi menggunakan log jaringan dan "
        "informasi keamanan tambahan sebelum mengambil tindakan."
    )

    for index, recommendation in enumerate(recommendations, 1):
        elements.append(
            Paragraph(
                f"{index}. {escape(recommendation)}",
                normal_style
            )
        )
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Catatan: Laporan ini dihasilkan secara otomatis "
            "berdasarkan hasil klasifikasi model AI-NIDS. "
            "Hasil prediksi merupakan indikasi awal dan perlu "
            "divalidasi lebih lanjut.",
            styles["Italic"]
        )
    )

    doc.build(elements)

    return buffer.getvalue()
