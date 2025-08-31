import csv
from fpdf import FPDF
from PyQt5.QtWidgets import QFileDialog

def export_to_csv(logs_list):
    file_path, _ = QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV Files (*.csv)")
    if not file_path:
        return

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for camera_name, logs in logs_list:
            writer.writerow([camera_name])
            writer.writerow(["Time", "Valid", "Confidence", "Scored By", "Body Part"])
            writer.writerows(logs)
            writer.writerow([])
    return file_path

def export_to_pdf(logs_list):
    """
    logs_list: List of tuples (camera_name, logs)
    """
    file_path, _ = QFileDialog.getSaveFileName(None, "Save PDF", "", "PDF Files (*.pdf)")
    if not file_path:
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=14)
    pdf.cell(200, 10, "Arnis Strike Detection Logs", ln=True, align="C")
    pdf.set_font("Arial", size=12)

    for camera_name, logs in logs_list:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, camera_name, ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(40, 10, "Time", 1)
        pdf.cell(30, 10, "Valid", 1)
        pdf.cell(40, 10, "Confidence", 1)
        pdf.cell(40, 10, "Scored By", 1)
        pdf.cell(40, 10, "Body Part", 1)
        pdf.ln()

        for log in logs:
            timestamp, valid, confidence, scored_by, body_part = log
            pdf.cell(40, 10, str(timestamp), 1)
            pdf.cell(30, 10, str(valid), 1)
            pdf.cell(40, 10, f"{confidence:.2f}", 1)
            pdf.cell(40, 10, scored_by, 1)
            pdf.cell(40, 10, body_part, 1)
            pdf.ln()
    pdf.output(file_path)
    return file_path

def export_cnn_lstm_logs_csv(title, logs):
    file_path, _ = QFileDialog.getSaveFileName(None, f"Save {title} as CSV", "", "CSV Files (*.csv)")
    if not file_path:
        return
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([title])
        writer.writerow(["Time", "Confidence", "Body Part", "Camera"])
        writer.writerows(logs)

def export_cnn_lstm_logs_pdf(title, logs):
    file_path, _ = QFileDialog.getSaveFileName(None, f"Save {title} as PDF", "", "PDF Files (*.pdf)")
    if not file_path:
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=14)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(45, 10, "Time", 1)
    pdf.cell(45, 10, "Confidence", 1)
    pdf.cell(45, 10, "Body Part", 1)
    pdf.cell(45, 10, "Camera", 1)
    pdf.ln()
    for log in logs:
        time, confidence, body_part, camera = log
        pdf.cell(45, 10, str(time), 1)
        pdf.cell(45, 10, str(confidence), 1)
        pdf.cell(45, 10, str(body_part), 1)
        pdf.cell(45, 10, str(camera), 1)
        pdf.ln()
    pdf.output(file_path)
