"""
Script 06: Export Research Report to DOCX and PDF
Converts experiments/REPORT.md to:
  - experiments/output/REPORT.docx
  - experiments/output/REPORT.pdf
Uses pandoc for DOCX generation and headless LibreOffice for PDF conversion.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = EXPERIMENTS_DIR / "REPORT.md"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
OUTPUT_DOCX = OUTPUT_DIR / "REPORT.docx"
OUTPUT_PDF = OUTPUT_DIR / "REPORT.pdf"


def check_prerequisites():
    """Verify that pandoc is installed."""
    pandoc_path = shutil.which("pandoc")
    if not pandoc_path:
        print("[ERROR] pandoc not found in PATH. Please install pandoc.")
        return False
    return True


def export_to_docx():
    """Convert REPORT.md to REPORT.docx using pandoc."""
    print(f"[INFO] Converting {REPORT_MD.name} -> {OUTPUT_DOCX.name} via pandoc...")
    cmd = [
        "pandoc",
        str(REPORT_MD),
        "-o",
        str(OUTPUT_DOCX),
        "--resource-path",
        str(EXPERIMENTS_DIR),
        "--toc",
        "--toc-depth=3",
    ]
    result = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] pandoc conversion failed:\n{result.stderr}")
        return False
    print(f"[SUCCESS] Generated DOCX: {OUTPUT_DOCX} ({OUTPUT_DOCX.stat().st_size / 1024:.1f} KB)")
    return True


def export_to_pdf():
    """Convert REPORT.docx to REPORT.pdf using LibreOffice headless."""
    libreoffice_path = shutil.which("libreoffice")
    if not libreoffice_path:
        print("[WARN] libreoffice not found in PATH. Skipping PDF generation.")
        return False

    print(f"[INFO] Converting {OUTPUT_DOCX.name} -> {OUTPUT_PDF.name} via LibreOffice headless...")
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        str(OUTPUT_DOCX),
        "--outdir",
        str(OUTPUT_DIR),
    ]
    result = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] LibreOffice PDF conversion failed:\n{result.stderr}")
        return False
    if OUTPUT_PDF.exists():
        print(f"[SUCCESS] Generated PDF: {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print("[ERROR] PDF file was not created by LibreOffice.")
        return False


def main():
    if not REPORT_MD.exists():
        print(f"[ERROR] Report file not found at {REPORT_MD}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not check_prerequisites():
        sys.exit(1)

    docx_ok = export_to_docx()
    if not docx_ok:
        sys.exit(1)

    export_to_pdf()
    print("[INFO] Document export pipeline complete.")


if __name__ == "__main__":
    main()
