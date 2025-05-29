import pytesseract
from src.preprocess import extract_text_from_pdf
import logging, os

os.makedirs(os.path.join("logs"), exist_ok=True)
logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log")),
        logging.StreamHandler() 
    ]
)

def extract_text(file_path: str) -> str:
    """Extract text from image or PDF."""
    logging.info(f"Starting text extraction for file: {file_path}")
    try:
        if file_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            logging.info("Text extracted from PDF")
        else:
            logging.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Use PDF, PNG, or JPEG.")
        return text
    except Exception as e:
        logging.error(f"Error in OCR for {file_path}: {str(e)}")
        raise