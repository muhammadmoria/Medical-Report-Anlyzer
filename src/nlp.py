import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import llm, GROQ_API_KEY
import json
from typing import List, Dict

load_dotenv()

os.makedirs(os.path.join("logs"), exist_ok=True)  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log")),
        logging.StreamHandler()
    ]
)

def structure_data(text: str) -> List[Dict]:
    """
    Use Groq LLM to extract all explicitly mentioned test result information from medical report text.
    Returns a list of dictionaries with all fields found in the report.
    """
    logging.info("Extracting structured data using LLM.")
    try:
        messages = [
            SystemMessage(content="You are a medical data extraction assistant."),
            HumanMessage(content=f"""
                You are a medical data extraction expert.

                Given the following medical report, extract all explicitly mentioned information related to test results and return it as a **valid JSON array** of dictionaries. Each dictionary should represent a test result or relevant metadata (e.g., patient information) as found in the text.

                **Important Instructions:**
                - Include **only** fields that are explicitly mentioned in the report (e.g., test name, value, unit, normal range, patient name, age, date, etc.).
                - Do **not** guess, hallucinate, or add fields not present in the text.
                - Do **not** perform any calculations or inferences (e.g., do not compute status or categorize values).
                - Each dictionary should contain key-value pairs for the fields explicitly stated in the report.
                - Your response must be a **JSON array of dictionaries**.
                - Return **only** the JSON array — no explanations, no markdown, no code formatting, no comments.

                Medical Report:
                {text}
                """)
        ]

        response = llm.invoke(messages)
        logging.info("✅ Response received from Groq.")
        
        try:
            results = json.loads(response.content.strip())
            if not isinstance(results, list):
                logging.warning("LLM returned non-list response")
                return []
            logging.info(f"Extracted {len(results)} results from LLM response")
            return results
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return []
    except Exception as e:
        logging.exception(f"LLM structuring failed: {str(e)}")
        return []