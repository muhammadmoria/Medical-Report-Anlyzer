import logging
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict
from src.config import llm, GROQ_API_KEY
import os
from dotenv import load_dotenv

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

def generate_summary_bullet_points(explanations: str) -> str:
    """
    Given a full explanation block (from explain.py), generate:
    - Summary of key findings
    - Possible risks/conditions with likelihood
    - Specific recommendations and next steps

    All returned in plain text bullet points.
    """
    logging.info("Generating summary bullet points from explanations")

    try:
        prompt = f"""
        You are a compassionate and professional medical assistant.

        You will receive a set of detailed medical explanations (already written in patient-friendly language).
        Your task is to generate the following — using **bullet points** only:

        - 🔍 **Summary**: 3–5 concise points highlighting what was found in the medical report.
        - ⚠️ **Risks/Conditions**: List potential health risks or conditions with likelihood (High, Possible, Low), based on the explanations.
        - ✅ **Actions/Recommendations**: Provide 2–5 very specific next steps, lifestyle tips, or suggestions (e.g., "Consult a cardiologist", "Reduce sugar intake", "Schedule follow-up in 1 month").

        Do NOT repeat the full explanations.
        Do NOT return any JSON or formatting instructions — just clean, readable bullet points grouped into the 3 sections above.

        Medical Explanations:
        {explanations}
        """

        messages = [
            SystemMessage(content="You are a compassionate and professional medical assistant."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        logging.error(f"❌ Error generating bullet summary: {str(e)}")
        return "Unable to generate summary due to an error."

