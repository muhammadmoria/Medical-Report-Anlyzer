import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import Dict, List
from src.config import llm, GROQ_API_KEY
import os, json
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

def explain_results_batch(results: List[Dict]) -> str:
    """
    Send all categorized results to the LLM and request detailed explanations
    for each one in a single response.
    """
    logging.info("Generating batch explanations for all categorized test results.")

    try:
        input_data = json.dumps(results, indent=2)

        prompt = f"""
        You are a professional medical explanation assistant.

        You will receive a list of medical test results in dictionary format. Each dictionary may include:
        - test_name
        - value
        - unit
        - normal_range
        - status
        - additional metadata

        Your job is to clearly and patiently explain **each test result** to a non-technical patient. For **each test**, give a separate explanation that includes:
        - What the test measures.
        - The patient's value and what it means.
        - The given status (Normal, Borderline, Critical, Unknown) and why it was assigned.
        - If needed, what the patient should do next.

        Use simple language.
        Only use provided data. Do not assume, infer, or invent missing details.

        Return a clearly separated explanation **for each test** — label them clearly with the test name.

        Input:
        {input_data}
        """

        messages = [
            SystemMessage(content="You are a professional medical explanation assistant."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        explanation = response.content.strip()
        logging.info("✅ Batch explanations received.")
        return explanation

    except Exception as e:
        logging.error(f"❌ Error generating batch explanation: {str(e)}")
        return "Unable to generate explanations due to an error."