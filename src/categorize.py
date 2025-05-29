import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
import json
from src.config import llm, GROQ_API_KEY
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
    
def categorize_results(results: List[Dict]) -> List[Dict]:
    """
    Use Groq LLM to categorize medical report data based on provided fields.
    Returns the list of dictionaries with a 'status' field added where applicable.
    """
    logging.info("Categorizing medical report data using LLM")
    try:
        results_text = json.dumps(results, indent=2)
        
        prompt = f"""
        You are an expert medical data categorizer with deep knowledge of medical reports.

        Given the following list of dictionaries containing medical report data (e.g., test results, patient metadata, or other fields), analyze each entry and assign a 'status' field with one of the values: 'Critical', 'Borderline', 'Normal', or 'Unknown'. Categorize based solely on the provided data, using your medical expertise to interpret the values and context. The data can contain any fields (e.g., test names, values, ranges, units, patient info, or others), and you should not assume specific fields are present.

        **Important Instructions:**
        - For entries likely representing test results (e.g., containing fields like test_name, value, or similar), assign a status based on the provided data:
          - Use 'Normal' if the data indicates a value within typical medical norms (e.g., based on a range or medical context).
          - Use 'Borderline' if the data suggests a value slightly outside typical norms.
          - Use 'Critical' if the data indicates a value significantly outside typical norms.
          - Use 'Unknown' if insufficient data is provided to determine status (e.g., missing values or context).
        - For non-test entries (e.g., patient_name, age, date), do not add a 'status' field unless the data directly informs a medical categorization (e.g., age indicating risk).
        - Do **not** perform numerical calculations or assume specific fields (e.g., value, normal_range) are present.
        - Do **not** guess or hallucinate information not provided in the input.
        - Return the original list of dictionaries, updated with a 'status' field where applicable, as a JSON array.
        - Return **only** the JSON array — no explanations, no markdown, no code formatting, no comments.

        Input Data:
        {results_text}
        """
        
        messages = [
            SystemMessage(content="You are an expert medical data categorizer."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        logging.info("✅ Response received from Groq for categorization")
        
        try:
            categorized_results = json.loads(response.content.strip())
            if not isinstance(categorized_results, list):
                logging.warning("LLM returned non-list response for categorization")
                return results
            logging.info(f"Categorized {len(categorized_results)} results")
            return categorized_results
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM categorization response as JSON: {str(e)}")
            return results  
    except Exception as e:
        logging.exception(f"LLM categorization failed: {str(e)}")
        return results  

    