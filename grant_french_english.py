import json
import logging
import requests
from langdetect import detect, DetectorFactory
import boto3
from botocore.exceptions import ClientError

# Set up consistent language detection
DetectorFactory.seed = 0

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hardcoded API endpoint
API_URL = "https://api.happly.ai/api/v1/portal/users/ba845892-3275-47a3-9327-fcf7cba266a6"

def fetch_user_data():
    """
    Fetch user data from the hardcoded API endpoint.
    """
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        logger.error(f"Error fetching user data from API: {e}")
    return {}

def detect_language(text):
    """
    Detect the language of a given text.
    """
    try:
        language = detect(text)
        if language not in ["en", "fr"]:
            raise ValueError(f"Unsupported language detected: {language}")
        return language
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

def generate_text(model_id, body):
    """
    Generate text using a Bedrock AI model.
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return response_body

def clean_response(response_text):
    """
    Removes any unwanted special tokens like <<SYS>> from the response.
    """
    return response_text.replace("[/SYS]", "").replace("<<SYS>>", "").strip()

def generate_prompt(language, question, user_data):
    """
    Generate the input prompt for the AI model in the appropriate language.
    """
    if language == "en":
        return f"""
        You are an expert at generating precise, professional answers to grant application questions.

        **Question**:
        {question}

        **User Data**:
        {json.dumps(user_data, indent=4)}

        **Response**:
        """
    elif language == "fr":
        return f"""
        Vous êtes un expert en réponses précises et professionnelles pour les demandes de subventions.

        **Question** :
        {question}

        **Données utilisateur** :
        {json.dumps(user_data, indent=4)}

        **Réponse** :
        """

def integrate_content_with_grant_writing(question, user_data):
    """
    Generate a response to the question based on user data.
    """
    language = detect_language(question)
    prompt = generate_prompt(language, question, user_data)

    # Prepare the request body for the AI model
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 300,
        "temperature": 0.7,
    })

    # Generate the response
    try:
        raw_response = generate_text(model_id, body)
        generated_text = raw_response.get("generation", "")
        cleaned_text = clean_response(generated_text)
        return cleaned_text
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"A client error occurred: {message}")
        return "An error occurred while generating the response."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred."

def lambda_handler(event, context):
    # Parse the input
    if "body" in event and event["body"]:
        event = json.loads(event["body"])

    question = event.get("question")

    # Fetch user data
    user_data = fetch_user_data()

    if not user_data:
        return {
            "statusCode": 500,
            "body": "Failed to fetch user data"
        }

    # Generate the response
    try:
        response = integrate_content_with_grant_writing(question, user_data)
        return {
            "statusCode": 200,
            "body": response  # Return the response as plain text
        }
    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "body": "An error occurred"
        }


