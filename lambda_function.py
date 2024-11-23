import json
import logging
import requests
from langdetect import detect, DetectorFactory
import boto3
from botocore.exceptions import ClientError

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fetch_user_data(api_url, headers=None):
    """
    Fetch user data from the provided API endpoint.
    """
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        logger.error(f"Error fetching user data from API ({api_url}): {e}")
    return {}

def detect_language(text):
    """
    Detects the language of a given text.
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
    Generate text using Meta Llama 3.2 Chat on demand.
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
    Cleans the response from the AI model to remove unwanted tokens or formatting.
    """
    res = response_text.split("\n", 1)
    if len(res) > 1:
        res = res[1].strip()
    else:
        res = res[0].strip()
    return res.replace("Response:", "").strip()

def generate_prompt(language, question, user_data):
    """
    Generate the input prompt for the AI model in the appropriate language.
    """
    if language == "en":
        return f"""
        You are an expert at generating precise, professional, and compelling answers to questions using the provided **User Information**.

        **Instructions**:
        - Use the user information to craft a professional and accurate response.
        - The response must be concise and directly address the question.

        **Question**:
        {question}

        **User Information**:
        {json.dumps(user_data, indent=4)}

        **Response**:
        """
    elif language == "fr":
        return f"""
        Vous êtes un expert dans la génération de réponses précises, professionnelles et convaincantes en utilisant les **Informations Utilisateur** fournies.

        **Instructions** :
        - Utilisez les informations utilisateur pour rédiger une réponse professionnelle et exacte.
        - La réponse doit être concise et répondre directement à la question.

        **Question** :
        {question}

        **Informations Utilisateur** :
        {json.dumps(user_data, indent=4)}

        **Réponse** :
        """
    else:
        raise ValueError("Unsupported language")

def integrate_document_content_with_grant_writing(question, user_data):
    """
    Generate a response to the given question using the user data.
    """
    language = detect_language(question)
    prompt = generate_prompt(language, question, user_data)
    
    logger.info(f"Generated prompt: {prompt}")
    
    # Prepare request body for the AI model
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    body = json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": 300,
            "temperature": 0.7,
        }
    )

    try:
        response = generate_text(model_id, body)
        generated_text = response.get("generation", "")
        cleaned_text = clean_response(generated_text)
        return cleaned_text
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"A client error occurred: {message}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None

def lambda_handler(event, context):
    try:
        # Check if the 'body' field exists and is not None
        if "body" in event and event["body"]:
            # Assume 'body' is JSON-encoded and parse it
            event = json.loads(event["body"])

        # Extract parameters from the event
        question = event.get("question")
        api_url = event.get("api_url")

        if not question or not api_url:
            return {"statusCode": 400, "body": "Missing 'question' or 'api_url' in request"}

        # Fetch user data from the API
        user_data = fetch_user_data(api_url)

        if not user_data:
            return {"statusCode": 500, "body": "Failed to fetch user data"}

        # Generate response based on the question and user data
        response_text = integrate_document_content_with_grant_writing(
            question=question,
            user_data=user_data
        )

        # Return the response text
        return {
            "statusCode": 200,
            "body": json.dumps({"result": response_text})
        }

    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}")
        return {"statusCode": 500, "body": str(e)}


