import json
import logging
import requests
import boto3
from langdetect import detect, DetectorFactory
from botocore.exceptions import ClientError

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hardcoded API Endpoint
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
        logger.error(f"Error fetching user data from API ({API_URL}): {e}")
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
    Cleans unwanted characters or system tokens from the response.
    """
    res = response_text.split("\n", 1)
    if len(res) > 1:
        res = res[1].strip()
    else:
        res = res[0].strip()  # Fallback if '\n' is not found
    res = res.replace("Response:", "").strip()
    res = res.replace("**Response**:", "").strip()
    return res


def generate_prompt(language, question, user_data, options, rewrite=None):
    """
    Generate the input prompt for the AI model in the appropriate language.
    If `rewrite` is provided, it is used as context for rewriting.
    """
    context = f"Rewrite context: {rewrite}" if rewrite else ""
    
    if language == "en":
        return f"""
        You are an expert at generating precise, professional, and compelling answers to grant application questions based on the provided **Business Information** and **Options**.

        **Instructions**:
        1. If options are provided:
           - Output only the exact text of the most appropriate option.
           - Do not include any additional words, sentences, or symbols.
        2. If options are not provided:
           - Provide a precise and accurate response without extra commentary.
           - Craft a detailed, compelling, and professional single-paragraph response using all relevant information.

        **{context}**

        **Question**: {question}
        **Options**: {options or "Not provided"}
        **Business Information**: {json.dumps(user_data, indent=4)}
        **Response**:
        """
    elif language == "fr":
        return f"""
        Vous êtes un expert dans la génération de réponses précises, professionnelles et convaincantes aux questions de demande de subvention basées sur les **Informations sur l'entreprise** et les **Options**.

        **Instructions**:
        1. Si des options sont fournies :
           - Fournissez uniquement le texte exact de l'option la plus appropriée.
           - N'incluez aucun mot, phrase ou symbole supplémentaire.
        2. Si des options ne sont pas fournies :
           - Fournissez une réponse précise et exacte sans commentaire supplémentaire.
           - Rédigez une réponse détaillée, convaincante et professionnelle en un seul paragraphe en utilisant toutes les informations pertinentes.

        **{context}**

        **Question** : {question}
        **Options** : {options or "Non fourni"}
        **Informations sur l'entreprise** : {json.dumps(user_data, indent=4)}
        **Réponse** :
        """


def integrate_content_with_grant_writing(question, user_data, options, rewrite=None):
    """
    Generate a response to a question using user data.
    Handles rewrites by including additional context.
    """
    language = detect_language(question)
    prompt = generate_prompt(language, question, user_data, options, rewrite)

    logger.info(f"Generated Prompt: {prompt}")

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
        return clean_response(generated_text)
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error(f"A client error occurred: {message}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    """
    if "body" in event and event["body"]:
        event = json.loads(event["body"])

    question = event.get("question")
    options = event.get("options", None)
    rewrite = event.get("rewrite", None)

    user_data = fetch_user_data()

    if not user_data:
        return {
            "statusCode": 500,
            "body": "Failed to fetch user data"
        }

    try:
        response = integrate_content_with_grant_writing(question, user_data, options, rewrite)
        return {
            "statusCode": 200,
            "body": json.dumps({"response": response})
        }
    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "body": "An error occurred"
        }


