import json
import logging
import requests
import boto3
import base64
from langdetect import detect, DetectorFactory
from botocore.exceptions import ClientError

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hardcoded API Endpoint
API_URL = "https://api.happly.ai/api/v1/portal/users/"

def fetch_user_data(client_id=None):
    """
    Fetch user data from the API endpoint. Use default profile if client_id is not provided.
    """
    url = API_URL + (client_id if client_id else "default-profile-id")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        logger.error(f"Error fetching user data from API ({url}): {e}")
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

def decode_document_content(encoded_content):
    """
    Decodes Base64-encoded document content.
    """
    try:
        decoded_content = base64.b64decode(encoded_content)
        return decoded_content.decode('utf-8', errors='replace')  # Replace invalid characters
    except Exception as e:
        logger.error(f"Error decoding document content: {e}")
        return None

def generate_prompt(language, question, user_data, document_text=None, options=None, rewrite=None):
    """
    Generate the input prompt for the AI model in the appropriate language.
    If `rewrite` is provided, it is used as context for rewriting.
    """
    context = f"Rewrite context: {rewrite}" if rewrite else ""

    if language == "en":
        return f"""
        You are an expert at generating precise, professional, and compelling answers to grant application questions based on the provided **Business Information**, **Options**, and **Document Text**.

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
        **Document Text**: {document_text or "Not provided"}
        **Response**:
        """
    elif language == "fr":
        return f"""
        Vous êtes un expert dans la génération de réponses précises, professionnelles et convaincantes aux questions de demande de subvention basées sur les **Informations sur l'entreprise**, les **Options** et le **Texte du document**.

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
        **Texte du document** : {document_text or "Non fourni"}
        **Réponse** :
        """

def integrate_content_with_grant_writing(question, user_data, document_text=None, options=None, rewrite=None):
    """
    Generate a response to a question using user data.
    Handles rewrites by including additional context and optionally includes document text.
    """
    language = detect_language(question)
    prompt = generate_prompt(language, question, user_data, document_text, options, rewrite)

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

    client_id = event.get("client_id", None)
    question = event.get("question")
    document_content = event.get("document_content", None)
    options = event.get("options", None)
    rewrite = event.get("rewrite", None)

    # Fetch user data based on client ID or default profile
    user_data = fetch_user_data(client_id)

    if not user_data:
        return {
            "statusCode": 500,
            "body": "Failed to fetch user data"
        }

    # Decode the document content if provided
    document_text = None
    if document_content:
        document_text = decode_document_content(document_content)
        if document_text is None:
            return {
                "statusCode": 400,
                "body": "Invalid document content"
            }

    try:
        # Pass the document_text as context
        response = integrate_content_with_grant_writing(
            question, user_data, document_text, options, rewrite
        )
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "body": "An error occurred"
        }



