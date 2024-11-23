import json
import logging
import requests
import boto3
from langdetect import detect, DetectorFactory
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
    Removes any unwanted special tokens or formatting from the response.
    """
    res = response_text.split("\n", 1)
    if len(res) > 1:
        res = res[1].strip()
    else:
        res = res[0].strip()
    res = res.replace("Response:", "").strip()
    res = res.replace("**Response**:", "").strip()
    return res


def generate_prompt(language, question, user_data, options):
    """
    Generate the input prompt for the AI model in the appropriate language.
    """
    if language == "en":
        return f"""
        You are an expert at generating precise, professional, and compelling answers to grant application questions.

        **Instructions**:
        - Use the provided **User Data** and **Options** (if any).
        - Respond in a professional and concise format.

        **Question**:
        {question}

        **Options**:
        {options or "Not provided"}

        **User Data**:
        {json.dumps(user_data, indent=4)}

        **Response**:
        """

    elif language == "fr":
        return f"""
        Vous êtes un expert en rédaction de réponses précises, professionnelles et convaincantes pour les demandes de subventions.

        **Instructions** :
        - Utilisez les **Données Utilisateur** et les **Options** (le cas échéant).
        - Répondez de manière professionnelle et concise.

        **Question** :
        {question}

        **Options** :
        {options or "Non fourni"}

        **Données Utilisateur** :
        {json.dumps(user_data, indent=4)}

        **Réponse** :
        """


def rewrite_section(previous_response, feedback, question, user_data, options):
    """
    Regenerate a response based on user feedback.
    """
    updated_prompt = f"""
    The user provided the following feedback to improve the previous response:
    "{feedback}"

    **Original Question**:
    {question}

    **Original Response**:
    {previous_response}

    Use this feedback to refine the response while adhering to the original requirements.

    **User Data**:
    {json.dumps(user_data, indent=4)}

    **Options**:
    {options or "Not provided"}

    **Improved Response**:
    """
    return updated_prompt


def integrate_document_content_with_grant_writing(
    action, question, user_data, options, feedback=None, previous_response=None
):
    """
    Handles the grant-writing process including regenerations.
    """
    language = detect_language(question)

    if action == "generate":
        prompt = generate_prompt(language, question, user_data, options)
    elif action == "regenerate" and feedback and previous_response:
        prompt = rewrite_section(previous_response, feedback, question, user_data, options)
    else:
        return "Invalid action or missing feedback for regeneration."

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
        logger.error("A client error occurred: %s", message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None


def lambda_handler(event, context):
    """
    Main AWS Lambda handler function.
    """
    if "body" in event and event["body"]:
        event = json.loads(event["body"])

    action = event.get("action", "generate")
    question = event.get("question")
    user_data = event.get("user_data")
    options = event.get("options", None)
    feedback = event.get("feedback", None)
    previous_response = event.get("previous_response", None)

    response = integrate_document_content_with_grant_writing(
        action, question, user_data, options, feedback, previous_response
    )

    return {"statusCode": 200, "body": json.dumps(response)}

