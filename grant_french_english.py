import json
import logging
import requests
import boto3
from langdetect import detect, DetectorFactory
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
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
    Removes any unwanted special tokens like <<SYS>> from the response.
    """
    return response_text.replace("[/SYS]", "").replace("<<SYS>>", "").strip()


def extract_text_from_pdf(file_path):
    """
    Extracts and cleans text from a PDF file.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return clean_extracted_text(text)
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(file_path):
    """
    Extracts and cleans text from a .docx Word document.
    """
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return clean_extracted_text("\n".join(text))
    except Exception as e:
        logger.error(f"Error reading Word document: {e}")
        return ""


def extract_text_from_image(file_path):
    """
    Extracts and cleans text from an image file using OCR.
    """
    try:
        image = Image.open(file_path)
        raw_text = pytesseract.image_to_string(image)
        return clean_extracted_text(raw_text)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return ""


def clean_extracted_text(text):
    """
    Cleans extracted text to remove artifacts like extra spaces, line breaks, and formatting.
    """
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def process_uploaded_document(file_path):
    """
    Processes an uploaded document and extracts cleaned text based on its type.
    """
    file_type = file_path.split(".")[-1].lower()

    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    elif file_type in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path)
    else:
        return f"Unsupported file type: {file_type}"


def generate_prompt(language, question, user_data, document_text):
    """
    Generate the input prompt for the AI model in the appropriate language.
    """
    if language == "en":
        return f"""
        You are an expert grant-writing AI. Based on the provided business information and document, generate a professional and compelling response to the question below:

        Question: {question}

        Business Information:
        {json.dumps(user_data, indent=4)}

        Supplemental Document:
        {document_text}
        """
    elif language == "fr":
        return f"""
        Vous êtes une IA experte en rédaction de subventions. En vous basant sur les informations commerciales et le document fournis, générez une réponse professionnelle et convaincante à la question ci-dessous :

        Question : {question}

        Informations commerciales :
        {json.dumps(user_data, indent=4)}

        Document supplémentaire :
        {document_text}
        """


def integrate_document_content_with_grant_writing(question, user_data, file_paths):
    """
    Integrates document content into the grant-writing process.
    """
    document_texts = []
    for file_path in file_paths:
        extracted_text = process_uploaded_document(file_path)
        document_texts.append(extracted_text)

    combined_document_text = "\n\n".join(document_texts)
    language = detect_language(question)
    prompt = generate_prompt(language, question, user_data, combined_document_text)

    # Prepare request body for the AI model
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 300,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    try:
        response = generate_text(model_id, body)
        generated_text = response.get('generation', '')
        cleaned_text = clean_response(generated_text)
        return cleaned_text
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None


if __name__ == "__main__":
    # API Endpoint
    user_api_url = "https://api.happly.ai/api/v1/portal/users/ba845892-3275-47a3-9327-fcf7cba266a6"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY"  # Replace with the actual API key
    }

    # Fetch User Data
    user_data = fetch_user_data(user_api_url, headers=headers)
    if not user_data:
        print("Failed to fetch user data. Exiting.")
        exit()

    # Example Question
    question = "Describe your propduct"  # English example

    # Path to the uploaded PDF document
    uploaded_files = ["/Users/chasesimard/Documents/Happly/Govago Business Plan.pdf"]

    # Generate Response
    response = integrate_document_content_with_grant_writing(question, user_data, uploaded_files)
    print("Generated Grant Response:\n", response)
