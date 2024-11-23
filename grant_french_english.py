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


# def clean_response(response_text):
#     """
#     Removes any unwanted special tokens like <<SYS>> from the response.
#     """
#     return response_text.replace("[/SYS]", "").replace("<<SYS>>", "").strip()
#     # Clean response


def clean_response(response_text):
    """
    Removes any unwanted special tokens like <<SYS>> from the response.
    """
    # res = response_text.replace("[/SYS]", "").strip()
    # return res.replace("<<SYS>>", "").strip()
    # Define the pattern with case-insensitive flag

    # print(response_text)

    res = response_text.split("\n", 1)
    if len(res) > 1:
        res = res[1].strip()
    else:
        res = res[0].strip()  # Fallback if '\n' is not found
    res = res.replace("Response:", "").strip()
    res = res.replace("**Response**:", "").strip()

    return res


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


def generate_prompt(language, question, user_data, document_text, options):
    """
    Generate the input prompt for the AI model in the appropriate language.
    """
    if language == "en":
        return f"""
        You are an expert at generating precise, professional, and compelling answers to grant application questions based on the provided **Business Information**, **Supplemental Document**, and **Options**.

        - The **Business Information** is provided as a JSON file containing the personal and business details of the applicant.
        - The **Supplemental Document** is optional and may include additional information extracted from files (e.g., PDFs, Word documents).
        - The **Options** field is optional. If it exists, select the most appropriate answer from the options provided. 

        **Guidelines for Answer Generation**:
        1. If **Options** are provided, select the most relevant option as the answer, reply with only the option, do not add any additional words or symbols.
        2. If the question seeks specific information (e.g., a date, address, or event), provide a precise and accurate response.
        3. For general questions, craft a detailed, compelling, and professional response, incorporating all relevant information.
        4. **Deliver your response in English as a polished and clear narrative.** The response should be a single, plain paragraph **without any special symbols, introductory phrases, or additional comments.**

        **Question**:
        {question}
        
        **Options**:
        {options or "Not provided"}

        **Business Information**:
        {json.dumps(user_data, indent=4)}

        **Supplemental Document**:
        {document_text or "Not provided"}
        """

    elif language == "fr":
        return f"""
        Vous êtes un expert dans la génération de réponses précises, professionnelles et convaincantes aux questions de demande de subvention basées sur les **Informations sur l'Entreprise**, le **Document Supplémentaire** et les **Options** fournies.

        - Les **Informations sur l'Entreprise** sont fournies sous forme de fichier JSON contenant les détails personnels et professionnels du demandeur.
        - Le **Document Supplémentaire** est facultatif et peut inclure des informations supplémentaires extraites de fichiers (par exemple, PDF, documents Word).
        - Le champ **Options** est facultatif. S'il existe, sélectionnez la réponse la plus appropriée parmi les options fournies. 

        **Directives pour la Génération des Réponses**:
        1. Si des **Options** sont fournies, sélectionnez l'option la plus pertinente comme réponse, répondez uniquement avec l'option, n'ajoutez aucun mot ou symbole supplémentaire.
        2. Si la question demande des informations spécifiques (par exemple, une date, une adresse ou un événement), fournissez une réponse précise et exacte.
        3. Pour les questions générales, élaborez une réponse détaillée, convaincante et professionnelle, en incorporant toutes les informations pertinentes.
        4. **Fournissez votre réponse en français sous forme d'un récit poli et clair.** La réponse doit être un paragraphe unique et simple **sans aucun symbole spécial, phrases introductrices ou commentaires supplémentaires.**

        **Question**:
        {question}
        
        **Options**:
        {options or "Non fourni"}

        **Informations sur l'Entreprise**:
        {json.dumps(user_data, indent=4)}

        **Document Supplémentaire**:
        {document_text or "Non fourni"}
        """


def integrate_document_content_with_grant_writing(
    question, user_data, file_paths, options
):
    """
    Integrates document content into the grant-writing process.
    """
    document_texts = []
    for file_path in file_paths:
        extracted_text = process_uploaded_document(file_path)
        document_texts.append(extracted_text)

    combined_document_text = "\n\n".join(document_texts)
    language = detect_language(question)

    prompt = generate_prompt(
        language, question, user_data, combined_document_text, options
    )

    print(prompt)

    print(f"--------The language is: {language}")

    # Prepare request body for the AI model
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    body = json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": 300,
            "temperature": 0.3,
            # "top_p": 0.9,
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


if __name__ == "__main__":
    # API Endpoint
    user_api_url = (
        "https://api.happly.ai/api/v1/portal/users/ba845892-3275-47a3-9327-fcf7cba266a6"
    )
    headers = {
        "Authorization": "Bearer YOUR_API_KEY"  # Replace with the actual API key
    }

    # Fetch User Data
    user_data = fetch_user_data(user_api_url, headers=headers)
    if not user_data:
        print("Failed to fetch user data. Exiting.")
        exit()

    # # Example Question1
    # question = "Describe your propduct"  # English example
    # options = None

    # Example Question2
    # question = "What is the name of the company?"  # English example
    # options = None

    # Example Question3
    question = "When was the company registered?"  # English example
    options = ["2021", "2022", "2023"]
    # options = None

    # Path to the uploaded PDF document
    uploaded_files = ["./Govago Business Plan.pdf"]

    # Options that might be used for the question

    # Generate Response
    response = integrate_document_content_with_grant_writing(
        question, user_data, uploaded_files, options
    )
    print("Generated Grant Response:\n", response)
