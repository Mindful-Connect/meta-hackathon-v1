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

        - **Business Information**: Provided as a JSON file containing the personal and business details of the applicant.
        - **Supplemental Document**: Optional additional information extracted from files (e.g., PDFs, Word documents).
        - **Options**: Optional field. If present, the answer must be one of these options only.

        **Instructions**:
        1. **If Options are provided**:
            - **Output only the exact text of the most appropriate option**.
            - **Do not include any additional words, sentences, or symbols**.
        2. **If Options are not provided**:
            - **For specific information requests** (e.g., dates, addresses), provide a precise and accurate response without extra commentary.
            - **For general questions**, craft a detailed, compelling, and professional single-paragraph response using all relevant information.
        3. **Formatting**:
            - **Respond in English**.
            - **Use a single, plain paragraph**.
            - **Do not include special symbols, introductory phrases, or additional comments**.

        **Question**:
        {question}
        
        **Options**:
        {options or "Not provided"}

        **Business Information**:
        {json.dumps(user_data, indent=4)}
        
        **Supplemental Document**:
        {document_text or "Not provided"}

        **Response**:
        """

    elif language == "fr":
        return f"""
        Vous êtes un expert dans la génération de réponses précises, professionnelles et convaincantes aux questions de demande de subvention basées sur les **Informations sur l'entreprise**, le **Document supplémentaire** et les **Options** fournies.

        - **Informations sur l'entreprise** : Fournies sous forme de fichier JSON contenant les détails personnels et commerciaux du demandeur.
        - **Document supplémentaire** : Informations supplémentaires optionnelles extraites de fichiers (par exemple, PDF, documents Word).
        - **Options** : Champ optionnel. Si présent, la réponse doit être uniquement l'une de ces options.

        **Instructions** :
        1. **Si des options sont fournies** :
            - **Fournissez uniquement le texte exact de l'option la plus appropriée**.
            - **N'incluez aucun mot, phrase ou symbole supplémentaire**.
        2. **Si les options ne sont pas fournies** :
            - **Pour les demandes d'informations spécifiques** (par exemple, dates, adresses), fournissez une réponse précise et exacte sans commentaire supplémentaire.
            - **Pour les questions générales**, rédigez une réponse détaillée, convaincante et professionnelle en un seul paragraphe en utilisant toutes les informations pertinentes.
        3. **Formatage** :
            - **Répondez en français**.
            - **Utilisez un seul paragraphe simple**.
            - **N'incluez pas de symboles spéciaux, de phrases introductives ou de commentaires supplémentaires**.

        **Question** :
        {question}

        **Options** :
        {options or "Non fourni"}

        **Informations sur l'entreprise** :
        {json.dumps(user_data, indent=4)}

        **Document supplémentaire** :
        {document_text or "Non fourni"}

        **Réponse** :
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
            "temperature": 0.5,
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
    question = "Describe your propduct"  # English example
    options = None

    # Example Question2
    # question = "What is the name of the company?"  # English example
    # options = None

    # Example Question3
    # question = "When was the company registered?"  # English example
    # # options = ["2021", "2022", "2023"]
    # options = None

    # Path to the uploaded PDF document
    uploaded_files = ["./Govago Business Plan.pdf"]

    # Options that might be used for the question

    # Generate Response
    response = integrate_document_content_with_grant_writing(
        question, user_data, uploaded_files, options
    )
    print("Generated Grant Response:\n", response)
