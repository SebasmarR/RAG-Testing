from pathlib import Path
from pypdf import PdfReader
import json
from PIL import Image
import pytesseract
from docx import Document
import time

REGISTRY_FILE = "loaded_files.json"


def is_already_loaded(file_path: str) -> bool:
    """
    Check if a file has already been loaded by looking it up in the registry.

    Args:
        file_path (str): The path to the file to check.
    Returns:
        bool: True if the file has already been loaded, False otherwise.
    """
    if not Path(REGISTRY_FILE).exists():
        return False

    with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
        loaded_files = json.load(f)

    return file_path in loaded_files


def save_to_registry(file_hash: str, file_name: str):
    """
    Save the file hash and name to the registry.

    Args:
        file_hash (str): The hash of the file.
        file_name (str): The name of the file.

    Returns:
        None
    """
    registry = {}

    if Path(REGISTRY_FILE).exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)

    registry[file_hash] = {
        "filename": file_name,
        "timestamp": time.time()
    }

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)


def load_folder(folder_path: str):
    """
    Load all files from a folder and return their contents as a list of strings.

    Args:
        folder_path (str): The path to the folder containing the files.
    Returns:
        list: A list of strings, each representing the content of a file.
    """

    path = Path(folder_path)

    if not path.exists():
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"The path {folder_path} is not a directory.")

    contents = []
    for file in path.iterdir():
        if file.is_file():
            try:
                content = load_file(file)
                contents.append(content)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return contents


def load_file(file_path: str):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if path.suffix.lower() == ".pdf":
        text = load_pdf(path)

    elif path.suffix.lower() in [".txt", ".md"]:
        text = load_txt_md(path)

    elif path.suffix.lower() == ".json":
        text = load_json(path)

    elif path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        text = load_image(path)

    elif path.suffix.lower() in [".docx", ".doc"]:
        text = load_word(path)

    else:
        raise ValueError(
            f"The file {file_path} has an unsupported format. Supported formats are: PDF, TXT, MD, JSON, PNG, JPG, JPEG, BMP, TIFF, DOCX, DOC.")

    return text


def load_pdf(file_path: Path):

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"The file {file_path} is not a PDF.")

    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text.strip()


def load_txt_md(file_path: Path):
    if file_path.suffix.lower() != ".txt" and file_path.suffix.lower() != ".md":
        raise ValueError(f"The file {file_path} is not a TXT or MD file.")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text.strip()


def load_json(file_path: Path):
    if file_path.suffix.lower() != ".json":
        raise ValueError(f"The file {file_path} is not a JSON file.")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text


def load_image(file_path: Path):
    if file_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        raise ValueError(
            f"The file {file_path} is not a supported image format.")

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    return text.strip()


def load_word(file_path: Path):
    if file_path.suffix.lower() not in [".docx", ".doc"]:
        raise ValueError(f"The file {file_path} is not a DOCX file.")

    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])

    return text.strip()
