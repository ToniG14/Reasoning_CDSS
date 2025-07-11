#################################
##### ---    IMPORTS    --- #####
#################################

from docling.document_converter import DocumentConverter
import os
import re
from langchain.docstore.document import Document
import uuid


#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   Convert PDF to Markdown Docling   --- #####
def convert_pdf_to_markdown(pdf_path: str, markdown_path: str) -> None:
    """Convert a PDF file to Markdown format using Docling.

    Args:
        pdf_path (str): The path to the PDF file to convert.
        markdown_path (str): The path to save the Markdown file.
    """
    # Create an instance of the document converter
    converter = DocumentConverter()

    # Perform the PDF conversion
    result = converter.convert(pdf_path)

    # Export the content to Markdown format
    markdown_content = result.document.export_to_markdown()

    # Save the content to a Markdown file
    with open(markdown_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"Conversion completed. File saved at: {markdown_path}")

##### ---   Read Markdown Files from local storage   --- #####

def read_markdown_files(folder_path: str) -> list[dict[str, str]]:
    """ Read all Markdown files from a specified folder and return their content.

    Args:
        folder_path (str): The path to the folder containing the Markdown files.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing the file name and content.
    """
    markdown_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):  # Ensure we only process Markdown files
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()
                markdown_files.append({"file_name": file_name, "content": markdown_content})
    return markdown_files

##### ---   Extract References from Markdown Text   --- #####

def find_references(text: str) -> str:
    """Extract references from the text and return them as a comma-separated string.

    Args:
        text (str): The text to extract references from.

    Returns:
        str: A comma-separated string of references found in the text.
    """
    pattern = (
        r'\b(in|see)\s+'
        r'(Table|Tables|Figure|Figures|section|sections|Supplementary Table|Supplementary Tables)\s+'
        r'(\d+)'  # First number
        r'(?:\s+and\s+(\d+))?'  # Additional number without type
        r'(?:\s+and\s+'
        r'(Table|Tables|Figure|Figures|section|sections|Supplementary Table|Supplementary Tables)\s+'
        r'(\d+))?'  # Additional type and number
    )
    references = []
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        base_type = match[1].rstrip('s')  # Remove plural "s" to standardize type
        base_number = match[2]
        references.append(f"{base_type} {base_number}")

        # Handle "and [number]" case
        if match[3]:
            references.append(f"{base_type} {match[3]}")

        # Handle "and [type] [number]" case
        if match[4] and match[5]:
            additional_type = match[4].rstrip('s')  # Remove plural "s"
            additional_number = match[5]
            references.append(f"{additional_type} {additional_number}")

    # Join references into a comma-separated string
    return ", ".join(references) if references else None

##### ---   Remove Duplicate References   --- #####



def remove_duplicate_references(chunks: list[Document]) -> list[Document]:
    """
    Remove duplicate references from the 'References' metadata in the chunks.

    Args:
        chunks (list): List of document chunks, each with a 'References' metadata field.

    Returns:
        list: The cleaned list of chunks with unique references.
    """
    for chunk in chunks:
        if "References" in chunk.metadata:
            # Get the references as a list
            references = chunk.metadata["References"].split(", ")

            # Remove duplicates while maintaining the order
            unique_references = list(dict.fromkeys(references))

            # Rejoin into a string and update the metadata
            chunk.metadata["References"] = ", ".join(unique_references)

    return chunks
