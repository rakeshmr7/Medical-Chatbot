import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_path
# Set tesseract executable path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image_or_pdf(file_path):
    text = ""
    salutation = ""
    name = ""
    if file_path.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    elif file_path.endswith('.pdf'):
        images = convert_from_path(file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
    # Extract name and age (Accommodating both 'Mr.' and 'Mrs.' with specific format)
    name_age_pattern = r"(Mr|Mrs)\s*\.\s*([A-Z]+\s*\.[A-Z]+)\s*\((\d+\/[MF])\)"
    name_age_match = re.search(name_age_pattern, text, re.IGNORECASE)
    if name_age_match:
     salutation = name_age_match.group(1).strip()
     name = name_age_match.group(2).strip()
     age_gender = name_age_match.group(3).strip()
    lab_report_pattern = r"LABORATORY REPORT(.*?)\* End of Report \*"
    lab_report_match = re.search(lab_report_pattern, text, re.DOTALL)

    if lab_report_match:
     lab_report_content = lab_report_match.group(1).strip()  
    # Extract relevant parts (modify as per your specific pattern requirements)
    text2=salutation+" "+name+" "+age_gender+"Laboratory Report Content:\n"+lab_report_content
    print("Extracted text going to return to bot")
    print("Bot Processing.....")
    return text2


#print(extract_text_from_image_or_pdf("C:\\Users\\rakes\\Documents\\MINI PROJECT\\mchatbot\\temp\\inverted_report.jpg"))