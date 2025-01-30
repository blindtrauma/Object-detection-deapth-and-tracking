from src.utils.load_func import save_frame_as_image
from src.LLM_functionality.generate_image_summary import analyze_image
def ar_sign_reading():
    print("Executing AR Sign Reading...")

def annotation_from_experts():
    print("Executing Annotation from Experts...")

def job_management_and_safety_checks():
    print("Executing Job Management and Safety Checks...")

def ai_studio():
    print("Executing AI Studio...")

def qr_bar_code_scanner():
    print("Executing QR/Barcode Scanner...")

def ai_image_analysis_and_summary(frame, query):
    print("this is being executed, ai image analysis")
    image_data = save_frame_as_image(frame)
    print(image_data)
    if not query:
        query = None
    target_text =analyze_image(query, frame_link=r'')
    return target_text

    



    
