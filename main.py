from fastapi import FastAPI, UploadFile, File, Form
from mm_gnn.extraction.feature_extractor import MultiModalExtractor
from mm_gnn.mm_gnn_core import MMGNN_Pipeline

app = FastAPI(title="MM-GNN VQA API")

# Load models into memory once when the server starts
extractor = MultiModalExtractor()
mm_gnn_model = MMGNN_Pipeline()

@app.get("/")
def read_root():
    return {"message": "Your MM-GNN API is successfully running!"}

@app.post("/vqa")
async def answer_question(
    image: UploadFile = File(...), 
    question: str = Form(...)
):
    try:
        # 1. Read the uploaded image bytes
        image_bytes = await image.read()
        
        # 2. Extract visual and text features
        extracted_data = extractor.extract(image_bytes, question)
        
        # 3. Pass through the Graph Neural Network to predict an answer
        predicted_answer = mm_gnn_model(extracted_data)
        
        return {
            "question": question,
            "predicted_answer": predicted_answer,
            "extracted_text_tokens": extracted_data["raw_texts"],
            "status": "Success"
        }
    except Exception as e:
        return {"error": str(e), "status": "Failed"}