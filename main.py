from fastapi import FastAPI, UploadFile, File, Form

# Initialize the FastAPI app
app = FastAPI(title="MM-GNN VQA API")

@app.get("/")
def read_root():
    return {"message": "Your MM-GNN API is successfully running!"}

@app.post("/vqa")
async def answer_question(
    image: UploadFile = File(...), 
    question: str = Form(...)
):
    # This is a placeholder. 
    # We will plug your ML Extractors and Graph Neural Network in here next!
    return {
        "received_question": question,
        "filename": image.filename,
        "status": "Success! Waiting for ML models to be implemented."
    }
    
