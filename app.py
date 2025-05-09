from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import functions from your prompt_model.py
from prompt_model import find_latest_checkpoint, load_model, generate_response

# Import RAG retriever if you have one
# from rag.retriever import get_context 

# Create the FastAPI app instance - THIS is what uvicorn looks for
app = FastAPI(title="Banking Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    top_k: int = 3

class QueryResponse(BaseModel):
    response: str
    context: Optional[str] = None

# Load model at startup
@app.on_event("startup")
async def startup_event():
    try:
        checkpoint_dir = find_latest_checkpoint()
        tokenizer, model, device = load_model(checkpoint_dir)
        
        # Store in app state for reuse
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.device = device
        
        print(f"Model loaded successfully from {checkpoint_dir}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Banking Q&A API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a question and return the answer"""
    try:
        # Get context if using RAG (uncomment if you have a retriever)
        context = ""
        # if request.use_rag:
        #     from rag.retriever import get_context
        #     context = get_context(request.query, top_k=request.top_k)
        
        # Generate response
        response = generate_response(
            app.state.tokenizer,
            app.state.model,
            app.state.device,
            request.query,
            context
        )
        
        return QueryResponse(
            response=response,
            context=context if context else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# This allows running directly with "python app.py"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)