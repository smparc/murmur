import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="Murmur LLM Telemetry Decoder")

# Configuration
MODEL_NAME = "Qwen/Qwen1.5-1.8B" # Lightweight model for edge/factory inference
GNN_EMBEDDING_DIM = 256          # Output dimension from our ST-GNN
LLM_HIDDEN_DIM = 2048            # Hidden dimension of the chosen LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TelemetryRequest(BaseModel):
    node_id: int
    timestamp: float
    gnn_embedding: list[float]

class EmbeddingProjector(nn.Module):
    """
    Adapter layer to translate the ST-GNN acoustic embedding 
    into the LLM's native latent space.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

# Global model state
tokenizer = None
llm_model = None
projector = None

@app.on_event("startup")
async def load_models():
    """Load the LLM and the trained projection adapter into VRAM."""
    global tokenizer, llm_model, projector
    
    print("[*] Loading Tokenizer and LLM into VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("[*] Initializing Acoustic Projection Adapter...")
    projector = EmbeddingProjector(GNN_EMBEDDING_DIM, LLM_HIDDEN_DIM).to(DEVICE)
    # In production, you would load pre-trained weights here:
    # projector.load_state_dict(torch.load("projector_weights.pth"))
    projector.eval()

@app.post("/generate_telemetry")
async def generate_telemetry(request: TelemetryRequest):
    """
    Receives an acoustic embedding, projects it, and generates a text diagnostic.
    """
    try:
        # 1. Format the raw GNN embedding
        raw_embedding = torch.tensor([request.gnn_embedding], dtype=torch.float32).to(DEVICE)
        
        # 2. Project into LLM latent space
        with torch.no_grad():
            acoustic_prompt_embeds = projector(raw_embedding).to(torch.float16)
        
        # 3. Create the text instruction prefix
        system_prompt = f"System Diagnostic for Node {request.node_id}:\n"
        inputs = tokenizer(system_prompt, return_tensors="pt").to(DEVICE)
        text_embeds = llm_model.get_input_embeddings()(inputs.input_ids)
        
        # 4. Concatenate the text prompt with the acoustic embedding
        # This forces the LLM to condition its text generation on the physical sound
        combined_embeds = torch.cat([text_embeds, acoustic_prompt_embeds.unsqueeze(1)], dim=1)
        
        # 5. Autoregressive Generation
        outputs = llm_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=50,
            temperature=0.2, # Low temperature for deterministic, factual diagnostics
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "node_id": request.node_id,
            "timestamp": request.timestamp,
            "telemetry": generated_text.strip()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)