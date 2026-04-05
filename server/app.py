from openenv.core.env_server import create_fastapi_app
from .environment import ResumeScreeningEnvironment
from models import ResumeAction, ResumeObservation

# Create the FastAPI app with required type classes
app = create_fastapi_app(
    ResumeScreeningEnvironment, 
    action_cls=ResumeAction, 
    observation_cls=ResumeObservation
)

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal Server Error",
            "detail": str(exc),
            "traceback": traceback.format_exc()
        }
    )

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def home():
    """Home page for the Hugging Face Space."""
    return """
    <html>
        <head>
            <title>Adversarial Resume Screening Environment</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 50px; background: #f4f7f6; }
                .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; }
                h1 { color: #2d3748; }
                p { color: #4a5568; }
                .status { color: #48bb78; font-weight: bold; }
                .link { color: #4299e1; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>📄 Adversarial Resume Screening</h1>
                <p>Status: <span class="status">ONLINE</span></p>
                <p>This is an OpenEnv environment for evaluating AI agents.</p>
                <hr>
                <p>Endpoints available:</p>
                <code style="background: #edf2f7; padding: 10px; display: block; border-radius: 6px;">
                    POST /reset <br>
                    POST /step <br>
                    GET /health
                </code>
                <p style="margin-top: 20px;"><a class="link" href="/docs">View API Documentation</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    """Health check endpoint required by the mandatory Docker test."""
    return {"status": "healthy"}

def main():
    """Entry point for the environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
