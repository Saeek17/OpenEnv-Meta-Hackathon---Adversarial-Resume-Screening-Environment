from openenv.core.env_server import create_fastapi_app
from .environment import ResumeScreeningEnvironment

# Create the FastAPI app for the Resume Screening Environment
app = create_fastapi_app(ResumeScreeningEnvironment)

def main():
    """Entry point for the environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
