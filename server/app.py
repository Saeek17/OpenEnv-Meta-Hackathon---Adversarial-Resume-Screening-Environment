from openenv.core.env_server import create_fastapi_app
from .environment import ResumeScreeningEnvironment
from models import ResumeAction, ResumeObservation

# Create the FastAPI app with required type classes
app = create_fastapi_app(
    ResumeScreeningEnvironment, 
    action_cls=ResumeAction, 
    observation_cls=ResumeObservation
)

def main():
    """Entry point for the environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
