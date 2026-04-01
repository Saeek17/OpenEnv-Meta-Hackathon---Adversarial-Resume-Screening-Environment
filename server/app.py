from openenv.core.env_server import create_fastapi_app
from .environment import ResumeScreeningEnvironment

# Create the FastAPI app for the Resume Screening Environment
app = create_fastapi_app(ResumeScreeningEnvironment)
