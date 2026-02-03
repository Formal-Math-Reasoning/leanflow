from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from .server import Server
from .utils import BANNER
from ..utils import logger

class RunRequest(BaseModel):
    """Request model for executing a Lean command."""
    command: str
    env: Optional[int] = None

class RunResponse(BaseModel):
    """Response model containing the execution result."""
    result: dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str

app = FastAPI(title="LeanFlow Server", version="0.0.2")

@app.on_event("startup")
async def startup_event():
    """Initializes the server on startup."""
    print(BANNER)
    global server
    config_path = getattr(app.state, "server_config_path", "config.yml")
    server = await Server.create_from_yaml(config_path)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on shutdown."""
    if "server" in globals():
        await server.shutdown()

@app.get("/status", response_model=HealthResponse, tags=["Monitoring"])
async def status_check():
    """Checks the server status.

    Returns:
        (HealthResponse): The status of the server.
    """
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse, tags=["Execution"])
async def run_endpoint(request: RunRequest):
    """Executes a single Lean command.

    Args:
        request (RunRequest): The execution request containing the command and optional environment ID.

    Returns:
        (RunResponse): The result of the execution.

    Raises:
        HTTPException: If an internal error occurs.
    """
    try:
        result = await server.run(request.command, env=request.env)
        return RunResponse(result=result.serialize())
    except Exception as e:
        logger.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.delete("/env/{env_id}", tags=["Management"])
async def delete_env_endpoint(env_id: int):
    """Deletes a specific environment.

    Args:
        env_id (int): The ID of the environment to delete.

    Returns:
        (dict[str, Any]): Status of the deletion.

    Raises:
        HTTPException: If the environment is not found.
    """
    success = await server.delete_environment(env_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Environment {env_id} not found.")
    return {"status": "deleted", "env_id": env_id}