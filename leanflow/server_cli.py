import os
import uvicorn
import argparse
from omegaconf import OmegaConf
from .server import app

def main():
    """
    The main entry point for the leanflow-serve command.
    """
    parser = argparse.ArgumentParser(description="LeanFlow Server: A Python API for Lean.")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="The host to bind the server to. Overrides config and environment variables.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="The port to run the server on. Overrides config and environment variables.",
    )
    
    args = parser.parse_args()
    
    host = "localhost"
    port = 8000
    
    # allow environment variables to override defaults
    host = os.getenv("LEANFLOW_HOST", str(host))
    port = int(os.getenv("LEANFLOW_PORT", str(port)))

    if args.host is not None:
        host = args.host
    if args.port is not None:
        port = args.port

    if args.config is not None:
        app.state.server_config_path = args.config
        cfg = OmegaConf.load(args.config)

        # load host and port from YAML, if they exist
        host = cfg.get("server", {}).get("host", host)
        port = cfg.get("server", {}).get("port", port)

    print(f"Starting LeanFlow Server on http://{host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()