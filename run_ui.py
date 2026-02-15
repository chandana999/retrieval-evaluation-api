#!/usr/bin/env python3
"""
Run the Advanced Retrieval Evaluation UI.
Starts the FastAPI server with frontend and API.
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import uvicorn
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        reload_dirs=[str(root)] if Path(root / "api").exists() else None,
    )
