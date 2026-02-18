"""
NPS Agent - MLflow deployment wrapper.

This module wraps the shared agent core (nps_agent_core.py) for deployment
via MLflow's ResponsesAgent HTTP API.

The core agent logic is imported from ../nps_agent_core.py to avoid duplication
with the development notebooks.
"""

import asyncio
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path so we can import the shared module
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Import shared agent logic from the core module
from nps_agent_core import run_nps_agent


# ---------------------------------------------------------------------------
# MLflow ResponsesAgent — wraps run_nps_agent into an HTTP API for deployment
# ---------------------------------------------------------------------------
class NPSResponsesAgent(ResponsesAgent):
    """
    MLflow ResponsesAgent wrapper for the NPS Agent.
    
    This class wraps the shared run_nps_agent function to expose it as an
    HTTP endpoint via MLflow's model serving infrastructure.
    """
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        try:
            result = asyncio.run(run_nps_agent(request.input))
        except Exception as e:
            result = f"Error: {e}"
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=result, id="msg_1")]
        )


# ---------------------------------------------------------------------------
# MLflow model registration
# ---------------------------------------------------------------------------
mlflow.openai.autolog()
set_model(NPSResponsesAgent())
