# Project Requirements Document (PRD)

# Project Overview
You are building an Ai-integrated EHR platform which aids clinicians in clinical decision making, documentation, charting,scheduling, data visualization, care orchestration, and more. The platform will be a webapp where the frontend and backend will be containerized and deployed on a cloud service provider such as AWS, GCP, or Azure. The frontend should look like a typical medplum EHR app, but with enhanced capabilities through the use of the LangGraph framework. Specifically, there should be a chat component on the frontend that allows clinicians to chat with the LLM directly. The chat should be accessible from any screen in the EHR, but in order for the clinician to make requests about a specific patient, they must first be in that patients chart. The chat history should be automatically stored in postgres on the backend in a way that enables retrieval of the chat history for a given patient or for general retrieval across all patients if the chat window is being accessed outside of a patients chart. The chat component should be able to handle multiple, ongoing conversations, and switch between them. Additionally, when the clinician makes a request, the frontend will send the request to the backend, where it will be processed by the LangGraph agent. The agent will return a response, which will be streamed to the frontend. The agent will always respond with a text response of what it plans to do, but as it uses tools downstream, the frontend will be able to display the results of the agent's actions to the side of the chat. For example, if the clinician requests a list of orders, the agent will return a response with a list of orders, and the frontend will display the orders to the side of the chat. Another example – if the clinician requests data for a patient to be visualized, the data returned by the agent will be visualized on the frontend alongside the chat.

You will be using pydantic, LangGraph, LangChain, FastAPI, fhirclient, django, and postgres on the backend, while on the frontend you will be using Vercel AI SDK, Medplum, shadcn, and Next.js 15, tailwind, lucid icon.

## Core functionalities
# Core Functionalities

Below is a list of core functionalities for the AI-integrated EHR platform, broken down into frontend and backend components.

---

## Frontend Functionalities

1. **User Interface for Standard EHR Features**
   - **Patient Charting:** Display patient information, medical history, and ongoing treatments.
   - **Documentation:** Enable clinicians to create and edit patient notes and medical records.
   - **Scheduling:** Provide a calendar interface for managing appointments and clinical schedules.
   - **Care Orchestration:** Facilitate coordination among care teams with task assignments and notifications.
   - **Data Visualization:** Present patient data in graphical formats like charts and graphs.

2. **Integrated Chat Component**
   - **Universal Accessibility:** Accessible from any screen within the EHR platform.
   - **LLM Interaction:** Allow clinicians to chat directly with the Language Learning Model (LLM).
   - **Context Awareness:** Enable patient-specific queries when within a patient's chart.
   - **Multiple Conversations:** Support multiple ongoing chat sessions with easy switching between them.

3. **Chat History Management**
   - **Automatic Storage:** Save all chat interactions automatically.
   - **Patient-Specific History:** Retrieve chat history related to a specific patient when in their chart.
   - **General History:** Access general chat history when outside any patient's chart.
   - **Display Past Conversations:** Show previous messages within the chat interface for continuity.

4. **Streaming Responses**
   - **Real-Time Updates:** Display the agent's responses as they are received from the backend.
   - **Partial Results:** Show partial responses to improve user engagement and reduce perceived latency.

5. **Agent Action Results Display**
   - **Side Panel Visualization:** Present results of the agent's actions (e.g., lists of orders, data visualizations) alongside the chat.
   - **Interactive Elements:** Allow clinicians to interact with the displayed results (e.g., click on an order to see details).

6. **Visualization Components**
   - **Dynamic Charts and Graphs:** Render patient data returned by the agent in visual formats.
   - **Customizable Views:** Let users adjust visualization parameters (e.g., time range, data types).

7. **User Authentication and Session Management**
   - **Secure Login Interface:** Implement robust authentication mechanisms.
   - **Session Persistence:** Maintain user sessions securely across different devices and sessions.
   - **Role-Based Access Control:** Restrict functionalities based on user roles and permissions.

8. **Responsive and Intuitive Design**
   - **Mobile Compatibility:** Ensure the platform is usable on tablets and mobile devices.
   - **User-Friendly Interface:** Design intuitive navigation and user workflows.
   - **Accessibility Compliance:** Adhere to standards like WCAG to support users with disabilities.

9. **Notifications and Alerts**
   - **Real-Time Notifications:** Inform clinicians of important events or updates.
   - **Customizable Alerts:** Allow users to set preferences for the types of alerts they receive.

10. **Error Handling and User Feedback**
    - **Graceful Degradation:** Provide meaningful error messages and fallback options.
    - **Loading Indicators:** Show progress indicators during data fetching or processing.

---

## Backend Functionalities

1. **LangGraph Agent Integration**
   - **Request Processing:** Handle incoming requests from the frontend chat component.
   - **LLM Communication:** Interface with the LLM to generate responses.
   - **Tool Utilization:** Enable the agent to use various tools for data retrieval and processing.

2. **Chat History Storage**
   - **Database Management:** Use PostgreSQL to store chat histories.
   - **Efficient Retrieval:** Optimize queries for quick access to chat histories based on patient context or general context.
   - **Data Encryption:** Secure chat data both at rest and in transit.

3. **Conversation Management**
   - **Session Tracking:** Manage multiple chat sessions per user with unique identifiers.
   - **Context Handling:** Maintain context awareness for patient-specific and general conversations.
   - **Scalability:** Ensure the system can handle multiple concurrent users and conversations.

4. **Streaming Response Support**
   - **Real-Time Communication:** Stream agent responses to the frontend as they are generated.
   - **Partial Data Handling:** Send partial results to improve responsiveness.

5. **Agent Action Execution**
   - **Data Retrieval:** Fetch data like orders or patient records as requested by the agent.
   - **Data Processing:** Perform computations or data transformations as needed.
   - **External API Integration:** Connect with other systems or services for additional functionalities.

6. **EHR Data APIs**
   - **Patient Data API:** Provide endpoints for accessing patient information securely.
   - **Orders and Results API:** Manage laboratory orders, results, and other clinical data.
   - **Scheduling API:** Allow interaction with appointment and resource schedules.
   - **Documentation API:** Enable creation and retrieval of clinical notes and documents.

7. **User Authentication and Authorization**
   - **Secure Authentication Services:** Implement OAuth 2.0 or similar protocols.
   - **Authorization Checks:** Enforce permissions for data access and actions.
   - **Audit Logging:** Record access and changes for compliance and security monitoring.

8. **Integration with Existing EHR Systems**
   - **Data Synchronization:** Keep data consistent between the platform and existing systems.
   - **Interoperability Standards:** Use HL7 FHIR or other standards for data exchange.
   - **Migration Tools:** Provide utilities for importing data from legacy systems.

9. **Security and Compliance**
   - **HIPAA Compliance:** Ensure all data handling meets regulatory requirements.
   - **Encryption Standards:** Use SSL/TLS for data in transit and encryption algorithms for data at rest.
   - **Penetration Testing:** Regularly test the system for vulnerabilities.

10. **Logging and Monitoring**
    - **System Logs:** Maintain logs for system activities and errors.
    - **Performance Monitoring:** Track system performance metrics and resource usage.
    - **Alerting Mechanisms:** Set up alerts for system failures or performance issues.

11. **Containerization and Deployment**
    - **Dockerization:** Containerize backend services for consistent deployment environments.
    - **Cloud Deployment:** Deploy containers on AWS, GCP, or Azure with Kubernetes or similar orchestration.
    - **CI/CD Pipelines:** Implement continuous integration and deployment workflows.

12. **Error Handling and Exception Management**
    - **Robust Error Responses:** Return meaningful error messages to the frontend.
    - **Fallback Mechanisms:** Provide default responses or actions when failures occur.

13. **Session Management**
    - **Secure Tokens:** Use JWTs or similar tokens for session management.
    - **Session Expiration:** Handle session timeouts and refresh mechanisms.

14. **API Security**
    - **Rate Limiting:** Prevent abuse by limiting the number of requests.
    - **Input Validation:** Protect against injection attacks by validating all inputs.
    - **Authentication Middleware:** Enforce security checks at the API gateway level.

15. **Scalability and Load Balancing**
    - **Horizontal Scaling:** Add more instances to handle increased load.
    - **Load Balancers:** Distribute incoming traffic efficiently across servers.
    - **Auto-Scaling:** Automatically adjust resources based on demand.

16. **Data Analytics and Reporting**
    - **Usage Metrics:** Collect data on how the platform is used for continual improvement.
    - **Reporting Tools:** Generate reports for administrative and compliance purposes.

17. **Audit Trails**
    - **Comprehensive Logging:** Record all user actions and system changes.
    - **Immutable Logs:** Ensure logs cannot be tampered with or deleted.

18. **Internationalization and Localization**
    - **Multi-Language Support:** Enable the platform to support multiple languages if required.
    - **Date and Number Formats:** Adapt to regional settings and conventions.

19. **Asynchronous Task Handling**
    - **Background Processing:** Use task queues for long-running operations.
    - **Notification of Completion:** Inform the frontend when background tasks are completed.


# Current File Structure
gen-ui-python
├── README.md
├── backend
│   ├── LICENSE
│   ├── Makefile
│   ├── README.md
│   ├── gen_ui_backend
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── auth.py
│   │   ├── chain.py
│   │   ├── py.typed
│   │   ├── requirements.txt
│   │   ├── server.py
│   │   ├── tools
│   │   └── types.py
│   ├── langgraph.json
│   ├── poetry.lock
│   ├── pyproject.toml
│   └── scripts
│       ├── check_imports.py
│       └── lint_imports.sh
├── filetree.py
├── frontend
│   ├── ai
│   │   └── message.tsx
│   ├── app
│   │   ├── agent.tsx
│   │   ├── favicon.ico
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── shared.tsx
│   ├── components
│   │   ├── prebuilt
│   │   ├── tools
│   │   └── ui
│   ├── components.json
│   ├── lib
│   │   ├── mui.ts
│   │   └── utils.ts
│   ├── next-env.d.ts
│   ├── next.config.mjs
│   ├── package.json
│   ├── postcss.config.mjs
│   ├── public
│   │   ├── gen_ui_charts_diagram.png
│   │   ├── gen_ui_diagram.png
│   │   ├── next.svg
│   │   └── vercel.svg
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── utils
│   │   ├── client.tsx
│   │   └── server.tsx
│   └── yarn.lock
└── instructions.md

# Docs


## Documentation for interacting with medplum fhir server
```python
import logging  # noqa: I001
import os
from typing import Any, Type
from venv import logger

from fhirclient.client import FHIRClient
from gen_ui_backend.auth import MedPlumOAuth2Auth, TokenManager

logger = logging.getLogger(__name__)


def get_fhir_client(patient_id: str | None = None) -> FHIRClient:

    client_settings = {
        "app_id": os.environ.get("MEDPLUM_CLIENT_ID", "123"),
        "app_secret": os.environ.get("MEDPLUM_CLIENT_SECRET", "123"),
        "api_base": os.environ.get("MEDPLUM_BASE_URL"),
        "patient_id": patient_id or "919d82f5-bd4d-45fb-a637-7c6c485640ee",
    }

    client = FHIRClient(settings=client_settings)
    token_manager = TokenManager(
        os.environ.get("MEDPLUM_CLIENT_ID", "123"),
        os.environ.get("MEDPLUM_CLIENT_SECRET", "123"),
        "https://api.medplum.dev.automated.co/oauth2/token",
    )

    oauth2_auth = MedPlumOAuth2Auth(token_manager)
    oauth2_auth.app_id = os.environ.get("MEDPLUM_CLIENT_ID")
    oauth2_auth.app_secret = os.environ.get("MEDPLUM_CLIENT_SECRET")
    oauth2_auth.aud = client.server.aud  # type: ignore
    oauth2_auth._token_uri = "https://api.medplum.dev.automated.co/oauth2/token"

    client.server.auth = oauth2_auth  # type: ignore
    return client


class FhirCrudOperations:
    def __init__(self) -> None:
        self.fhir_client = get_fhir_client()

    def _get_resource_class(self, resource_type: str) -> Type:  # type: ignore
        try:
            resource_module = __import__(
                f"fhirclient.models.{resource_type.lower()}", fromlist=[resource_type]
            )
            resource_class = getattr(resource_module, resource_type)
            return resource_class
        except (ImportError, AttributeError) as err:
            logger.error(f"Resource type '{resource_type}' is not supported.")
            raise ValueError(
                f"Resource type '{resource_type}' is not supported."
            ) from err

    def _validate_resource(self, resource: Type) -> None:  # type: ignore
        errors = resource.validate()
        if errors:
            logger.error(f"Validation errors: {errors}")
            raise ValueError(f"Validation errors: {errors}")
            # TODO: add in llm-based error handling & retry logic

    def create_resource(self, resource_type: str, resource_data: dict[str, Any]) -> Any:
        resource_class = self._get_resource_class(resource_type)
        resource = resource_class(resource_data)
        self._validate_resource(resource)
        created_resource = resource.create(server=self.fhir_client.server)
        return {"content": str(created_resource.as_json())}

    def read_resource(self, resource_type: str, resource_id: str) -> Any:
        resource_class = self._get_resource_class(resource_type)
        resource = resource_class.read(resource_id, server=self.fhir_client.server)
        return {"content": str(resource.as_json())}

    def update_resource(
        self, resource_type: str, resource_id: str, updated_data: dict[str, Any]
    ) -> Any:
        resource_class = self._get_resource_class(resource_type)
        resource = resource_class.read(resource_id, server=self.fhir_client.server)
        for key, value in updated_data.items():
            setattr(resource, key, value)
        self._validate_resource(resource)
        updated_resource = resource.update(server=self.fhir_client.server)
        return {"content": str(updated_resource.as_json())}

    def delete_resource(self, resource_type: str, resource_id: str) -> Any:
        resource_class = self._get_resource_class(resource_type)
        resource = resource_class.read(resource_id, server=self.fhir_client.server)
        deleted = resource.delete(server=self.fhir_client.server)
        return {"content": str(deleted.as_json())}

    def get_resources(
        self, resource_type: str, search_params: dict[str, Any] | None = None
    ) -> Any:
        resource_class = self._get_resource_class(resource_type)
        search = (
            resource_class.where(search_params)
            if search_params
            else resource_class.where()
        )
        resources = search.perform(server=self.fhir_client.server)
        return {"content": str(resources.as_json())}
```

## documentation for creating FHIR CRUD tools for the LangGraph agent to use

```python
from typing import Any, Dict, Optional

from gen_ui_backend.tools.fhir.client import FhirCrudOperations
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# Define the input schema for the FHIR CRUD tool
class FHIRCrudInput(BaseModel):
    operation: str = Field(
        ...,
        description="The CRUD operation to perform. One of 'create', 'read', 'update', or 'delete'.",
    )
    resource_type: str = Field(
        ..., description="The type of FHIR resource (e.g., 'Patient', 'Observation')."
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "The data for the resource. For 'create' and 'update' operations, "
            "this is the resource data. For 'read' and 'delete' operations, "
            "this should contain the 'id' of the resource."
        ),
    )


# Define the FHIR CRUD tool using the @tool decorator
@tool("fhir-crud", args_schema=FHIRCrudInput, return_direct=True)
def fhir_crud(
    operation: str, resource_type: str, data: Optional[Dict[str, Any]]
) -> Any:
    """Performs CRUD operations on FHIR resources."""
    crud_operator = FhirCrudOperations()

    if operation == "create":
        if data is None:
            return "Data is required for create operation."
        result = crud_operator.create_resource(resource_type, data)
    elif operation == "read":
        resource_id = data.get("id") if data else None
        if not resource_id:
            return "Resource ID is required for read operation."
        result = crud_operator.read_resource(resource_type, resource_id)
    elif operation == "update":
        resource_id = data.get("id") if data else None
        if not resource_id or not data:
            return "Resource ID and data are required for update operation."
        result = crud_operator.update_resource(resource_type, resource_id, data)
    elif operation == "delete":
        resource_id = data.get("id") if data else None
        if not resource_id:
            return "Resource ID is required for delete operation."
        result = crud_operator.delete_resource(resource_type, resource_id)
    else:
        result = f"Unsupported operation: {operation}"
    return result
```

## Documentation for authorixzation with the Medplum FHIR server
```python
import base64
import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from fhirclient.auth import FHIROAuth2Auth

logger = logging.getLogger(__name__)


class TokenManager:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.access_token = None
        self.expiration_time = 0

    def get_token(self) -> str:
        current_time = time.time()
        if self.access_token and current_time < self.expiration_time:
            return self.access_token
        else:
            self._fetch_new_token()
            return self.access_token  # type: ignore

    def _fetch_new_token(self) -> None:
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode("ascii")
        auth_base64 = base64.b64encode(auth_bytes).decode("ascii")

        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}

        if self.scope:
            data["scope"] = self.scope

        response = requests.post(self.token_url, headers=headers, data=data)

        if response.status_code == 200:
            token_response = response.json()
            self.access_token = token_response.get("access_token")
            expires_in = token_response.get(
                "expires_in", 3600
            )  # Default to 1 hour if not provided
            self.expiration_time = (
                time.time() + expires_in - 60
            )  # Subtract 60 seconds as a buffer
        else:
            error_message = f"Failed to obtain access token. Status code: {response.status_code}, Error: {response.text}"
            raise Exception(error_message)


def get_token_manager() -> TokenManager:
    return TokenManager(
        os.environ.get("MEDPLUM_CLIENT_ID", "123"),
        os.environ.get("MEDPLUM_CLIENT_SECRET", "123"),
        "https://api.medplum.dev.automated.co/oauth2/token",
    )


class MedPlumOAuth2Auth(FHIROAuth2Auth):  # type: ignore
    def __init__(
        self, token_manager: TokenManager, state: Optional[str] = None
    ) -> None:
        super().__init__(state=state)
        self.token_manager = token_manager

    @property
    def ready(self) -> bool:
        return True

    def can_sign_headers(self) -> bool:
        return True

    def signed_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Returns updated HTTP request headers using the access token from TokenManager."""
        access_token = self.token_manager.get_token()
        if not access_token:
            raise Exception("Cannot sign headers since I have no access token")

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def authorize(self, server: Any) -> None:
        """No action needed; TokenManager handles authorization."""
        pass

    def reauthorize(self, server: Any) -> None:
        """No action needed; TokenManager handles token refreshing."""
        pass

    @property
    def state(self) -> Any:
        s = super().state

        return s

    def from_state(self, state: Any) -> None:
        super().from_state(state)
```

## Documentation for creating a simple LangGraph agent
```python
import os
from typing import List, Optional, TypedDict

from gen_ui_backend.tools.fhir.crud_tool import fhir_crud
from gen_ui_backend.tools.github import github_repo
from gen_ui_backend.tools.invoice import invoice_parser
from gen_ui_backend.tools.weather import weather_data
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph


class GenerativeUIState(TypedDict, total=False):
    input: HumanMessage
    result: Optional[str]
    """Plain text response if no tool was used."""
    tool_calls: Optional[List[dict]]
    """A list of parsed tool calls."""
    tool_result: Optional[dict]
    """The result of a tool call."""
    patient_id: Optional[str]


def invoke_model(state: GenerativeUIState, config: RunnableConfig) -> GenerativeUIState:
    tools_parser = JsonOutputToolsParser()
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You're provided a list of tools, and an input from the user.\n"
                + "Your job is to determine whether or not you have a tool which can handle the users input, or respond with plain text.",
            ),
            MessagesPlaceholder("input"),
        ]
    )
    AZURE_OPENAI_API_BASE = os.environ.get(
        "AZURE_OPENAI_API_BASE",
        "https://bionic-health-openai-eastus-2.openai.azure.com/",
    )
    AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT_NAME", "bionic-health-gpt-4o-structured-output"
    )
    model = AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", "123"),  # type: ignore
        azure_endpoint=AZURE_OPENAI_API_BASE,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version="2023-03-15-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    tools = [github_repo, invoice_parser, weather_data, fhir_crud]
    model_with_tools = model.bind_tools(tools)
    chain = initial_prompt | model_with_tools
    result = chain.invoke(
        {"input": state["input"]},
        config,
    )

    if not isinstance(result, AIMessage):
        raise ValueError("Invalid result from model. Expected AIMessage.")

    if isinstance(result.tool_calls, list) and len(result.tool_calls) > 0:
        parsed_tools = tools_parser.invoke(result, config)
        return {"tool_calls": parsed_tools}
    else:
        return {"result": str(result.content)}


def invoke_tools_or_return(state: GenerativeUIState) -> str:
    if "result" in state and isinstance(state["result"], str):
        return END
    elif "tool_calls" in state and isinstance(state["tool_calls"], list):
        return "invoke_tools"
    else:
        raise ValueError("Invalid state. No result or tool calls found.")


def invoke_tools(state: GenerativeUIState) -> GenerativeUIState:
    tools_map = {
        "fhir-crud": fhir_crud,
    }

    if state["tool_calls"] is not None:
        tool = state["tool_calls"][0]
        selected_tool = tools_map[tool["type"]]
        return {"tool_result": selected_tool.invoke(tool["args"])}
    else:
        raise ValueError("No tool calls found in state.")


def create_graph() -> CompiledGraph:
    workflow = StateGraph(GenerativeUIState)

    workflow.add_node("invoke_model", invoke_model)  # type: ignore
    workflow.add_node("invoke_tools", invoke_tools)
    workflow.add_conditional_edges("invoke_model", invoke_tools_or_return)
    workflow.set_entry_point("invoke_model")
    workflow.set_finish_point("invoke_tools")

    graph = workflow.compile()
    return graph
```

## Documentation for sending messages to the LangGraph agent
```typescript
"use server"; // Mark this file as server-side

import { AIMessage } from "@/ai/message";
import { ContentLoading, FHIRContext } from "@/components/prebuilt/fhir-content";
import { EventHandlerFields, exposeEndpoints, streamRunnableUI } from "@/utils/server";
import { RemoteRunnable } from "@langchain/core/runnables/remote";
import { StreamEvent } from "@langchain/core/tracers/log_stream";
import { createStreamableUI, createStreamableValue } from "ai/rsc";
import "server-only"; // Keep this line

const API_URL = "http://localhost:8000/chat";

type ToolComponent = {
  loading: (props?: any) => JSX.Element;
  final: (props?: any) => JSX.Element;
};

type ToolComponentMap = {
  [tool: string]: ToolComponent;
};

const TOOL_COMPONENT_MAP: ToolComponentMap = {
  "fhir-crud": {
    loading: (props?: any) => <ContentLoading {...props} />,
    final: (props?: any) => <FHIRContext {...props} />,
  },
};

async function agent(inputs: {
  input: string;
  chat_history: [role: string, content: string][];
  file?: {
    base64: string;
    extension: string;
  };
}) {
  const remoteRunnable = new RemoteRunnable({
    url: API_URL,
  });

  let selectedToolComponent: ToolComponent | null = null;
  let selectedToolUI: ReturnType<typeof createStreamableUI> | null = null;

  const handleInvokeToolsEvent = (event: StreamEvent) => {
    const [type] = event.event.split("_").slice(2);
    if (
      type !== "end" ||
      !event.data.output ||
      typeof event.data.output !== "object" ||
      event.name !== "invoke_tools"
    ) {
      return;
    }

    if (selectedToolUI && selectedToolComponent) {
      const toolData = event.data.output.tool_result;
      selectedToolUI.done(selectedToolComponent.final(toolData));
      // Instead of setResponse, return the toolData
      return toolData; // Return the response data
    }
  };

  /**
   * Handles the 'invoke_model' event by checking for tool calls in the output.
   * If a tool call is found and no tool component is selected yet, it sets the
   * selected tool component based on the tool type and appends its loading state to the UI.
   *
   * @param output - The output object from the 'invoke_model' event
   */
  const handleInvokeModelEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    const [type] = event.event.split("_").slice(2);
    if (
      type !== "end" ||
      !event.data.output ||
      typeof event.data.output !== "object" ||
      event.name !== "invoke_model"
    ) {
      return;
    }

    if (
      "tool_calls" in event.data.output &&
      event.data.output.tool_calls.length > 0
    ) {
      const toolCall = event.data.output.tool_calls[0];
      if (!selectedToolComponent && !selectedToolUI) {
        selectedToolComponent = TOOL_COMPONENT_MAP[toolCall.type];
        selectedToolUI = createStreamableUI(selectedToolComponent.loading());
        fields.ui.append(selectedToolUI?.value);
      }
    }
  };

  /**
   * Handles the 'on_chat_model_stream' event by creating a new text stream
   * for the AI message if one doesn't exist for the current run ID.
   * It then appends the chunk content to the corresponding text stream.
   *
   * @param streamEvent - The stream event object
   * @param chunk - The chunk object containing the content
   */
  const handleChatModelStreamEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    if (
      event.event !== "on_chat_model_stream" ||
      !event.data.chunk ||
      typeof event.data.chunk !== "object"
    )
      return;
    if (!fields.callbacks[event.run_id]) {
      const textStream = createStreamableValue();
      fields.ui.append(<AIMessage value={textStream.value} />);
      fields.callbacks[event.run_id] = textStream;
    }

    if (fields.callbacks[event.run_id]) {
      fields.callbacks[event.run_id].append(event.data.chunk.content);
    }
  };

  return streamRunnableUI(
    remoteRunnable,
    {
      input: [
        ...inputs.chat_history.map(([role, content]) => ({
          type: role,
          content,
        })),
        {
          type: "human",
          content: inputs.input,
        },
      ],
    },
    {
      eventHandlers: [
        handleInvokeModelEvent,
        handleInvokeToolsEvent,
        handleChatModelStreamEvent,
      ],
    },
  );
}

// Expose the agent function correctly without setResponse
export const EndpointsContext = exposeEndpoints({ agent: (inputs) => agent(inputs) });


```
