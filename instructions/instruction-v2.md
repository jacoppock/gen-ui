# Project Requirements Document (PRD)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [File Structure](#file-structure)
4. [Core Functionalities](#core-functionalities)
   - [Frontend Functionalities](#frontend-functionalities)
   - [Backend Functionalities](#backend-functionalities)
5. [Documentation](#documentation)
   - [Interacting with Medplum FHIR Server](#interacting-with-medplum-fhir-server)
   - [Creating FHIR CRUD Tools for LangGraph Agent](#creating-fhir-crud-tools-for-langgraph-agent)
   - [Authorization with Medplum FHIR Server](#authorization-with-medplum-fhir-server)
   - [Creating a Simple LangGraph Agent](#creating-a-simple-langgraph-agent)
   - [Sending Messages to the LangGraph Agent](#sending-messages-to-the-langgraph-agent)

---

## Project Overview

We are developing an **AI-assisted Patient Onboarding Application** designed to streamline the onboarding process by leveraging artificial intelligence. The application will guide patients through the onboarding steps via a chat interface, collecting necessary information interactively. As the patient provides information, the adjacent panel will dynamically display components related to the submitted data, progressively building out their profile.

### Key Features

- **Interactive Chat Interface**: The frontend will consist of a chat interface that guides patients through the onboarding process using AI.
- **Dynamic Data Visualization**: An adjacent panel will update in real-time to display components related to the information the patient has provided.
- **AI-Driven Data Collection**: The backend will use an AI agent to process patient inputs, validate information, and store data securely.
- **Progressive Profile Building**: As the patient interacts with the chat, their profile is progressively built and visualized.

---

## Tech Stack

### Backend

- **Languages & Frameworks**: Python, FastAPI, Django
- **Libraries**:
  - **LangGraph**: For building the agent's computational graph.
  - **LangChain**: For language model interactions.
  - **Pydantic**: For data validation and settings management.
  - **FHIRClient**: For interacting with FHIR-compliant EHR systems.
- **Database**: PostgreSQL (for storing chat histories and other persistent data)
- **Other**: Docker (for containerization), Cloud services (AWS, GCP, or Azure)

### Frontend

- **Languages & Frameworks**: TypeScript, Next.js 15
- **Libraries**:
  - **Medplum**: For EHR UI components.
  - **Vercel AI SDK**: For AI integrations.
  - **shadcn**: For UI components.
  - **Tailwind CSS**: For styling.
  - **Lucide Icons**: For iconography.

---

## File Structure

To maintain clarity and facilitate development, the project files are structured as follows:

```
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
```
---

## Core Functionalities

### Frontend Functionalities

1. **Interactive Chat Interface**
   - **AI-Guided Onboarding**: Use AI to guide patients through onboarding questions.
   - **User Input Handling**: Accept various types of input (text, selections).
   - **Real-Time Response**: Provide immediate feedback and guidance.

2. **Dynamic Data Visualization Panel**
   - **Progressive Profile Display**: As patients provide information, the panel updates to reflect new data.
   - **Component Rendering**: Display forms, summaries, or visual elements related to the submitted information.
   - **Interactive Elements**: Allow patients to review and edit provided information.

3. **Chat History Management**
   - **Session Persistence**: Maintain chat history throughout the onboarding session.
   - **Data Security**: Ensure sensitive information is handled securely.
   - **Conversation Continuity**: Allow patients to pause and resume onboarding without loss of progress.

4. **User Authentication and Verification**
   - **Secure Access**: Authenticate patients before starting the onboarding process.
   - **Identity Verification**: Implement measures to verify patient identity if required.

5. **Responsive and Intuitive Design**
   - **Mobile Compatibility**: Ensure usability on mobile devices.
   - **User-Friendly Interface**: Simplify navigation and interaction.
   - **Accessibility Compliance**: Support users with disabilities by adhering to standards like WCAG.

6. **Error Handling and User Feedback**
   - **Validation Messages**: Inform users of invalid inputs.
   - **Guidance Prompts**: Provide helpful suggestions when users are unsure.
   - **Loading Indicators**: Show progress indicators during data processing.

### Backend Functionalities

1. **AI Agent Integration**
   - **Input Processing**: Handle inputs from the frontend chat interface.
   - **Dynamic Question Generation**: Use AI to generate appropriate follow-up questions.
   - **Data Validation**: Validate the information provided by patients.

2. **Data Storage and Management**
   - **Patient Data Storage**: Securely store onboarding data in PostgreSQL.
   - **Data Encryption**: Protect data at rest and in transit.
   - **Compliance**: Ensure data handling meets regulatory requirements (e.g., HIPAA).

3. **FHIR Interoperability**
   - **FHIR-Compliant Data Models**: Store patient data using FHIR standards.
   - **Integration with EHR Systems**: Enable future interoperability with full EHR systems.

4. **Session and State Management**
   - **Session Tracking**: Manage patient onboarding sessions.
   - **State Persistence**: Maintain state between interactions.
   - **Scalability**: Support multiple concurrent onboarding sessions.

5. **Authentication and Authorization**
   - **Secure Authentication**: Implement secure patient authentication mechanisms.
   - **Authorization Checks**: Ensure patients can only access their own data.

6. **Error Handling and Logging**
   - **Exception Management**: Handle unexpected errors gracefully.
   - **Audit Trails**: Log actions for compliance and troubleshooting.
   - **Monitoring**: Track system performance and health.

7. **Containerization and Deployment**
   - **Dockerization**: Containerize backend services.
   - **Cloud Deployment**: Deploy on cloud platforms like AWS, GCP, or Azure.
   - **CI/CD Pipelines**: Implement continuous integration and deployment workflows.


---

## Alignment with Developers

This PRD provides a comprehensive guide for developers implementing the AI-assisted Patient Onboarding Application:

- **Clear Project Scope**: Focused on patient onboarding, ensuring efforts are directed appropriately.
- **Detailed Functionalities**: Each core functionality is explicitly outlined, facilitating task assignment and development planning.
- **Tech Stack Specification**: Lists all technologies and libraries to be used, promoting consistency and simplifying environment setup.
- **File Structure**: Offers a clear project organization, aiding developers in navigating the codebase.
- **Documentation**: Includes in-depth explanations and conceptual outlines of key components, supporting developers in understanding and implementing critical features.
- **Security and Compliance Emphasis**: Highlights the importance of secure data handling and compliance with regulations like HIPAA.
- **Integration Points**: Describes how the frontend and backend interact, ensuring seamless communication between components.

---


## Documentation

The following sections provide detailed explanations and code snippets for key components, aiding developers in implementation.

### Interacting with Medplum FHIR Server

To interact with the Medplum FHIR server, a FHIR client must be established with proper authentication. Below is a conceptual outline:

```python
import os
from fhirclient.client import FHIRClient
from gen_ui_backend.auth import MedPlumOAuth2Auth, TokenManager

def get_fhir_client(patient_id: str | None = None) -> FHIRClient:
    client_settings = {
        "app_id": os.environ.get("MEDPLUM_CLIENT_ID"),
        "app_secret": os.environ.get("MEDPLUM_CLIENT_SECRET"),
        "api_base": os.environ.get("MEDPLUM_BASE_URL"),
        "patient_id": patient_id,
    }
    client = FHIRClient(settings=client_settings)
    token_manager = TokenManager(
        client_settings["app_id"],
        client_settings["app_secret"],
        f"{client_settings['api_base']}/oauth2/token",
    )
    oauth2_auth = MedPlumOAuth2Auth(token_manager)
    client.server.auth = oauth2_auth
    return client
```

- **Purpose**: Establishes a FHIR client with OAuth2 authentication.
- **Components**:
  - **TokenManager**: Manages access tokens.
  - **MedPlumOAuth2Auth**: Custom authentication class integrating with the token manager.

### Creating FHIR CRUD Tools for LangGraph Agent

The FHIR CRUD operations are encapsulated in a tool that the LangGraph agent can utilize:

```python
from typing import Any, Dict, Optional
from gen_ui_backend.tools.fhir.client import FhirCrudOperations
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class FHIRCrudInput(BaseModel):
    operation: str = Field(..., description="CRUD operation: 'create', 'read', 'update', 'delete'.")
    resource_type: str = Field(..., description="Type of FHIR resource.")
    data: Optional[Dict[str, Any]] = Field(None, description="Resource data or 'id' for operations.")

@tool("fhir-crud", args_schema=FHIRCrudInput, return_direct=True)
def fhir_crud(operation: str, resource_type: str, data: Optional[Dict[str, Any]]) -> Any:
    """Performs CRUD operations on FHIR resources."""
    crud_operator = FhirCrudOperations()
    # Operation handling logic...
    return result
```

- **Purpose**: Defines a tool for CRUD operations on FHIR resources.
- **Components**:
  - **FHIRCrudInput**: Input schema using Pydantic for validation.
  - **`fhir_crud` Function**: Executes the specified CRUD operation.

### Authorization with Medplum FHIR Server

Handling secure authentication with the Medplum FHIR server is crucial:

```python
import base64
import os
import time
import requests
from fhirclient.auth import FHIROAuth2Auth

class TokenManager:
    def __init__(self, client_id: str, client_secret: str, token_url: str):
        # Initialization logic...

    def get_token(self) -> str:
        # Token retrieval logic...

class MedPlumOAuth2Auth(FHIROAuth2Auth):
    def __init__(self, token_manager: TokenManager):
        # Initialization logic...

    def signed_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Returns updated HTTP request headers using the access token."""
        access_token = self.token_manager.get_token()
        headers["Authorization"] = f"Bearer {access_token}"
        return headers
```

- **Purpose**: Manages OAuth2 authentication flow.
- **Components**:
  - **TokenManager**: Fetches and refreshes tokens as needed.
  - **MedPlumOAuth2Auth**: Overrides methods to insert authentication headers.

### Creating a LangGraph Agent with Async Tools, and Persistance (Postgres)

The agent processes user inputs, decides whether to use tools, and generates responses:

```python
import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, TypedDict
from uuid import UUID

from bionic.clients import http
from bionic.clients.auth import generate_headers
from bionic.clients.fhir import FhirServiceClient, ObservationFilter
from bionic.clients.health_app import HealthAppServiceClient
from bionic.clients.health_app.schemas import PatientDetails
from django.conf import settings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from bionic_onboarding_bot.llm.llm import get_chat_llm
from bionic_onboarding_bot.llm.tools import fetch_specific_observations

if TYPE_CHECKING:
    from base import settings  # type: ignore # noqa

logger = logging.getLogger(__name__)


POSTGRES_DB_URI = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    patient_details: dict[str, Any]
    patient_codes: list[dict[str, str]]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI health assistant designed to support physicians by answering their questions about patients using the tools at your disposal. Your primary goal is to provide accurate, relevant information by leveraging the available tools and data sources to address the physician's specific inquiries.

Key guidelines:
1. Focus on using your available tools to answer the physician's questions directly and efficiently.
2. Maintain a professional, concise, and informative tone.
3. Provide evidence-based information when possible, using your tools to access relevant medical data.
4. When asked about diagnoses, tests, or treatments, use your tools to retrieve and present the most up-to-date and relevant information.
5. If a tool doesn't exist for a specific query, acknowledge this limitation and provide the best possible answer based on your general knowledge.
6. Prioritize patient safety by highlighting any potential drug interactions or contraindications that your tools can identify.
7. Use tools to check for any red flags or urgent concerns in the patient's data that require immediate attention.
8. Respect patient privacy and adhere to medical ethics at all times.

When responding:
- Start by identifying which tools are most appropriate to answer the physician's question.
- Use the selected tools to gather relevant information.
- Present the information obtained from the tools in a clear, structured manner.
- If multiple tools are used, synthesize the information coherently.
- Clearly state when you're using a specific tool and when you're providing general knowledge.
- If a tool doesn't provide sufficient information, explain this and offer alternative approaches or suggest additional data that might be needed.

Remember, your primary function is to assist the physician by utilizing your tools effectively. Always encourage the physician to use their clinical judgment in conjunction with the information you provide.

If you lack a specific tool or sufficient information to provide a comprehensive answer, clearly state this limitation and suggest what additional resources or data might be helpful.""",
        ),
        (
            "system",
            "Patient details: {patient_details}\nUse this information to inform your responses, but only reference it explicitly if directly relevant to the physician's query.",
        ),
        (
            "system",
            "Patient codes: {patient_codes}\nThese are the codes for the patient's observations. Use this to guide your inputs when using the fetch_specific_observations tool.",
        ),
        ("placeholder", "{messages}"),
    ]
)

graph_builder = StateGraph(State)


@tool
async def get_patient_notes(config: RunnableConfig) -> str:
    """
    Retrieve the patient's notes
    """
    patient_id = config.get("configurable", {}).get("patient_id")
    if patient_id is None:
        raise ValueError("Patient ID is required to update the action plan")

    logger.info(f"Fetching notes for patient with ID: {patient_id}")
    url = f"{settings.BIONIC_API_BASE_URL}/api/v1/patients/{patient_id}/notes"
    response = await http.AsyncClient().get(url, headers=generate_headers("chat-bot-service"))
    response.raise_for_status()
    return response.json()


@tool
async def get_patient_health_targets(config: RunnableConfig) -> str:
    """
    Retrieve the patient's health targets (key health metrics, that are a n important indicator of a specific patient's health/action plan)
    """
    patient_id = config.get("configurable", {}).get("patient_id")
    if patient_id is None:
        raise ValueError("Patient ID is required to update the action plan")

    logger.info(f"Fetching health-targets for patient with ID: {patient_id}")
    url = f"{settings.BIONIC_API_BASE_URL}/api/v1/patients/{patient_id}/health-targets"
    response = await http.AsyncClient().get(url, headers=generate_headers("chat-bot-service"))
    response.raise_for_status()
    return response.json()


@tool
async def get_patient_codes_list(config: RunnableConfig) -> list[dict[str, str]]:
    """
    Retrieve all unique codes associated with a patient's observations, including the observation name.
    Args:
    """

    patient_id = config.get("configurable", {}).get("patient_id")
    if patient_id is None:
        raise ValueError("Patient ID is required to fetch patient details")

    logger.info(f"Fetching patient details for patient with ID: {patient_id}")
    try:
        client = FhirServiceClient[http.AsyncClient](
            "chat-bot-service", settings.BIONIC_API_BASE_URL, http_client=http.AsyncClient()
        )

        observations = await client.aget_observations(
            filters=ObservationFilter(subject=UUID(patient_id), latest_only=True)
        )

        unique_codes = set()
        for observation in observations.items:
            name = observation.name
            code = ""
            system = ""
            for coding in observation.code.coding:
                code = coding.code
                system = coding.system
            unique_codes.add((system, code, name))

        code_list = [{"system": system, "code": code, "name": name} for system, code, name in unique_codes]

        return code_list

    except ValueError:
        return []


@tool
async def fetch_patient_details(config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieve comprehensive details about a patient
    """
    patient_id = config.get("configurable", {}).get("patient_id")
    if patient_id is None:
        raise ValueError("Patient ID is required to fetch patient details")

    logger.info(f"Fetching patient details for patient with ID: {patient_id}")
    try:
        client = HealthAppServiceClient[http.AsyncClient](
            "chat-bot-service",
            settings.BIONIC_API_BASE_URL,
            http_client=http.AsyncClient(),
        )
        patient_details: PatientDetails = await client.aget_patient_details(patient_id)

        details_dict = json.loads(patient_details.json(exclude_none=True))

    except Exception as e:
        logger.exception(f"Failed to fetch patient details from tool run: {e}")
        raise ValueError("Failed to fetch patient details") from e

    return details_dict


async def chatbot(state: State) -> State:
    return {"messages": [await onboarding_llm.ainvoke(state)]}  # type: ignore


async def fetch_patient_info(state: State, config: RunnableConfig) -> dict[str, Any]:
    patient = await fetch_patient_details.ainvoke({}, config=config)  # type: ignore
    patient_codes = await get_patient_codes_list.ainvoke({}, config=config)  # type: ignore
    return {"patient_details": patient, "patient_codes": patient_codes}


llm = get_chat_llm()
tools = [fetch_specific_observations, get_patient_notes, get_patient_health_targets]
onboarding_llm = prompt | llm.bind_tools(tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("fetch_patient_info", fetch_patient_info)  # type: ignore
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("fetch_patient_info", "chatbot")
graph_builder.set_entry_point("fetch_patient_info")
graph_builder.set_finish_point("chatbot")


class ClinicalAiAssistantService:
    def __init__(self, thread_id: str, patient_id: str) -> None:
        self._config = {"configurable": {"thread_id": thread_id, "patient_id": patient_id}}

    async def aget_response(self, message: str) -> str:
        async with AsyncPostgresSaver.from_conn_string(conn_string=POSTGRES_DB_URI) as checkpointer:
            await checkpointer.setup()

            graph = graph_builder.compile(checkpointer=checkpointer)
            res = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, self._config)  # type: ignore

            return res["messages"][-1].content

```

- **Purpose**: Defines the computational graph for the LangGraph agent.
- **Components**:
  - **`invoke_model` Function**: Calls the language model and processes the output.
  - **`invoke_tools_or_return` Function**: Decides whether to invoke tools or return the result.
  - **`invoke_tools` Function**: Executes the required tools based on the model's output.
  - **`create_graph` Function**: Compiles the state graph.

### Sending Messages to the LangGraph Agent

The frontend communicates with the backend agent through defined endpoints:

```typescript
"use server";

import { RemoteRunnable } from "@langchain/core/runnables/remote";
import { createStreamableUI, createStreamableValue } from "ai/rsc";

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

  // Event handlers for streaming and tool responses...

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

// Expose the agent function without setResponse
export const EndpointsContext = exposeEndpoints({ agent: (inputs) => agent(inputs) });
```

## Documentation for making a postgres checkpointer for langgraph

```python
#         except OnboardingThread.DoesNotExist:
#             raise ValueError(f"OnboardingThread with id {self.thread_id} does not exist") from None
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class PostgresCheckpointer(AsyncPostgresSaver, BaseCheckpointSaver):
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await super().aget_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for checkpoint in super().alist(config, filter=filter, before=before, limit=limit):
            yield checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        await super().aput_writes(config, writes, task_id)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self._run_sync(self.aget_tuple(config))

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        return iter(list(self._run_sync(self.alist(config, filter=filter, before=before, limit=limit))))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._run_sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        self._run_sync(self.aput_writes(config, writes, task_id))

    def _run_sync(self, coroutine: Any) -> Any:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(coroutine)

```

---

## Documentation for creating langchain/langgraph tools

```python

import datetime
import json
import logging
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from bionic.clients import http
from bionic.clients.fhir import (
    DiagnosticReportFilter,
    FhirServiceClient,
    ObservationFilter,
    ObservationStatisticsFilter,
    TargetFilter,
)
from bionic.clients.fhir.schemas import ObservationIn
from bionic.clients.health_app import HealthAppServiceClient
from bionic.clients.health_app.schemas import (
    Allergen,
    Allergy,
    Diagnosis,
    Domain,
    Drug,
    FamilyMedicalHistory,
    Medication,
    PatientDetails,
    PatientHistoryField,
    PatientPatch,
    Relationship,
)
from django.conf import settings
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PatchPatientInput(BaseModel):
    patient_id: UUID
    patch_data: dict[str, Any]


@tool
def update_medical_history(medical_history: str, state: Annotated[dict[str, Any], InjectedState]) -> PatientPatch:
    """Update the medical history of the patient."""
    patient_patch = state["patient_patch"]
    if patient_patch.medical_history is None:
        patient_patch.medical_history = PatientHistoryField(raw=medical_history)
    else:
        patient_patch.medical_history.raw += f"\n{medical_history}"  # type: ignore
    return patient_patch


@tool
def update_social_history(
    social_history: str, patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")]
) -> PatientPatch:
    """Update the social history of the patient."""
    if patient_patch.social_history is None:
        patient_patch.social_history = PatientHistoryField(raw=social_history)
    else:
        patient_patch.social_history.raw += f"\n{social_history}"  # type: ignore
    return patient_patch


@tool
def add_family_medical_history(
    domain: Domain,
    relationship: Relationship,
    issue: str,
    patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")],
    age_of_onset: int | None = None,
    notes: str = "",
) -> PatientPatch:
    """Add a family medical history entry for the patient."""
    new_entry = FamilyMedicalHistory(
        domain=domain, relationship=relationship, issue=issue, age_of_onset=age_of_onset, notes=notes
    )
    patient_patch.family_medical_history.append(new_entry)
    return patient_patch


@tool
def update_exercise(
    exercise: str, patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")]
) -> PatientPatch:
    """Update the exercise information of the patient."""
    if patient_patch.exercise is None:
        patient_patch.exercise = PatientHistoryField(raw=exercise)
    else:
        patient_patch.exercise.raw += f"\n{exercise}"  # type: ignore
    return patient_patch


@tool
def update_diet_info(diet: str, state: Annotated[dict[str, Any], InjectedState]) -> PatientPatch:
    """Update the information about the patients diet of the patient."""
    patient_patch = state["patient_patch"]
    if patient_patch.diet is None:
        patient_patch.diet = PatientHistoryField(raw=diet)
    else:
        patient_patch.diet.raw += f"\n{diet}"  # type: ignore
    return patient_patch


@tool
def update_sleep(sleep: str, patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")]) -> PatientPatch:
    """Update the sleep information of the patient."""
    if patient_patch.sleep is None:
        patient_patch.sleep = PatientHistoryField(raw=sleep)
    else:
        patient_patch.sleep.raw += f"\n{sleep}"  # type: ignore
    return patient_patch


@tool
def update_stress(stress: str, patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")]) -> PatientPatch:
    """Update the stress information of the patient."""
    if patient_patch.stress is None:
        patient_patch.stress = PatientHistoryField(raw=stress)
    else:
        patient_patch.stress.raw += f"\n{stress}"  # type: ignore
    return patient_patch


@tool
def add_medication(
    description: str,
    name: str,
    dosage: str,
    frequency: str,
    patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")],
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> PatientPatch:
    """Add a medication entry for the patient."""
    new_medication = Medication(
        drug=Drug(description=description, name=name),
        dosage=dosage,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )
    patient_patch.medications.append(new_medication)
    return patient_patch


@tool
def add_allergy(
    allergen_name: str,
    reaction: str,
    severity: str,
    patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")],
    drug_description: str | None = None,
    drug_name: str | None = None,
) -> PatientPatch:
    """Add an allergy entry for the patient."""
    drug = None if drug_name is None else Drug(description=drug_description or "", name=drug_name)
    new_allergy = Allergy(allergen=Allergen(name=allergen_name, drug=drug), reaction=reaction, severity=severity)
    patient_patch.allergies.append(new_allergy)
    return patient_patch


@tool
def add_diagnosis(
    code: str,
    description: str,
    patient_patch: Annotated[PatientPatch, InjectedState("patient_patch")],
    date_diagnosed: datetime.datetime | None = None,
) -> PatientPatch:
    """Add a diagnosis entry for the patient."""
    new_diagnosis = Diagnosis(code=code, description=description, date_diagnosed=date_diagnosed)
    patient_patch.diagnoses.append(new_diagnosis)
    return patient_patch


@tool
def fetch_current_question() -> str:
    """
    Retrieve the current question to be asked in the onboarding flow.
    """
    config = ensure_config()  # Fetch from the context
    configuration = config.get("configurable", {})
    current_question = configuration.get("unanswered_questions", None).pop(0)

    if current_question is None:
        raise ValueError("No current_question configured.")

    return current_question


@tool
def fetch_next_question() -> str:
    """
    Retrieve the current question to be asked in the onboarding flow.
    """
    config = ensure_config()  # Fetch from the context
    configuration = config.get("configurable", {})
    next_question = configuration.get("unanswered_questions", None)[0]

    if next_question is None:
        raise ValueError("No current_question configured.")

    return next_question


@tool
def fetch_patient_details(patient_id: str) -> str:
    """
    Retrieve comprehensive details about a patient using their ID.
    Returns:
    Dict[str, Any]: A dictionary containing all available details about the patient,
    including personal information, medical history, medications, diagnoses, and allergies.
    """
    try:
        client = HealthAppServiceClient[http.SyncClient](
            "chat-bot-service-patch-patient-details", settings.BIONIC_API_BASE_URL
        )
        patient_details: PatientDetails = client.get_patient_details(patient_id)

        details_dict = patient_details.json()

    except Exception as e:
        logger.exception(f"Failed to fetch patient details from tool run: {e}")
        return ""

    return details_dict


@tool
async def fetch_specific_observations(
    code_strings: List[str],
    config: RunnableConfig,
    limit: int = 100,
    offset: int = 0,
    code_text: Optional[str] = None,
    category: Optional[List[str]] = None,
    effective_datetime: Optional[List[str]] = None,
    latest_only: bool = False,
) -> List[dict[str, Any]]:
    """
    Retrieve specific observations for a patient based on various filters.

    Args:
    code_strings (List[str]): A list of strings formatted as "<system>|<code>".
    config (RunnableConfig): Configuration object containing patient_id.
    limit (int, optional): Maximum number of observations to return. Default is 100.
    offset (int, optional): Number of observations to skip. Default is 0.
    code_text (str, optional): Text to search in CodeableConcept.text and Coding.display.
    category (List[str], optional): List of categories to filter by.
    effective_datetime (List[str], optional): List of datetime strings to filter by (e.g., ["gt2024-01-01"])
    latest_only (bool, optional): Get the latest observation only. Default is False.
    """
    patient_id = config.get("configurable", {}).get("patient_id")
    if patient_id is None:
        raise ValueError("Patient ID is required to fetch patient details")

    logger.info(f"Fetching observations {code_strings} for patient with ID: {patient_id}")
    client = FhirServiceClient[http.AsyncClient](
        "chat-bot-service", settings.BIONIC_API_BASE_URL, http_client=http.AsyncClient()
    )

    codes = [code_string.split("|") for code_string in code_strings]
    code_list = [code for system, code in codes]
    try:
        filter_params = ObservationFilter(
            limit=limit,
            offset=offset,
            code=[",".join(code_list)],
            code_text=code_text,  # type: ignore
            category=category,
            subject=UUID(patient_id),
            effective_datetime=effective_datetime,
            latest_only=latest_only,
        )

        observations = await client.aget_observations(filters=filter_params)

        matching_observations = [observation.dict() for observation in observations.items]
    except Exception as e:
        logger.exception(f"Failed to fetch specific observations: {e}")
        return [{}]

    return matching_observations

@tool
def patch_patient(patient_id: str, patient_patch: PatientPatch) -> str:
    """
    patch patient details using the HealthAppServiceClient.

    Args:
        input_data (PatchPatientInput): An object containing:
            - patient_id (str): The ID of the patient to update.
            - patient_patch (PatientPatch): A pydantic model for the patch

    Returns:
        str: A message indicating the success of the operation and a summary of the updated fields,
             or an error message if the operation failed.
    """
    client = HealthAppServiceClient[http.SyncClient](
        "chat-bot-service-patch-patient-details", settings.BIONIC_API_BASE_URL
    )

    try:
        client.patch_patient(patient_id, patient_patch)
    except Exception as e:
        error_msg = f"Failed to update patient information: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    success_msg = f"Successfully patched patient details. for patient with id: {patient_id}"
    logger.info(success_msg)
    return success_msg

```

## Alignment with Developers

This document is structured to provide clear guidance to developers implementing the project:

- **Detailed Functionalities**: Each core functionality is broken down into specific requirements.
- **Tech Stack Specification**: Lists all technologies and libraries to be used, ensuring consistency across development.
- **File Structure**: Provides a simplified yet comprehensive file structure to keep the project organized.
- **Code Documentation**: Includes code snippets and explanations for critical components, aiding in understanding and implementation.
- **Integration Points**: Highlights how different parts of the system interact, such as the frontend invoking backend endpoints.
- **Security and Compliance**: Emphasizes the importance of adhering to regulatory requirements like HIPAA.

---
