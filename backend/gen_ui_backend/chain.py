import os
from typing import List, Optional, TypedDict

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

    tools = [github_repo, invoice_parser, weather_data]
    model_with_tools = model.bind_tools(tools)
    chain = initial_prompt | model_with_tools
    result = chain.invoke({"input": state["input"]}, config)

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
        "github-repo": github_repo,
        "invoice-parser": invoice_parser,
        "weather-data": weather_data,
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
