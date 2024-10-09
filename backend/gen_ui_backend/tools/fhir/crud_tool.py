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
