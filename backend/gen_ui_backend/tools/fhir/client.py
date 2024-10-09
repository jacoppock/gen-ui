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
