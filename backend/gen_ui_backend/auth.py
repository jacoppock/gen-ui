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
            expires_in = token_response.get("expires_in", 3600)  # Default to 1 hour if not provided
            self.expiration_time = time.time() + expires_in - 60  # Subtract 60 seconds as a buffer
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
    def __init__(self, token_manager: TokenManager, state: Optional[str] = None) -> None:
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
