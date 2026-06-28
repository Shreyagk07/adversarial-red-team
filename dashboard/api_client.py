"""Thin HTTP client for the FastAPI backend.

The dashboard talks to the backend only through this client, so all networking,
error handling, and URL construction live in one place. Every method raises
:class:`ApiError` with a readable message on failure, which the UI surfaces as
a friendly error instead of a traceback.
"""

from __future__ import annotations

from typing import Any

import requests


class ApiError(Exception):
    """Raised when a backend call fails (network error or non-2xx response)."""


class ApiClient:
    """Client for the red-team backend API."""

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        # Trailing slash normalized away so we can join paths predictably.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- low-level helpers --------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = requests.request(
                method, self._url(path), timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach backend at {self.base_url}: {exc}") from exc

        if not resp.ok:
            # Surface FastAPI's {"detail": ...} message when present.
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:  # noqa: BLE001 - body may not be JSON
                detail = resp.text
            raise ApiError(f"{resp.status_code}: {detail or resp.reason}")

        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # --- meta ---------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_categories(self) -> list[dict[str, Any]]:
        return self._request("GET", "/categories")

    # --- targets ------------------------------------------------------------
    def list_targets(self) -> list[dict[str, Any]]:
        return self._request("GET", "/targets")

    def create_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/targets", json=payload)

    # --- runs ---------------------------------------------------------------
    def launch_evaluation(
        self, target_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request("POST", f"/targets/{target_id}/evaluate", json=body)

    def list_runs(self, target_id: str | None = None) -> list[dict[str, Any]]:
        params = {"target_id": target_id} if target_id else None
        return self._request("GET", "/runs", params=params)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}")

    def get_run_report(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}/report")

    def compare(self, before: str, after: str) -> dict[str, Any]:
        return self._request(
            "GET", "/compare", params={"before": before, "after": after}
        )
