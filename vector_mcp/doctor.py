"""Static vector backend readiness check without importing backend SDKs."""

from __future__ import annotations

import json

from agent_utilities.core.config import setting
from agent_utilities.core.transport_security import (
    TransportSecurityError,
    resolve_configured_tls_profile,
)
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceError,
    resolve_runtime_secret_reference,
)

from vector_mcp.backend_policy import backend_status


def _credential_available(reference_name: str) -> bool:
    reference = str(setting(reference_name, "") or "")
    if not reference:
        return False
    try:
        return bool(resolve_runtime_secret_reference(reference))
    except RuntimeSecretReferenceError:
        return False


def _tls_status(service: str) -> tuple[bool, dict[str, bool]]:
    prefix = service.upper()
    try:
        tls = resolve_configured_tls_profile(
            service.casefold(),
            profile_name=setting(f"{prefix}_TLS_PROFILE", None),
            profile_ref=setting(f"{prefix}_TLS_PROFILE_REF", None),
        )
        return True, {
            "verify_enabled": tls.verify_enabled,
            "profile_configured": tls.configured,
            "custom_trust_configured": bool(tls.ca_bundle_path or tls.ca_directory),
            "mutual_tls_configured": bool(tls.client_cert_path),
        }
    except (OSError, RuntimeError, TransportSecurityError, ValueError):
        return False, {
            "verify_enabled": False,
            "profile_configured": False,
            "custom_trust_configured": False,
            "mutual_tls_configured": False,
        }


def main() -> int:
    status = dict(
        backend_status(setting("DATABASE_TYPE", "epistemic_graph") or "epistemic_graph")
    )
    status["configured"] = bool(status["available"])
    if status["backend"] == "qdrant" and status["available"]:
        endpoint_configured = bool(setting("DB_HOST", None))
        credential_configured = _credential_available("QDRANT_API_KEY_REF")
        tls_valid, tls_summary = _tls_status("QDRANT")
        status["configured"] = bool(
            endpoint_configured and credential_configured and tls_valid
        )
        status["available"] = bool(status["available"] and status["configured"])
        status["reason"] = (
            "available" if status["available"] else "backend configuration incomplete"
        )
        status["connection"] = {
            "endpoint_configured": endpoint_configured,
            "credential_configured": credential_configured,
            "private_host_allowlist_configured": bool(
                setting("QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS", [])
            ),
            "tls": tls_summary,
        }
    elif status["backend"] == "mongodb" and status["available"]:
        uri_configured = _credential_available("MONGODB_URI_REF")
        endpoint_configured = bool(uri_configured)
        credentials_configured = bool(uri_configured)
        tls_valid, tls_summary = _tls_status("MONGODB")
        status["configured"] = bool(
            endpoint_configured and credentials_configured and tls_valid
        )
        status["available"] = bool(status["available"] and status["configured"])
        status["reason"] = (
            "available" if status["available"] else "backend configuration incomplete"
        )
        status["connection"] = {
            "endpoint_configured": endpoint_configured,
            "credentials_configured": credentials_configured,
            "tls": tls_summary,
        }
    elif status["backend"] == "postgres" and status["available"]:
        endpoint_configured = bool(setting("DB_HOST", None) and setting("DBNAME", None))
        credentials_configured = bool(
            _credential_available("DB_USERNAME_REF")
            and _credential_available("DB_PASSWORD_REF")
        )
        tls_valid, tls_summary = _tls_status("POSTGRES")
        status["configured"] = bool(
            endpoint_configured and credentials_configured and tls_valid
        )
        status["available"] = bool(status["available"] and status["configured"])
        status["reason"] = (
            "available" if status["available"] else "backend configuration incomplete"
        )
        status["connection"] = {
            "endpoint_configured": endpoint_configured,
            "credentials_configured": credentials_configured,
            "tls": tls_summary,
        }
    print(json.dumps(status, sort_keys=True))
    return 0 if status["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
