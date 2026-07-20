from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.security.cli_secrets import RuntimeSecretReferenceError

from vector_mcp import doctor


def test_doctor_resolves_only_runtime_secret_references(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "setting",
        lambda name, default="": (
            "env://BACKEND_RUNTIME_SECRET" if name == "DB_PASSWORD_REF" else default
        ),
    )
    monkeypatch.setattr(
        doctor,
        "resolve_runtime_secret_reference",
        lambda _reference: "runtime value",
    )

    assert doctor._credential_available("DB_PASSWORD_REF") is True
    assert doctor._credential_available("DB_USERNAME_REF") is False


def test_doctor_sanitizes_unavailable_secret_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "setting",
        lambda _name, default="": "env://UNAVAILABLE_RUNTIME_SECRET",
    )

    def unavailable(_reference: str) -> str:
        raise RuntimeSecretReferenceError("runtime secret reference is unavailable")

    monkeypatch.setattr(doctor, "resolve_runtime_secret_reference", unavailable)

    assert doctor._credential_available("DB_PASSWORD_REF") is False


def test_doctor_passes_named_and_referenced_tls_selectors(monkeypatch) -> None:
    values = {
        "POSTGRES_TLS_PROFILE": "managed-profile",
        "POSTGRES_TLS_PROFILE_REF": "secret://runtime/tls-profile",
    }
    monkeypatch.setattr(
        doctor, "setting", lambda name, default=None: values.get(name, default)
    )
    observed = {}

    def resolve(service, **kwargs):
        observed["service"] = service
        observed.update(kwargs)
        return SimpleNamespace(
            verify_enabled=True,
            configured=True,
            ca_bundle_path=None,
            ca_directory=None,
            client_cert_path=None,
        )

    monkeypatch.setattr(doctor, "resolve_configured_tls_profile", resolve)

    valid, summary = doctor._tls_status("POSTGRES")

    assert valid is True
    assert summary["verify_enabled"] is True
    assert observed == {
        "service": "postgres",
        "profile_name": "managed-profile",
        "profile_ref": "secret://runtime/tls-profile",
    }
