import os
import secrets

from fastapi import HTTPException, Request
from pydantic import SecretStr

from nonslop.web.schemas import WebSettings


def environment_settings() -> WebSettings:
    return WebSettings(
        api_key=SecretStr(os.environ["APP_API_KEY"]),
        password=SecretStr(os.environ["APP_PASSWORD"]),
        session_secret=SecretStr(os.environ["SESSION_SECRET"]),
    )


def credentials_match(settings: WebSettings, api_key: str, password: str) -> bool:
    key_matches = secrets.compare_digest(
        api_key.encode(), settings.api_key.get_secret_value().encode()
    )
    password_matches = secrets.compare_digest(
        password.encode(), settings.password.get_secret_value().encode()
    )
    return key_matches and password_matches


def require_api(request: Request, settings: WebSettings) -> None:
    if not credentials_match(
        settings,
        request.headers.get("x-api-key", ""),
        request.headers.get("x-password", ""),
    ):
        raise HTTPException(401, "API key and password required.")


def csrf_token(request: Request) -> str:
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return str(request.session["csrf"])


def require_csrf(request: Request, token: str) -> None:
    expected = request.session.get("csrf")
    if not isinstance(expected, str) or not secrets.compare_digest(
        token.encode(), expected.encode()
    ):
        raise HTTPException(403, "Your form expired. Reload the page and try again.")
