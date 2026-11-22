import os

import modal
from dotenv import load_dotenv


def configured_access() -> modal.Secret:
    required_keys = ["APP_API_KEY", "APP_PASSWORD", "SESSION_SECRET"]
    if modal.is_local():
        load_dotenv(
            dotenv_path=os.environ.get("NONSLOP_ENV_FILE", ".env"), override=False
        )
        values = {name: os.environ.get(name, "") for name in required_keys}
        configured = [bool(value.strip()) for value in values.values()]
        if any(configured) and not all(configured):
            raise ValueError(
                "APP_API_KEY, APP_PASSWORD, and SESSION_SECRET must be configured "
                "together; otherwise leave all three unset to use nonslop-access"
            )
        if all(configured):
            return modal.Secret.from_dict(values)
    return modal.Secret.from_name("nonslop-access", required_keys=required_keys)
