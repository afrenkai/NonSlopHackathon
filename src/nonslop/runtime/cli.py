import argparse
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nonslop")
    commands = parser.add_subparsers(dest="command", required=True)
    deploy = commands.add_parser("deploy", help="Deploy the Modal applications")
    deploy.add_argument(
        "--app",
        choices=("cpu", "video", "world", "inference", "all"),
        default="all",
    )
    deploy.add_argument("--env-file", type=Path, default=Path(".env"))
    deploy.add_argument("--api-key", help="APP_API_KEY override")
    deploy.add_argument("--password", help="APP_PASSWORD override")
    deploy.add_argument("--session-secret", help="SESSION_SECRET override")
    return parser


def deployment_environment(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, str]:
    environment = {
        name: value
        for name, value in dotenv_values(arguments.env_file).items()
        if value is not None
    }
    environment.update(os.environ)
    environment["NONSLOP_ENV_FILE"] = str(arguments.env_file.resolve())
    overrides = {
        "APP_API_KEY": arguments.api_key,
        "APP_PASSWORD": arguments.password,
        "SESSION_SECRET": arguments.session_secret,
    }
    for name, value in overrides.items():
        if value is not None:
            if not value.strip():
                parser.error(f"{name} must not be blank")
            environment[name] = value

    configured = [bool(environment.get(name, "").strip()) for name in overrides]
    if any(configured) and not all(configured):
        parser.error(
            "APP_API_KEY, APP_PASSWORD, and SESSION_SECRET must be configured together "
            "through CLI flags, environment variables, or the env file"
        )
    return environment


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    environment = deployment_environment(arguments, parser)
    modules = {
        "video": "nonslop.runtime.video_app",
        "world": "nonslop.runtime.world_app",
        "inference": "nonslop.runtime.inference_app",
        "cpu": "nonslop.runtime.cpu_app",
    }
    selected_modules = (
        list(modules.values()) if arguments.app == "all" else [modules[arguments.app]]
    )
    for module in selected_modules:
        try:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--locked",
                    "modal",
                    "deploy",
                    "--strategy",
                    "recreate",
                    "-m",
                    module,
                ],
                check=True,
                env=environment,
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
