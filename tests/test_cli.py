from pathlib import Path

import pytest

from nonslop.runtime.cli import build_parser, deployment_environment


def test_credentials_follow_flags_environment_then_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "access.env"
    env_file.write_text(
        "APP_API_KEY=file-key\nAPP_PASSWORD=file-password\nSESSION_SECRET=file-session\n"
    )
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.setenv("APP_PASSWORD", "environment-password")
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    parser = build_parser()
    arguments = parser.parse_args(
        ["deploy", "--app", "cpu", "--env-file", str(env_file), "--api-key", "flag-key"]
    )
    environment = deployment_environment(arguments, parser)
    assert environment["APP_API_KEY"] == "flag-key"
    assert environment["APP_PASSWORD"] == "environment-password"
    assert environment["SESSION_SECRET"] == "file-session"
    assert environment["NONSLOP_ENV_FILE"] == str(env_file)


def test_incomplete_credentials_fail_without_echoing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("APP_API_KEY", "APP_PASSWORD", "SESSION_SECRET"):
        monkeypatch.delenv(name, raising=False)
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "deploy",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--api-key",
            "hidden-key",
        ]
    )
    with pytest.raises(SystemExit, match="2"):
        deployment_environment(arguments, parser)
    assert "hidden-key" not in capsys.readouterr().err
