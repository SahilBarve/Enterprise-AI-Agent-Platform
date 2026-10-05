# PowerShell helper script for development workflow
param(
    [Parameter(Position=0)]
    [ValidateSet("up", "down", "restart", "logs", "test", "lint", "format", "typecheck")]
    [string]$Command = "help"
)

$PythonVenv = ".\.venv\Scripts\python.exe"
$Ruff = ".\.venv\Scripts\ruff.exe"
$Mypy = ".\.venv\Scripts\mypy.exe"
$Pytest = ".\.venv\Scripts\pytest.exe"

switch ($Command) {
    "up" {
        docker compose up -d
    }
    "down" {
        docker compose down
    }
    "restart" {
        docker compose restart
    }
    "logs" {
        docker compose logs -f
    }
    "test" {
        & $Pytest tests/unit -v --cov=libs --cov=services
    }
    "lint" {
        & $Ruff check .
        & $Ruff format --check .
    }
    "format" {
        & $Ruff format .
        & $Ruff check --fix .
    }
    "typecheck" {
        & $Mypy libs services
    }
    Default {
        Write-Host "Usage: .\scripts\dev.ps1 [up|down|restart|logs|test|lint|format|typecheck]"
    }
}
