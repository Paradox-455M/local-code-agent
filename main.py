from __future__ import annotations

import typer
from rich.console import Console

from agent.cli import app
from core.exceptions import LocalCodeAgentError

console = Console()


if __name__ == "__main__":
    try:
        app()
    except LocalCodeAgentError as e:
        console.print(f"[red]Agent Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected Error: {e}[/red]")
        raise typer.Exit(code=1)

