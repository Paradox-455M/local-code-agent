"""Session management commands for the Local Code Agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from memory.conversation import get_conversation_manager, ConversationManager

console = Console()


def list_sessions(detailed: bool = False) -> None:
    """
    List all available conversation sessions.
    
    Args:
        detailed: If True, show detailed information for each session.
    """
    conv_manager = get_conversation_manager()
    sessions = conv_manager.list_sessions()
    
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        console.print("[dim]Tip: Start a new session with --session <name>[/dim]")
        return
    
    if not detailed:
        console.print(f"[bold]Available sessions ({len(sessions)}):[/bold]\n")
        for sess_id in sessions:
            # Load session to get turn count
            if sess_id not in conv_manager.sessions:
                conv_manager._load_session(sess_id)
            
            session = conv_manager.sessions.get(sess_id)
            if session:
                turn_count = len(session.turns)
                last_turn = session.turns[-1] if session.turns else None
                timestamp = last_turn.timestamp if last_turn else "unknown"
                
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    time_str = timestamp
                
                console.print(f"  • [cyan]{sess_id}[/cyan] - {turn_count} turns - Last: {time_str}")
            else:
                console.print(f"  • [cyan]{sess_id}[/cyan]")
    else:
        # Detailed view with table
        table = Table(title="Conversation Sessions", show_header=True, header_style="bold magenta")
        table.add_column("Session ID", style="cyan")
        table.add_column("Turns", justify="right", style="green")
        table.add_column("Files", justify="right", style="yellow")
        table.add_column("Last Task Type", style="blue")
        table.add_column("Last Updated", style="dim")
        
        for sess_id in sessions:
            if sess_id not in conv_manager.sessions:
                conv_manager._load_session(sess_id)
            
            session = conv_manager.sessions.get(sess_id)
            if session:
                turn_count = str(len(session.turns))
                file_count = str(len(session.context_files))
                task_type = session.last_task_type or "-"
                
                last_turn = session.turns[-1] if session.turns else None
                timestamp = last_turn.timestamp if last_turn else "unknown"
                
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    time_str = timestamp
                
                table.add_row(sess_id, turn_count, file_count, task_type, time_str)
            else:
                table.add_row(sess_id, "-", "-", "-", "-")
        
        console.print(table)


def show_session(session_id: str, full: bool = False) -> None:
    """
    Show details of a specific session.
    
    Args:
        session_id: The session ID to show.
        full: If True, show full task and response content.
    """
    conv_manager = get_conversation_manager()
    
    # Load session if not already loaded
    if session_id not in conv_manager.sessions:
        if not conv_manager._load_session(session_id):
            console.print(f"[red]Error: Session '{session_id}' not found.[/red]")
            console.print(f"[dim]Tip: Use 'local-code-agent sessions list' to see available sessions.[/dim]")
            return
    
    session = conv_manager.sessions[session_id]
    
    # Display session header
    console.print(f"\n[bold cyan]Session: {session_id}[/bold cyan]")
    console.print(f"[dim]Turns: {len(session.turns)} | Files: {len(session.context_files)}[/dim]\n")
    
    if not session.turns:
        console.print("[dim]No turns in this session yet.[/dim]")
        return
    
    # Display turns
    for i, turn in enumerate(session.turns, 1):
        try:
            dt = datetime.fromisoformat(turn.timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = turn.timestamp
        
        # Turn header
        console.print(f"[bold]Turn {i}[/bold] [dim]({time_str})[/dim]")
        console.print(f"[blue]Type:[/blue] {turn.task_type}")
        
        # Task content
        if full:
            console.print(Panel(turn.task, title="Task", border_style="green", padding=(0, 1)))
        else:
            # Truncate long tasks
            task_display = turn.task if len(turn.task) <= 100 else turn.task[:97] + "..."
            console.print(f"[green]Task:[/green] {task_display}")
        
        # Files referenced
        if turn.files_referenced:
            files_str = ", ".join(turn.files_referenced[:3])
            if len(turn.files_referenced) > 3:
                files_str += f" (+{len(turn.files_referenced) - 3} more)"
            console.print(f"[yellow]Files:[/yellow] {files_str}")
        
        # Response
        if turn.response:
            if full:
                console.print(Panel(turn.response, title="Response", border_style="blue", padding=(0, 1)))
            else:
                response_display = turn.response if len(turn.response) <= 150 else turn.response[:147] + "..."
                console.print(f"[cyan]Response:[/cyan] {response_display}")
        
        console.print()  # Blank line between turns
    
    # Display context files
    if session.context_files:
        console.print(f"[bold]Context Files ({len(session.context_files)}):[/bold]")
        for f in sorted(session.context_files)[:10]:
            console.print(f"  • {f}")
        if len(session.context_files) > 10:
            console.print(f"  [dim]... and {len(session.context_files) - 10} more[/dim]")
    
    # Display user preferences
    if session.user_preferences:
        console.print(f"\n[bold]User Preferences:[/bold]")
        for key, value in session.user_preferences.items():
            console.print(f"  • {key}: {value}")


def delete_session(session_id: str, force: bool = False) -> None:
    """
    Delete a conversation session.
    
    Args:
        session_id: The session ID to delete.
        force: If True, skip confirmation prompt.
    """
    conv_manager = get_conversation_manager()
    
    # Check if session exists
    if session_id not in conv_manager.list_sessions():
        console.print(f"[red]Error: Session '{session_id}' not found.[/red]")
        return
    
    # Confirm deletion
    if not force:
        from rich.prompt import Confirm
        if not Confirm.ask(f"Delete session '{session_id}'?", default=False):
            console.print("[yellow]Cancelled.[/yellow]")
            return
    
    # Delete session
    if conv_manager.delete_session(session_id):
        console.print(f"[green]✓[/green] Session '{session_id}' deleted.")
    else:
        console.print(f"[red]Error: Failed to delete session '{session_id}'.[/red]")


def continue_last_session() -> Optional[str]:
    """
    Get the ID of the most recent session.
    
    Returns:
        Session ID of the most recent session, or None if no sessions exist.
    """
    conv_manager = get_conversation_manager()
    sessions = conv_manager.list_sessions()
    
    if not sessions:
        return None
    
    # Find session with most recent turn
    most_recent_session = None
    most_recent_time = None
    
    for sess_id in sessions:
        if sess_id not in conv_manager.sessions:
            conv_manager._load_session(sess_id)
        
        session = conv_manager.sessions.get(sess_id)
        if session and session.turns:
            last_turn = session.turns[-1]
            try:
                dt = datetime.fromisoformat(last_turn.timestamp)
                if most_recent_time is None or dt > most_recent_time:
                    most_recent_time = dt
                    most_recent_session = sess_id
            except Exception:
                continue
    
    return most_recent_session


def export_session(session_id: str, output_path: Optional[Path] = None) -> None:
    """
    Export a session to a markdown file.
    
    Args:
        session_id: The session ID to export.
        output_path: Optional output path. Defaults to .lca/exports/{session_id}.md
    """
    conv_manager = get_conversation_manager()
    
    # Load session
    if session_id not in conv_manager.sessions:
        if not conv_manager._load_session(session_id):
            console.print(f"[red]Error: Session '{session_id}' not found.[/red]")
            return
    
    session = conv_manager.sessions[session_id]
    
    # Determine output path
    if output_path is None:
        output_path = Path(".lca/exports") / f"{session_id}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build markdown content
    lines = [
        f"# Session: {session_id}",
        "",
        f"**Turns:** {len(session.turns)}",
        f"**Context Files:** {len(session.context_files)}",
        "",
        "---",
        "",
    ]
    
    for i, turn in enumerate(session.turns, 1):
        try:
            dt = datetime.fromisoformat(turn.timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = turn.timestamp
        
        lines.append(f"## Turn {i} ({time_str})")
        lines.append("")
        lines.append(f"**Type:** {turn.task_type}")
        lines.append("")
        lines.append("### Task")
        lines.append("")
        lines.append(turn.task)
        lines.append("")
        
        if turn.files_referenced:
            lines.append("### Files Referenced")
            lines.append("")
            for f in turn.files_referenced:
                lines.append(f"- `{f}`")
            lines.append("")
        
        if turn.response:
            lines.append("### Response")
            lines.append("")
            lines.append(turn.response)
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    if session.context_files:
        lines.append("## All Context Files")
        lines.append("")
        for f in sorted(session.context_files):
            lines.append(f"- `{f}`")
        lines.append("")
    
    # Write to file
    try:
        output_path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]✓[/green] Session exported to: {output_path}")
    except Exception as e:
        console.print(f"[red]Error: Failed to export session: {e}[/red]")
