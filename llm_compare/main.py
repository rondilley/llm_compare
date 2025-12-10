"""CLI entry point for LLM Compare."""

import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .config import Config, default_config, MAX_PROMPT_SIZE
from .session.manager import SessionManager
from .report.generator import ReportGenerator
from .utils.logging import setup_logging, get_logger


class PromptValidationError(Exception):
    """Raised when prompt validation fails."""
    pass


def validate_prompt(prompt: str) -> str:
    """
    Validate and sanitize user prompt.

    Args:
        prompt: User-provided prompt string

    Returns:
        Validated prompt string

    Raises:
        PromptValidationError: If prompt exceeds size limits
    """
    if not prompt:
        raise PromptValidationError("Prompt cannot be empty")

    if len(prompt) > MAX_PROMPT_SIZE:
        raise PromptValidationError(
            f"Prompt too large ({len(prompt):,} characters). "
            f"Maximum allowed: {MAX_PROMPT_SIZE:,} characters."
        )

    return prompt

# Create console - use ASCII-safe mode on Windows to avoid encoding issues
if sys.platform == 'win32':
    console = Console(force_terminal=True, no_color=False)
else:
    console = Console()
logger = get_logger(__name__)


def get_progress():
    """Get a Progress instance with ASCII-safe spinner for Windows."""
    if sys.platform == 'win32':
        # Use ASCII-only spinner on Windows
        return Progress(
            SpinnerColumn(spinner_name="line"),  # Uses -, \, |, /
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        )
    else:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )


def print_banner():
    """Print application banner."""
    banner = """
    LLM Compare - Multi-AI Evaluation Tool
    ======================================
    """
    console.print(Panel(banner.strip(), style="bold blue"))


def print_providers(providers: List[str]):
    """Print discovered providers."""
    table = Table(title="Discovered Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")

    for provider in providers:
        table.add_row(provider.upper(), "Ready")

    console.print(table)


def print_rankings(rankings):
    """Print final rankings."""
    table = Table(title="Final Rankings")
    table.add_column("Rank", style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("95% CI")

    for ranked in rankings.rankings:
        ci = f"[{ranked.confidence_interval[0]:.2f}, {ranked.confidence_interval[1]:.2f}]"
        table.add_row(
            str(ranked.rank),
            ranked.provider.upper(),
            f"{ranked.score:.2f}",
            ci
        )

    console.print(table)


@click.group(invoke_without_command=True)
@click.option('--prompt', '-p', type=str, help='Prompt to evaluate')
@click.option('--providers', '-P', type=str, help='Comma-separated list of providers to use')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--skip', '-s', type=str, help='Phases to skip (comma-separated)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, prompt, providers, output, skip, verbose):
    """LLM Compare - Multi-AI comparison and evaluation tool."""
    ctx.ensure_object(dict)

    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=getattr(__import__('logging'), log_level))

    # Store options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['output'] = Path(output) if output else default_config.output_dir
    ctx.obj['skip'] = skip.split(',') if skip else []
    ctx.obj['providers_filter'] = providers.split(',') if providers else None

    # If no subcommand, run evaluation
    if ctx.invoked_subcommand is None:
        if prompt:
            ctx.invoke(evaluate, prompt=prompt)
        else:
            ctx.invoke(interactive)


@cli.command()
@click.option('--prompt', '-p', type=str, required=True, help='Prompt to evaluate')
@click.pass_context
def evaluate(ctx, prompt):
    """Evaluate a prompt across all available LLMs."""
    # Validate prompt size
    try:
        prompt = validate_prompt(prompt)
    except PromptValidationError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    print_banner()

    config = Config(output_dir=ctx.obj.get('output', default_config.output_dir))
    manager = SessionManager(config=config)

    # Discover providers
    with get_progress() as progress:
        task = progress.add_task("Discovering providers...", total=None)
        available = manager.discover_providers()
        progress.update(task, completed=True)

    if not available:
        console.print("[red]Error: No providers found. Check API key files.[/red]")
        sys.exit(1)

    # Filter providers if specified
    provider_filter = ctx.obj.get('providers_filter')
    if provider_filter:
        available = [p for p in available if p in provider_filter]
        if len(available) < 2:
            console.print("[red]Error: Need at least 2 providers after filtering.[/red]")
            sys.exit(1)

    print_providers(available)
    console.print()

    if len(available) < 2:
        console.print("[red]Error: Need at least 2 providers for comparison.[/red]")
        sys.exit(1)

    # Create and run session
    session = manager.create_session(prompt)
    console.print(f"[bold]Session ID:[/bold] {session.session_id}")
    console.print(f"[bold]Prompt:[/bold] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    console.print()

    skip_phases = ctx.obj.get('skip', [])

    with get_progress() as progress:
        # Phase 0: Collect responses
        task = progress.add_task("Collecting responses from providers...", total=None)
        try:
            session = manager.run_session(session, skip_phases=skip_phases)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        progress.update(task, completed=True)

    console.print()
    console.print("[green]Evaluation complete![/green]")
    console.print()

    # Print rankings
    if session.final_rankings:
        print_rankings(session.final_rankings)
        console.print()

    # Generate PDF report
    console.print("Generating PDF report...")
    report_gen = ReportGenerator()
    session_dir = manager.storage.get_session_dir(session.session_id)
    report_path = session_dir / "report.pdf"

    try:
        report_gen.generate(session, report_path)
        console.print(f"[green]Report saved:[/green] {report_path}")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate PDF report: {e}[/yellow]")

    console.print(f"[green]Session data saved:[/green] {session_dir / 'data.json'}")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Interactive mode - enter prompts interactively."""
    print_banner()

    config = Config(output_dir=ctx.obj.get('output', default_config.output_dir))
    manager = SessionManager(config=config)

    # Discover providers
    console.print("Discovering providers...")
    available = manager.discover_providers()

    if not available:
        console.print("[red]Error: No providers found. Check API key files.[/red]")
        sys.exit(1)

    print_providers(available)
    console.print()

    if len(available) < 2:
        console.print("[red]Error: Need at least 2 providers for comparison.[/red]")
        sys.exit(1)

    console.print("Enter prompts to evaluate. Type 'quit' or 'exit' to stop.")
    console.print()

    while True:
        try:
            prompt = console.input("[bold cyan]Prompt>[/bold cyan] ").strip()

            if not prompt:
                continue

            if prompt.lower() in ('quit', 'exit', 'q'):
                console.print("Goodbye!")
                break

            # Validate prompt
            try:
                prompt = validate_prompt(prompt)
            except PromptValidationError as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

            # Create and run session
            session = manager.create_session(prompt)
            console.print(f"[bold]Session ID:[/bold] {session.session_id}")
            console.print()

            with get_progress() as progress:
                task = progress.add_task("Running evaluation pipeline...", total=None)
                try:
                    session = manager.run_session(session)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue
                progress.update(task, completed=True)

            console.print()

            # Print rankings
            if session.final_rankings:
                print_rankings(session.final_rankings)
                console.print()

            # Generate PDF report
            report_gen = ReportGenerator()
            session_dir = manager.storage.get_session_dir(session.session_id)
            report_path = session_dir / "report.pdf"

            try:
                report_gen.generate(session, report_path)
                console.print(f"[green]Report saved:[/green] {report_path}")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to generate PDF: {e}[/yellow]")

            console.print()

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break
        except EOFError:
            break


@cli.command()
@click.pass_context
def providers(ctx):
    """List available providers."""
    config = Config(output_dir=ctx.obj.get('output', default_config.output_dir))
    manager = SessionManager(config=config)

    available = manager.discover_providers()

    if not available:
        console.print("[yellow]No providers found. Check API key files:[/yellow]")
        for provider, key_file in config.provider_key_files.items():
            exists = (config.project_root / key_file).exists()
            status = "[green]Found[/green]" if exists else "[red]Missing[/red]"
            console.print(f"  {provider}: {key_file} - {status}")
    else:
        print_providers(available)


@cli.command()
@click.pass_context
def sessions(ctx):
    """List saved evaluation sessions."""
    config = Config(output_dir=ctx.obj.get('output', default_config.output_dir))
    manager = SessionManager(config=config)

    session_list = manager.list_sessions()

    if not session_list:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Status")

    for session_id in session_list:
        data = manager.get_session(session_id)
        status = data.get('status', 'unknown') if data else 'unknown'
        table.add_row(session_id, status)

    console.print(table)


@cli.command()
@click.argument('session_id')
@click.pass_context
def show(ctx, session_id):
    """Show details of a saved session."""
    config = Config(output_dir=ctx.obj.get('output', default_config.output_dir))
    manager = SessionManager(config=config)

    data = manager.get_session(session_id)

    if not data:
        console.print(f"[red]Session not found: {session_id}[/red]")
        sys.exit(1)

    console.print(Panel(f"Session: {session_id}", style="bold"))

    console.print(f"[bold]Status:[/bold] {data.get('status', 'unknown')}")
    console.print(f"[bold]Created:[/bold] {data.get('created_at', 'unknown')}")
    console.print(f"[bold]Prompt:[/bold] {data.get('prompt', '')[:200]}...")
    console.print()

    # Show rankings if available
    rankings_data = data.get('rankings')
    if rankings_data and rankings_data.get('rankings'):
        table = Table(title="Rankings")
        table.add_column("Rank")
        table.add_column("Provider")
        table.add_column("Score")

        for r in rankings_data['rankings']:
            table.add_row(str(r['rank']), r['provider'].upper(), f"{r['score']:.2f}")

        console.print(table)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
