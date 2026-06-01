import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

path = "../outputs/pipeline_results.jsonl"

with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        title = f"[bold blue]Example {i+1}[/bold blue]"

        content = Text()

        content.append("EN:\n", style="bold")
        content.append(data["source_en"] + "\n\n")

        content.append("MT:\n", style="bold red")
        content.append(data["mt_es"] + "\n\n")

        content.append("Corrected:\n", style="bold green")
        content.append(data["corrected_es"] + "\n\n")

        content.append("Needs human review: ", style="bold")
        if data["needs_human_review"]:
            content.append("YES\n\n", style="bold red")
        else:
            content.append("NO\n\n", style="bold green")

        content.append("Issues:\n", style="bold yellow")
        if data["issues"]:
            for issue in data["issues"]:
                content.append(f"- {issue['type']} ({issue['severity']}): {issue['explanation']}\n")
        else:
            content.append("None\n")

        console.print(Panel(content, title=title))

        if i >= 9:  # show only first 10
            break