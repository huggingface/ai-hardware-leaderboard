from typing import List
from rich.table import Table
from rich.prompt import Prompt
from rich.console import Console

console = Console()

def display_hardware_menu(recommended_hardware: List[str] = []) -> str:
    table = Table(title="Hardware Configuration Options")
    table.add_column("Option", justify="right", style="cyan")
    table.add_column("Hardware Type", style="magenta")
    table.add_column("Status", style="green")

    # Add recommended hardware first
    option_num = 1
    for hw_type, status in recommended_hardware:
        table.add_row(str(option_num), hw_type, f"✓ {status}")
        option_num += 1

    # Add other available options
    all_hardware = [
        'cuda', 'intel_cpu', 'habana', 'intel_gpu', 'tpu', 
        'inferentia', 'rocm', 'amd_cpu', 'apple_silicon'
    ]
    
    recommended_types = [hw[0] for hw in recommended_hardware]
    for hw_type in all_hardware:
        if hw_type not in recommended_types:
            table.add_row(str(option_num), hw_type, "Available")
            option_num += 1

    # Add default option
    table.add_row(str(option_num), "default_settings", "Fallback option")

    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "Select hardware type (enter number)",
            default="1"
        )
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= option_num:
                if choice_num <= len(recommended_hardware):
                    return recommended_hardware[choice_num - 1][0]
                elif choice_num == option_num:
                    return 'default_settings'
                else:
                    remaining_options = [hw for hw in all_hardware if hw not in recommended_types]
                    return remaining_options[choice_num - len(recommended_hardware) - 1]
        except ValueError:
            pass
        
        console.print("[red]Invalid choice. Please try again.[/red]")