#!/usr/bin/env python3
"""
DeepSeek Terminal Chat - A rich CLI interface for OpenRouter AI models
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import pickle

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Initialize Rich console
console = Console()

# Session management
SESSION_DIR = Path.home() / ".deepseek_chat"
SESSION_DIR.mkdir(exist_ok=True)

class ChatSession:
    """Manages chat sessions with persistent storage"""
    
    def __init__(self, name: str = "default", model: str = "deepseek/deepseek-chat"):
        self.name = name
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.session_file = SESSION_DIR / f"{name}.session"
        self.load()
    
    def load(self):
        """Load session from disk if exists"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'rb') as f:
                    data = pickle.load(f)
                    self.__dict__.update(data)
                    self.updated_at = datetime.now()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load session: {e}[/yellow]")
    
    def save(self):
        """Save session to disk"""
        self.updated_at = datetime.now()
        try:
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.__dict__, f)
        except Exception as e:
            console.print(f"[red]Error saving session: {e}[/red]")
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})
        self.save()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages including system prompt"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.messages)
        return messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.save()
    
    def delete(self):
        """Delete the session file"""
        if self.session_file.exists():
            self.session_file.unlink()
    
    @staticmethod
    def list_sessions() -> List[str]:
        """List all available session names"""
        return [f.stem for f in SESSION_DIR.glob("*.session")]

class DeepSeekChat:
    """Main chat interface with rich formatting"""
    
    AVAILABLE_MODELS = [
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "anthropic/claude-3-haiku",
        "qwen/qwen-2.5-72b-instruct"
    ]
    
    def __init__(self):
        self.current_session: Optional[ChatSession] = None
        self.check_api_key()
    
    def check_api_key(self):
        """Check if API key is set"""
        if not API_KEY:
            console.print("[red]❌ API_KEY not found in .env file![/red]")
            console.print("[yellow]Please create a .env file with API_KEY=your_key[/yellow]")
            sys.exit(1)
    
    def create_header(self) -> Panel:
        """Create the application header"""
        header_text = Text()
        header_text.append("🤖 DeepSeek Terminal Chat\n", style="bold cyan")
        header_text.append(f"Session: {self.current_session.name if self.current_session else 'None'}", style="green")
        if self.current_session and self.current_session.model:
            header_text.append(f" | Model: {self.current_session.model}", style="yellow")
        if self.current_session and self.current_session.messages:
            header_text.append(f" | Messages: {len(self.current_session.messages)}", style="magenta")
        return Panel(header_text, box=box.HEAVY, style="blue")
    
    def show_welcome(self):
        """Display welcome message with commands"""
        console.clear()
        console.print(self.create_header())
        
        welcome_text = """
[bold cyan]Welcome to DeepSeek Terminal Chat![/bold cyan]
[dim]Type your messages and press Enter to send[/dim]
[dim]Use /help to see available commands[/dim]
"""
        console.print(Panel(welcome_text, box=box.ROUNDED, style="green"))
    
    def show_help(self):
        """Display help panel"""
        help_text = """
[bold]Available Commands:[/bold]

[green]/help[/green]     - Show this help message
[green]/new[/green]      - Start a new session (you'll be prompted for a name)
[green]/load[/green]     - Load an existing session
[green]/model[/green]    - Change the AI model
[green]/prompt[/green]   - Set or change system prompt
[green]/clear[/green]    - Clear current session's conversation history
[green]/save[/green]     - Save current session manually
[green]/list[/green]     - List all available sessions
[green]/delete[/green]   - Delete a session
[green]/history[/green]  - Show conversation history
[green]/exit[/green]     - Exit the application

[dim]Tip: You can also just type / followed by the command[/dim]
"""
        console.print(Panel(help_text, title="📚 Help", box=box.HEAVY, style="cyan"))
    
    def show_history(self):
        """Display conversation history with syntax highlighting"""
        if not self.current_session or not self.current_session.messages:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        console.print(f"\n[bold]📜 Conversation History ({len(self.current_session.messages)} messages)[/bold]\n")
        
        for i, msg in enumerate(self.current_session.messages, 1):
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                console.print(f"[bold cyan]{i}. You:[/bold cyan] {content}")
            else:
                console.print(f"[bold green]{i}. Assistant:[/bold green]")
                markdown = Markdown(content)
                console.print(markdown)
            console.print("─" * 60)
    
    def select_model(self) -> str:
        """Interactive model selection"""
        console.print("\n[bold]Available Models:[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim")
        table.add_column("Model Name", style="cyan")
        table.add_column("Description", style="yellow")
        
        descriptions = {
            "deepseek/deepseek-chat": "General purpose chat model",
            "deepseek/deepseek-coder": "Specialized for coding tasks",
            "meta-llama/llama-3.3-70b-instruct": "Meta's latest instruct model",
            "mistralai/mistral-7b-instruct": "Efficient and capable",
            "google/gemini-pro": "Google's advanced model",
            "anthropic/claude-3-haiku": "Fast and affordable",
            "microsoft/phi-3-mini-128k": "Small but powerful",
            "qwen/qwen-2.5-72b-instruct": "Large parameter model"
        }
        
        for i, model in enumerate(self.AVAILABLE_MODELS, 1):
            desc = descriptions.get(model, "No description")
            table.add_row(str(i), model, desc)
        
        console.print(table)
        
        choice = Prompt.ask(
            "\n[bold]Select a model number[/bold]",
            choices=[str(i) for i in range(1, len(self.AVAILABLE_MODELS) + 1)],
            default="1"
        )
        
        return self.AVAILABLE_MODELS[int(choice) - 1]
    
    def create_new_session(self):
        """Create a new chat session"""
        name = Prompt.ask("[bold cyan]Enter session name[/bold cyan]", default=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        # Check if session already exists
        if (SESSION_DIR / f"{name}.session").exists():
            if not Confirm.ask(f"[yellow]Session '{name}' already exists. Override?[/yellow]"):
                return
        
        console.print("[bold]Select model for this session:[/bold]")
        model = self.select_model()
        
        # Option to set system prompt
        if Confirm.ask("[bold cyan]Would you like to set a system prompt?[/bold cyan]"):
            prompt = Prompt.ask("[bold]Enter system prompt[/bold]")
        else:
            prompt = None
        
        self.current_session = ChatSession(name, model)
        if prompt:
            self.current_session.system_prompt = prompt
            self.current_session.save()
        
        console.print(f"[green]✅ Session '{name}' created successfully![/green]")
        self.show_welcome()
    
    def load_session(self):
        """Load an existing session"""
        sessions = ChatSession.list_sessions()
        
        if not sessions:
            console.print("[yellow]No saved sessions found.[/yellow]")
            return
        
        console.print("\n[bold]Available Sessions:[/bold]")
        for i, session in enumerate(sessions, 1):
            console.print(f"  {i}. {session}")
        
        choice = Prompt.ask(
            "\n[bold]Select session number[/bold]",
            choices=[str(i) for i in range(1, len(sessions) + 1)]
        )
        
        session_name = sessions[int(choice) - 1]
        self.current_session = ChatSession(session_name)
        console.print(f"[green]✅ Loaded session: {session_name}[/green]")
        self.show_welcome()
    
    def delete_session(self):
        """Delete a session"""
        sessions = ChatSession.list_sessions()
        
        if not sessions:
            console.print("[yellow]No saved sessions to delete.[/yellow]")
            return
        
        console.print("\n[bold]Sessions to delete:[/bold]")
        for i, session in enumerate(sessions, 1):
            console.print(f"  {i}. {session}")
        
        choice = Prompt.ask(
            "\n[bold]Select session to delete[/bold]",
            choices=[str(i) for i in range(1, len(sessions) + 1)]
        )
        
        session_name = sessions[int(choice) - 1]
        if Confirm.ask(f"[red]Are you sure you want to delete '{session_name}'?[/red]"):
            session = ChatSession(session_name)
            session.delete()
            if self.current_session and self.current_session.name == session_name:
                self.current_session = None
            console.print(f"[green]✅ Session '{session_name}' deleted.[/green]")
    
    def send_message(self, user_input: str) -> str:
        """Send a message and get AI response"""
        if not self.current_session:
            console.print("[red]No active session. Use /new or /load first.[/red]")
            return ""
        
        # Add user message
        self.current_session.add_message("user", user_input)
        
        # Prepare API request
        data = {
            "model": self.current_session.model,
            "messages": self.current_session.get_messages(),
            "stream": False
        }
        
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="🤔 Thinking...", total=None)
                
                response = requests.post(API_URL, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                ai_message = response.json()['choices'][0]['message']['content']
                self.current_session.add_message("assistant", ai_message)
                return ai_message
            else:
                error_msg = f"❌ Error: {response.status_code} - {response.text}"
                console.print(f"[red]{error_msg}[/red]")
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "⏰ Request timed out. Please try again."
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"🔌 Network error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
        except Exception as e:
            error_msg = f"💥 Unexpected error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def run(self):
        """Main application loop"""
        # Check for command-line arguments
        if len(sys.argv) > 1:
            session_name = sys.argv[1]
            if (SESSION_DIR / f"{session_name}.session").exists():
                self.current_session = ChatSession(session_name)
                console.print(f"[green]Loaded session: {session_name}[/green]")
            else:
                console.print(f"[yellow]Session '{session_name}' not found. Starting new session...[/yellow]")
                self.current_session = ChatSession(session_name)
        
        if not self.current_session:
            # Show initial menu
            console.clear()
            console.print(Panel.fit(
                "[bold cyan]🤖 DeepSeek Terminal Chat[/bold cyan]",
                box=box.DOUBLE_EDGE
            ))
            console.print(Panel.fit(
               "[bold yellow]1. New Session\n2. Existing Session\n3. Exit[/]",
               box=box.DOUBLE_EDGE
            ))
            choice = Prompt.ask(
                "\n[bold]Choose an option[/bold]",
                choices=["1", "2", "3"],
                default="1"
            )
            
            if choice == "1":
                self.create_new_session()
            elif choice == "2":
                self.load_session()
            else:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                sys.exit(0)
        
        # Main chat loop
        self.show_welcome()
        
        while True:
            try:
                user_input = Prompt.ask(f"\n[bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower().strip()
                    
                    if command == "exit" or command == "quit":
                        console.print("[green]👋 Goodbye![/green]")
                        break
                    elif command == "help":
                        self.show_help()
                        continue
                    elif command == "new":
                        self.create_new_session()
                        continue
                    elif command == "load":
                        self.load_session()
                        continue
                    elif command == "model":
                        if self.current_session:
                            new_model = self.select_model()
                            self.current_session.model = new_model
                            self.current_session.save()
                            console.print(f"[green]✅ Model updated to: {new_model}[/green]")
                        continue
                    elif command == "prompt":
                        if self.current_session:
                            new_prompt = Prompt.ask("[bold]Enter new system prompt[/bold]")
                            self.current_session.system_prompt = new_prompt
                            self.current_session.save()
                            console.print("[green]✅ System prompt updated![/green]")
                        continue
                    elif command == "clear":
                        if self.current_session and Confirm.ask("[yellow]Clear conversation history?[/yellow]"):
                            self.current_session.clear()
                            console.print("[green]✅ Conversation history cleared.[/green]")
                        continue
                    elif command == "save":
                        if self.current_session:
                            self.current_session.save()
                            console.print("[green]✅ Session saved![/green]")
                        continue
                    elif command == "list":
                        sessions = ChatSession.list_sessions()
                        if sessions:
                            console.print("\n[bold]📋 Available Sessions:[/bold]")
                            for s in sessions:
                                console.print(f"  • {s}")
                        else:
                            console.print("[yellow]No saved sessions.[/yellow]")
                        continue
                    elif command == "delete":
                        self.delete_session()
                        continue
                    elif command == "history":
                        self.show_history()
                        continue
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        console.print("[dim]Type /help for available commands[/dim]")
                        continue
                
                # Send message and get response
                with console.status("[bold green]Processing...[/bold green]"):
                    response = self.send_message(user_input)
                
                # Display response with markdown
                if response and not response.startswith("❌") and not response.startswith("⏰") and not response.startswith("🔌") and not response.startswith("💥"):
                    console.print("\n[bold green]Assistant:[/bold green]")
                    markdown = Markdown(response)
                    console.print(markdown)
                    console.print("─" * 60)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Received interrupt. Use /exit to quit.[/yellow]")
                continue
            except EOFError:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                continue

def main():
    """Entry point"""
    try:
        chat = DeepSeekChat()
        chat.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    main()
