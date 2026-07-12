import json
import logging
from pathlib import Path

import requests

from config import AGENT_MAX_RETRIES, OLLAMA_BASE_URL, OLLAMA_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def summarize_chunk(text: str) -> str:
    """Generate a short Polish summary for one transcript chunk."""
    prompt = (
        "Odpowiadaj po polsku.\n"
        "Stwórz krótkie, spójne streszczenie fragmentu spotkania.\n"
        "Pisz pełnymi zdaniami, bez punktów.\n\n"
        f"{text}"
    )

    last_error: Exception | None = None

    for attempt in range(1, 3):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            summary = data.get("response")
            if not isinstance(summary, str):
                raise ValueError("Invalid Ollama response: missing 'response' field")

            return summary
        except requests.ReadTimeout as err:
            last_error = err
            logger.warning("Summary timeout (attempt %s/2)", attempt)

    raise RuntimeError(
        "Ollama summary request timed out. Increase REQUEST_TIMEOUT in .env "
        "or use a faster model."
    ) from last_error


def save_action_items(tasks: list[str], output_path: str) -> str:
    """Save action items as a markdown checklist."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Zadania ze spotkania (Action Items)",
        "",
        "*Wygenerowane autonomicznie przez Agenta na podstawie podsumowania.*",
        "",
    ]
    lines.extend(f"- [ ] {task}" for task in tasks)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved %s action items to %s", len(tasks), output_path)
    return f"Sukces: Zapisano {len(tasks)} zadań w {output_path}."


def _extract_tasks_from_tool_call(tool_call: dict) -> list[str]:
    args = tool_call.get("function", {}).get("arguments", {})

    if isinstance(args, str):
        args = json.loads(args)

    tasks = args.get("tasks", [])
    if isinstance(tasks, list):
        return [str(t).strip() for t in tasks if str(t).strip()]

    if isinstance(tasks, str):
        # Sometimes model returns string instead of JSON array.
        try:
            parsed = json.loads(tasks)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            return [t.strip() for t in tasks.split(",") if t.strip()]

    return []


def _looks_like_action_item(text: str) -> bool:
    text = text.strip()
    if len(text.split()) < 3:
        return False

    bad_starts = ("temat", "podsumowanie", "spotkanie", "budownictwo", "geotechnika")
    return not text.lower().startswith(bad_starts)


def _filter_action_items(tasks: list[str]) -> list[str]:
    filtered: list[str] = []
    for task in tasks:
        cleaned = " ".join(task.split())
        if _looks_like_action_item(cleaned):
            filtered.append(cleaned)
    return filtered


def run_agentic_analysis(full_summary: str, output_file_path: str) -> None:
    """Run tool-calling agent to extract and save action items."""
    logger.info("Starting agentic analysis")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_action_items",
                "description": "Save list of extracted action items to markdown file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of extracted action items",
                        }
                    },
                    "required": ["tasks"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem ekstrakcji zadań. "
                "Zwracaj tylko konkretne action items (kto/co ma zrobić). "
                "Jeśli nie ma realnych zadań, zwróć pustą listę tasks. "
                "Nie zwracaj tematów spotkania ani ogólnych haseł. "
                "Zwracaj poprawny JSON argumentów funkcji."
            ),
        },
        {"role": "user", "content": f"Oto podsumowanie spotkania:\n{full_summary}"},
    ]

    for attempt in range(1, AGENT_MAX_RETRIES + 2):
        try:
            logger.info("Agent attempt %s/%s", attempt, AGENT_MAX_RETRIES + 1)

            response = requests.post(
                f"{OLLAMA_BASE_URL}/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                logger.info("No action items detected")
                return

            for tool_call in tool_calls:
                if tool_call.get("function", {}).get("name") != "save_action_items":
                    continue

                tasks = _extract_tasks_from_tool_call(tool_call)
                tasks = _filter_action_items(tasks)

                if not tasks:
                    logger.info("No valid action items after filtering")
                    return

                result = save_action_items(tasks, output_file_path)
                logger.info(result)
                return

            logger.warning("Tool call present, but save_action_items not found")
            return

        except json.JSONDecodeError as err:
            logger.warning("JSON parse error on attempt %s: %s", attempt, err)
            if attempt <= AGENT_MAX_RETRIES:
                messages.append({"role": "assistant", "content": str(message)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Błąd JSON: {err}. "
                            "Popraw format argumentów funkcji i spróbuj ponownie."
                        ),
                    }
                )
            else:
                logger.error("Agent failed to produce valid JSON after retries")
                return

        except requests.RequestException as err:
            logger.error("Ollama request failed: %s", err)
            return

        except ValueError as err:
            logger.error("Invalid data format: %s", err)
            return