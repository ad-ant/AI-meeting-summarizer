from unittest.mock import MagicMock, patch

import pytest
import requests

from processing.summarize import (
    _extract_tasks_from_tool_call,
    merge_summaries,
    run_agentic_analysis,
    summarize_chunk,
)


def _mock_response(json_data: dict) -> MagicMock:
    """Build a fake requests.Response that returns json_data and never errors on status."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def test_extract_tasks_from_list_arguments():
    tool_call = {
        "function": {
            "arguments": {
                "tasks": ["Task A", "  Task B  ", ""]
            }
        }
    }

    tasks = _extract_tasks_from_tool_call(tool_call)

    assert tasks == ["Task A", "Task B"]


def test_extract_tasks_from_json_string_arguments():
    tool_call = {
        "function": {
            "arguments": '{"tasks": ["Task A", "Task B"]}'
        }
    }

    tasks = _extract_tasks_from_tool_call(tool_call)

    assert tasks == ["Task A", "Task B"]


def test_extract_tasks_from_comma_separated_string():
    tool_call = {
        "function": {
            "arguments": {
                "tasks": "Task A, Task B, Task C"
            }
        }
    }

    tasks = _extract_tasks_from_tool_call(tool_call)

    assert tasks == ["Task A", "Task B", "Task C"]


@patch("processing.summarize.requests.post")
def test_summarize_chunk_returns_response_text(mock_post):
    mock_post.return_value = _mock_response({"response": "Krotkie podsumowanie."})

    result = summarize_chunk("tresc spotkania")

    assert result == "Krotkie podsumowanie."


@patch("processing.summarize.requests.post")
def test_summarize_chunk_retries_after_timeout_then_succeeds(mock_post, monkeypatch):
    monkeypatch.setattr("processing.summarize.SUMMARY_MAX_RETRIES", 2)
    mock_post.side_effect = [
        requests.ReadTimeout("timed out"),
        _mock_response({"response": "OK po drugiej probie."}),
    ]

    result = summarize_chunk("tresc")

    assert result == "OK po drugiej probie."
    assert mock_post.call_count == 2


@patch("processing.summarize.requests.post")
def test_summarize_chunk_raises_when_ollama_never_responds(mock_post, monkeypatch):
    monkeypatch.setattr("processing.summarize.SUMMARY_MAX_RETRIES", 2)
    mock_post.side_effect = requests.ReadTimeout("timed out")

    with pytest.raises(RuntimeError):
        summarize_chunk("tresc")


def test_merge_summaries_skips_ollama_for_single_chunk():
    result = merge_summaries(["Jedyne streszczenie."])

    assert result == "Jedyne streszczenie."


@patch("processing.summarize.requests.post")
def test_merge_summaries_combines_multiple_chunks(mock_post):
    mock_post.return_value = _mock_response({"response": "Spojne podsumowanie calego spotkania."})

    result = merge_summaries(["Fragment 1.", "Fragment 2."])

    assert result == "Spojne podsumowanie calego spotkania."
    mock_post.assert_called_once()


@patch("processing.summarize.requests.post")
def test_run_agentic_analysis_saves_action_items(mock_post, tmp_path):
    tool_call = {
        "function": {
            "name": "save_action_items",
            "arguments": {"tasks": ["Jan wysyla raport do piatku"]},
        }
    }
    mock_post.return_value = _mock_response({"message": {"tool_calls": [tool_call]}})
    output_file = tmp_path / "action_items.md"

    run_agentic_analysis("podsumowanie spotkania", str(output_file))

    assert output_file.exists()
    assert "Jan wysyla raport do piatku" in output_file.read_text(encoding="utf-8")


@patch("processing.summarize.requests.post")
def test_run_agentic_analysis_returns_quietly_when_no_tool_call(mock_post, tmp_path):
    mock_post.return_value = _mock_response({"message": {"tool_calls": []}})
    output_file = tmp_path / "action_items.md"

    run_agentic_analysis("podsumowanie spotkania", str(output_file))

    assert not output_file.exists()


@patch("processing.summarize.requests.post")
def test_run_agentic_analysis_retries_after_bad_json_then_succeeds(mock_post, monkeypatch, tmp_path):
    monkeypatch.setattr("processing.summarize.AGENT_MAX_RETRIES", 2)
    bad_tool_call = {
        "function": {"name": "save_action_items", "arguments": "not valid json"}
    }
    good_tool_call = {
        "function": {
            "name": "save_action_items",
            "arguments": {"tasks": ["Anna przygotowuje prezentacje"]},
        }
    }
    mock_post.side_effect = [
        _mock_response({"message": {"tool_calls": [bad_tool_call]}}),
        _mock_response({"message": {"tool_calls": [good_tool_call]}}),
    ]
    output_file = tmp_path / "action_items.md"

    run_agentic_analysis("podsumowanie spotkania", str(output_file))

    assert mock_post.call_count == 2
    assert "Anna przygotowuje prezentacje" in output_file.read_text(encoding="utf-8")
