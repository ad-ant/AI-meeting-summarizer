from processing.summarize import _extract_tasks_from_tool_call


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
