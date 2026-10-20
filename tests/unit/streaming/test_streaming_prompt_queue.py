from xfuser.streaming.prompt_cli import clean_prompt
from xfuser.streaming.prompt_queue import (
    append_prompt,
    clear_prompt_files,
    latest_prompt,
    read_new_prompts,
    read_overlay,
    read_prompt_state,
    toggle_overlay,
    write_overlay,
    write_prompt_state,
)


def test_append_assigns_sequential_ids_and_strips(tmp_path):
    first = append_prompt(tmp_path, "a cat")
    second = append_prompt(tmp_path, "  a dog  ")

    assert (first.id, first.text) == (1, "a cat")
    assert (second.id, second.text) == (2, "a dog")


def test_read_new_prompts_filters_by_id(tmp_path):
    append_prompt(tmp_path, "one")
    append_prompt(tmp_path, "two")
    append_prompt(tmp_path, "three")

    assert [p.id for p in read_new_prompts(tmp_path, 0)] == [1, 2, 3]
    newest = read_new_prompts(tmp_path, 2)
    assert [p.text for p in newest] == ["three"]
    assert read_new_prompts(tmp_path, 3) == []


def test_read_new_prompts_on_missing_file(tmp_path):
    assert read_new_prompts(tmp_path, 0) == []


def test_prompt_state_round_trip_and_clear(tmp_path):
    assert read_prompt_state(tmp_path) is None

    write_prompt_state(tmp_path, {"id": 2, "status": "live", "user_text": "a dog"})
    state = read_prompt_state(tmp_path)
    assert state["status"] == "live" and state["user_text"] == "a dog"

    append_prompt(tmp_path, "later")
    clear_prompt_files(tmp_path)
    assert read_prompt_state(tmp_path) is None
    assert read_new_prompts(tmp_path, 0) == []


def test_latest_prompt_is_the_newest_typed(tmp_path):
    assert latest_prompt(tmp_path) is None
    append_prompt(tmp_path, "one")
    append_prompt(tmp_path, "two")
    assert latest_prompt(tmp_path).text == "two"


def test_overlay_switch_toggles_and_clears(tmp_path):
    assert read_overlay(tmp_path) is None
    assert toggle_overlay(tmp_path) is True
    assert read_overlay(tmp_path) is True
    assert toggle_overlay(tmp_path) is False
    assert read_overlay(tmp_path) is False

    write_overlay(tmp_path, True)
    clear_prompt_files(tmp_path)
    assert read_overlay(tmp_path) is None


def test_clean_prompt_strips_arrow_key_escapes():
    typed = "They are oing\x1b[D\x1b[D\x1b[D to a submarine\x1b[C\x1b[C,  then\tswim"
    assert clean_prompt(typed) == "They are oing to a submarine, then swim"
