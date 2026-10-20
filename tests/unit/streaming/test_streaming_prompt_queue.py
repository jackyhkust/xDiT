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


def test_watch_prompts_announces_new_prompts_from_elsewhere(tmp_path):
    import threading
    import time

    from xfuser.streaming.prompt_cli import watch_prompts

    append_prompt(tmp_path, "before the console opened")
    stop, seen, typed, lock = threading.Event(), [], set(), threading.Lock()
    watcher = threading.Thread(
        target=watch_prompts, args=(tmp_path, stop, seen.append, typed, lock), kwargs={"poll_s": 0.01}
    )
    watcher.start()
    time.sleep(0.05)
    append_prompt(tmp_path, "from the page", source="web")
    with lock:
        typed.add(append_prompt(tmp_path, "typed here").id)
    append_prompt(tmp_path, "scripted", source="demo")
    deadline = time.monotonic() + 5
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    watcher.join()

    assert [(p.text, p.source) for p in seen] == [("from the page", "web"), ("scripted", "demo")]


def test_prompts_keep_their_source(tmp_path):
    append_prompt(tmp_path, "typed")
    append_prompt(tmp_path, "from the page", source="web")
    append_prompt(tmp_path, "scripted", source="demo")

    assert [p.source for p in read_new_prompts(tmp_path, 0)] == ["console", "web", "demo"]


def test_concurrent_appends_get_unique_ids(tmp_path):
    import threading

    threads = [threading.Thread(target=append_prompt, args=(tmp_path, f"prompt {i}")) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    prompts = read_new_prompts(tmp_path, 0)
    assert sorted(p.id for p in prompts) == list(range(1, 33))
    assert sorted(p.text for p in prompts) == sorted(f"prompt {i}" for i in range(32))
