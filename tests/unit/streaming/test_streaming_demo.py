import threading
import time

from xfuser.streaming.chunks import write_position
from xfuser.streaming.demo import PRODUCTS, STYLES, DemoDirector, run_demo
from xfuser.streaming.prompt_queue import append_prompt, read_new_prompts


def test_cast_grows_to_three_then_one_leaves_as_one_joins():
    director = DemoDirector(seed=0)
    sizes, casts = [], []
    for _ in range(8):
        before = list(director.cast)
        prompt = director.next_prompt()
        after = list(director.cast)
        sizes.append(len(after))
        casts.append((before, after, prompt))

    assert sizes == [1, 2, 3, 3, 3, 3, 3, 3]
    for before, after, prompt in casts:
        newcomer = after[-1]
        assert newcomer not in before
        assert len({c.name for c in after}) == len(after)
        if len(before) == 3:
            (leaver,) = [c for c in before if c not in after]
            assert "leaves the scene" in prompt
            assert leaver.name.split(" ", 1)[1] in prompt
        else:
            assert "walks into the scene" in prompt


def test_every_prompt_promotes_an_amd_product_with_a_style():
    director = DemoDirector(seed=1)
    prompts = [director.next_prompt() for _ in range(40)]
    products = {p for options in PRODUCTS.values() for p in options}

    for prompt in prompts:
        assert "AMD" in prompt
        assert any(product in prompt for product in products)
        assert any(f"Style: {style}." in prompt for style in STYLES)
    categories = {c for c, options in PRODUCTS.items() if any(o in p for o in options for p in prompts)}
    assert categories == set(PRODUCTS)


def test_storylines_styles_and_product_types_do_not_repeat_until_all_are_used():
    director = DemoDirector(seed=4)
    prompts = [director.next_prompt() for _ in range(len(STYLES))]

    assert {next(s for s in STYLES if f"Style: {s}." in p) for p in prompts} == set(STYLES)
    for start in range(0, len(prompts) - 3, 4):
        window = prompts[start : start + 4]
        assert {c for c, opts in PRODUCTS.items() for p in window if any(o in p for o in opts)} == set(PRODUCTS)


def test_given_cast_counts_toward_three():
    director = DemoDirector(cast=["the white cat", "the grey cat"], seed=2)

    first = director.next_prompt()
    assert "joins the white cat and the grey cat" in first
    assert len(director.cast) == 3
    director.next_prompt()
    assert len(director.cast) == 3


def test_same_seed_gives_the_same_script():
    a, b = DemoDirector(seed=7), DemoDirector(seed=7)
    assert [a.next_prompt() for _ in range(5)] == [b.next_prompt() for _ in range(5)]


def test_run_demo_waits_for_playback_and_restarts_after_a_typed_prompt(tmp_path):
    stop = threading.Event()
    thread = threading.Thread(
        target=run_demo,
        kwargs=dict(stream_dir=tmp_path, director=DemoDirector(seed=3), stop=stop, interval_s=0.4, poll_s=0.02),
        daemon=True,
    )
    thread.start()
    try:
        time.sleep(0.6)
        assert read_new_prompts(tmp_path, 0) == []  # nothing playing yet

        write_position(tmp_path, 10)
        deadline = time.monotonic() + 3
        while not read_new_prompts(tmp_path, 0) and time.monotonic() < deadline:
            time.sleep(0.02)
        first = read_new_prompts(tmp_path, 0)
        assert len(first) == 1 and "AMD" in first[0].text

        typed = append_prompt(tmp_path, "a typed prompt")
        time.sleep(0.25)
        assert read_new_prompts(tmp_path, typed.id) == []  # countdown restarted
        deadline = time.monotonic() + 3
        while not read_new_prompts(tmp_path, typed.id) and time.monotonic() < deadline:
            time.sleep(0.02)
        (demo,) = read_new_prompts(tmp_path, typed.id)
        assert demo.ts - typed.ts >= 0.4
    finally:
        stop.set()
        thread.join(timeout=2)
