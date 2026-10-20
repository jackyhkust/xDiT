from xfuser.streaming.chunks import read_chunk
from xfuser.streaming.fake_generator import main
from xfuser.streaming.prompt_queue import append_prompt, read_prompt_state


def test_fake_generator_writes_chunks_and_applies_queued_prompts(tmp_path):
    append_prompt(tmp_path, "a robot joins", source="web")
    main(
        [
            "--stream_dir", str(tmp_path), "--width", "64", "--height", "48", "--num_frames", "9",
            "--sample_rate", "8000", "--gen_s", "0", "--enhance_s", "0", "--num_chunks", "2",
        ]
    )  # fmt: skip

    first, second = read_chunk(tmp_path, 0), read_chunk(tmp_path, 1)
    assert first.frames.shape == (9, 48, 64, 3)
    assert first.audio.shape[0] == 2 and first.audio.shape[1] >= 9 * 8000 // 24
    assert first.meta["prompt_id"] == 1
    assert "prompt_id" not in second.meta
    assert read_prompt_state(tmp_path)["status"] == "rendering"
    assert (tmp_path / "done").exists()
