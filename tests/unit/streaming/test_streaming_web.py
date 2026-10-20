import pytest

from xfuser.streaming.prompt_queue import read_new_prompts, read_overlay
from xfuser.streaming.web import Mp4FanOut, codecs_from_init, submit


def _box(kind: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


INIT = _box(b"ftyp", b"isom") + _box(b"moov", b"\x00" * 20)


def _fragment(tag: bytes) -> bytes:
    return _box(b"moof", tag) + _box(b"mdat", tag * 3)


def test_init_segment_and_fragments_survive_split_writes():
    fan_out = Mp4FanOut()
    stream = INIT + _fragment(b"a") + _fragment(b"b")
    fan_out.write(stream[:5])
    fan_out.write(stream[5 : len(INIT)])
    client, init = fan_out.add_client(timeout=1)
    assert init == INIT

    for byte in stream[len(INIT) :]:
        fan_out.write(bytes([byte]))
    assert client.get_nowait() == _fragment(b"a")
    assert client.get_nowait() == _fragment(b"b")
    assert fan_out.fragments == 2


def test_a_viewer_only_gets_fragments_published_after_it_joined():
    fan_out = Mp4FanOut()
    fan_out.write(INIT + _fragment(b"a"))
    client, _ = fan_out.add_client(timeout=1)
    fan_out.write(_fragment(b"b"))

    assert client.get_nowait() == _fragment(b"b")
    assert client.empty()


def test_a_viewer_that_falls_behind_is_dropped():
    fan_out = Mp4FanOut(max_queued_fragments=2)
    fan_out.write(INIT)
    slow, _ = fan_out.add_client(timeout=1)
    for tag in (b"a", b"b", b"c"):
        fan_out.write(_fragment(tag))

    assert fan_out.num_clients == 0
    assert slow.get_nowait() is None


def test_close_ends_every_viewer_after_its_queued_fragments():
    fan_out = Mp4FanOut()
    fan_out.write(INIT)
    client, _ = fan_out.add_client(timeout=1)
    fan_out.write(_fragment(b"last"))
    fan_out.close()

    assert client.get_nowait() == _fragment(b"last")
    assert client.get_nowait() is None
    assert fan_out.num_clients == 0


def test_large_size_boxes_and_malformed_boxes():
    fan_out = Mp4FanOut()
    fan_out.write(INIT)
    client, _ = fan_out.add_client(timeout=1)
    payload = b"x" * 10
    large_mdat = (1).to_bytes(4, "big") + b"mdat" + (16 + len(payload)).to_bytes(8, "big") + payload
    fan_out.write(_box(b"moof") + large_mdat)
    assert client.get_nowait() == _box(b"moof") + large_mdat

    with pytest.raises(ValueError):
        fan_out.write((4).to_bytes(4, "big") + b"free")


def test_add_client_times_out_before_the_init_segment():
    assert Mp4FanOut().add_client(timeout=0.01) is None


def test_codecs_string_comes_from_the_avc_config():
    avcc = b"avcC" + bytes([1, 0x64, 0x00, 0x1F]) + b"\xff"
    assert codecs_from_init(b"..." + avcc + b"...mp4a...") == "avc1.64001f,mp4a.40.2"
    assert codecs_from_init(b"no config") == "avc1.640028"


def test_submit_queues_web_prompts_and_runs_overlay(tmp_path):
    assert submit(tmp_path, "   ")[0] == 400
    assert submit(tmp_path, "/nope")[0] == 400

    code, body = submit(tmp_path, "  a red panda  joins the stage ")
    assert code == 200
    assert body["id"] == 1
    (prompt,) = read_new_prompts(tmp_path, 0)
    assert (prompt.text, prompt.source) == (body["text"], "web")

    assert submit(tmp_path, "/overlay") == (200, {"overlay": True})
    assert read_overlay(tmp_path) is True
    assert submit(tmp_path, "/overlay") == (200, {"overlay": False})
