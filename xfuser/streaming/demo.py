"""Demo mode: queue a new scripted prompt whenever the stream has been quiet for a while.

Each demo prompt changes the cast by one character (a random animal or person
walks in; once three demo characters are on screen, one of them leaves as the
newcomer arrives), has the cast promote an AMD product (a CPU, a graphics card,
a laptop or a server), and picks a random storyline and visual style. The text
goes through the same queue as a typed prompt, so the generator enhances it and
cuts it in exactly like viewer input.

Typed prompts keep working: each one restarts the countdown, so the next demo
prompt comes ``interval_s`` after the latest prompt of either kind.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from xfuser.streaming.chunks import read_position
from xfuser.streaming.prompt_queue import UserPrompt, append_prompt, latest_prompt

MAX_CAST = 3

ANIMALS = (
    "a red fox",
    "a golden retriever",
    "a penguin",
    "a giant panda",
    "a raccoon",
    "a snowy owl",
    "a corgi",
    "a koala",
    "a lion cub",
    "a sea otter",
    "a pink flamingo",
    "a hedgehog",
    "a chameleon",
    "a husky",
    "a capybara",
)

HUMANS = (
    "a young gamer in a hoodie",
    "a data-center engineer in a hard hat",
    "a grandmother in a knitted cardigan",
    "a street dancer in a tracksuit",
    "a chef in a white jacket",
    "a scientist in a lab coat",
    "an astronaut in a white spacesuit",
    "a skateboarder with a backwards cap",
    "a businesswoman in a sharp suit",
    "a DJ wearing big headphones",
    "a kid with a school backpack",
    "a rock guitarist in a leather jacket",
)

PRODUCTS = {
    "CPU": (
        "an AMD Ryzen desktop processor",
        "an AMD Ryzen processor in its retail box",
        "an AMD Ryzen Threadripper processor",
    ),
    "graphics card": (
        "an AMD Radeon RX graphics card",
        "an AMD Radeon graphics card with glowing red fans",
    ),
    "laptop": (
        "a slim laptop powered by AMD Ryzen AI",
        "a gaming laptop with AMD Ryzen and Radeon inside",
    ),
    "server": (
        "an AMD EPYC server",
        "a server rack packed with AMD Instinct accelerators",
    ),
}

# Each follows "... promote <product> in", with "it" for the product, and reads
# correctly for one character or for a group.
STORYLINES = (
    "a surprise unboxing of it; everyone gasps at what is inside",
    "a race to be the first to plug it in, ending in a big high five",
    "a TV commercial for it, with a director shouting 'action'",
    "a stage show where it is unveiled to a cheering crowd under spotlights",
    "a playful tug-of-war over it that ends with everyone sharing it",
    "a late-night PC build around it that ends in cheers when it boots up",
    "a trade-show booth where it is shown off to curious visitors",
    "a victory parade with it held high like a trophy",
    "a video game tournament won thanks to it",
    "a rescue mission to carry it out of a storm and keep it dry",
    "a cooking-show parody where it is the secret ingredient",
    "a museum tour where it is the star exhibit behind glass",
)

STYLES = (
    "Pixar-like 3D animation",
    "hand-drawn watercolor anime",
    "stop-motion claymation",
    "1980s synthwave with neon glow",
    "black-and-white silent film",
    "comic book with bold ink outlines",
    "cyberpunk city at night",
    "toy brick animation",
    "paper cut-out animation",
    "cinematic live action",
    "low-poly video game",
    "impressionist oil painting",
)


@dataclass(frozen=True)
class Character:
    name: str
    kind: str  # "animal" or "human"


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _the(name: str) -> str:
    """'a red fox' -> 'the red fox', for characters already introduced."""
    for article in ("a ", "an "):
        if name.startswith(article):
            return "the " + name[len(article) :]
    return name


class _ShuffleBag:
    """Random picks that use every option once before any repeats."""

    def __init__(self, options: Sequence[str], rng: random.Random) -> None:
        self.options = list(options)
        self.rng = rng
        self._left: list[str] = []

    def draw(self) -> str:
        if not self._left:
            self._left = self.options[:]
            self.rng.shuffle(self._left)
        return self._left.pop()


class DemoDirector:
    """Writes the scripted prompts and keeps track of the demo cast."""

    def __init__(self, cast: Sequence[str] = (), seed: int | None = None, max_cast: int = MAX_CAST) -> None:
        self.rng = random.Random(seed)
        self.max_cast = max_cast
        self.cast: list[Character] = [Character(name, "given") for name in cast][:max_cast]
        self._categories = _ShuffleBag(sorted(PRODUCTS), self.rng)
        self._storylines = _ShuffleBag(STORYLINES, self.rng)
        self._styles = _ShuffleBag(STYLES, self.rng)

    def _new_character(self) -> Character:
        taken = {c.name for c in self.cast}
        kind = self.rng.choice(("animal", "human"))
        pool = [name for name in (ANIMALS if kind == "animal" else HUMANS) if name not in taken]
        return Character(self.rng.choice(pool), kind)

    def next_prompt(self) -> str:
        newcomer = self._new_character()
        if len(self.cast) >= self.max_cast:
            leaver = self.cast.pop(self.rng.randrange(len(self.cast)))
            stays = _join([_the(c.name) for c in self.cast])
            change = (
                f"{_cap(_the(leaver.name))} waves goodbye and leaves the scene, "
                f"and {newcomer.name} walks in to join {stays}."
            )
        elif self.cast:
            change = f"{_cap(newcomer.name)} walks into the scene and joins {_join([_the(c.name) for c in self.cast])}."
        else:
            change = f"{_cap(newcomer.name)} walks into the scene."
        self.cast.append(newcomer)

        product = self.rng.choice(PRODUCTS[self._categories.draw()])
        storyline = self._storylines.draw()
        style = self._styles.draw()

        who = _join([_the(c.name) for c in self.cast])
        verb = "promote" if len(self.cast) > 1 else "promotes"
        return f"{change} {_cap(who)} {verb} {product} in {storyline}. Style: {style}."


def run_demo(
    stream_dir: Path,
    director: DemoDirector,
    stop: threading.Event,
    interval_s: float = 60.0,
    on_prompt: Callable[[UserPrompt], None] | None = None,
    lock: threading.Lock | None = None,
    poll_s: float = 1.0,
) -> None:
    """Queue ``director.next_prompt()`` each time no prompt has arrived for ``interval_s``.

    The countdown starts once the streamer is playing content, so the generator's
    multi-minute model load and compile do not eat into it.
    """
    lock = lock or threading.Lock()
    playing_since = None
    while not stop.wait(poll_s):
        if playing_since is None:
            if read_position(stream_dir) <= 0:
                continue
            playing_since = time.time()
        latest = latest_prompt(stream_dir)
        last = max(playing_since, latest.ts if latest else 0.0)
        if time.time() - last < interval_s:
            continue
        with lock:
            prompt = append_prompt(stream_dir, director.next_prompt())
        if on_prompt is not None:
            on_prompt(prompt)
