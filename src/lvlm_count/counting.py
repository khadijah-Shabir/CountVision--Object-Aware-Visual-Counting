"""Stage 4: LVLM clients, prompts, output parsing, retry logic."""
from __future__ import annotations

import base64
import io
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from text_to_num import alpha2digit

log = logging.getLogger("lvlm_count")

ZERO_SENTENCE = "Say 0 if you do not see any."
BRACKET_SENTENCE = "Report your final answer with only one number inside [[Double Brackets]]"
_BR = re.compile(r"\[\[(.+?)\]\]", re.S)


class ParseError(ValueError):
    pass


_LENIENT = re.compile(r"\s*\[?\s*([A-Za-z0-9,.\- ]{1,40}?)\s*\]?\s*\.?\s*", re.S)


def parse_count(text: str, lenient: bool = False) -> int:
    """Extract an integer from '[[12]]', '[[ twelve ]]', 'about [[12]] objects'. Raises ParseError.

    lenient=True additionally accepts a reply that is *only* a number or number-word, optionally in
    single brackets ('[5]', '5', 'five.'); anything else still fails. Used identically for every method.
    """
    m = _BR.findall(text or "")
    if not m:
        lm = _LENIENT.fullmatch(text or "") if lenient else None
        if not lm:
            raise ParseError(f"no [[...]] in {text!r}")
        m = [lm.group(1)]
    s = alpha2digit(m[-1].strip().lower(), "en")
    s = s.replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
            if f == int(f):
                return int(f)
        except (ValueError, OverflowError):   # '1e999' -> inf: not a usable count
            pass
        raise ParseError(f"not a number: {s!r}")


def build_prompt(question: str, zero_aware: bool = True, cot: bool = False) -> str:
    parts = [question.strip()]
    if zero_aware:
        parts.append(ZERO_SENTENCE)
    if cot:
        parts.append("Think step-by-step.")
    parts.append(BRACKET_SENTENCE)
    return " ".join(parts)


def to_pil(img) -> Image.Image:
    if isinstance(img, (str,)):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


class LVLM:
    """Interface: ask(image, prompt) -> raw text. Subclasses talk to a provider."""
    name = "base"

    def ask(self, image, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def ask_text(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicLVLM(LVLM):
    """Claude through the Anthropic API (needs ANTHROPIC_API_KEY). Default sampling."""

    def __init__(self, model="claude-sonnet-5-5", max_tokens=256):
        import anthropic
        self.client, self.model, self.max_tokens = anthropic.Anthropic(), model, max_tokens
        self.name = model

    def ask(self, image, prompt):
        buf = io.BytesIO()
        to_pil(image).save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}]}])
        return msg.content[0].text

    def ask_text(self, prompt):
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                          messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text


class ClaudeCLILVLM(LVLM):
    """Claude through the local `claude -p` CLI (uses the logged-in account; slow, quota-limited)."""

    def __init__(self, model="claude-sonnet-5-5", timeout=180):
        self.model, self.timeout, self.name = model, timeout, f"cli:{model}"

    def _run(self, prompt):
        r = subprocess.run(["claude", "-p", "--model", self.model, "--allowedTools", "Read",
                            "--permission-mode", "dontAsk"],
                           input=prompt, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode:
            raise RuntimeError(r.stderr[:300])
        return r.stdout

    def ask(self, image, prompt):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            to_pil(image).save(f.name)
            path = f.name
        return self._run(f"Look at the image file {path} (use the Read tool), then answer: {prompt}")

    def ask_text(self, prompt):
        return self._run(prompt)


class MockLVLM(LVLM):
    """For plumbing tests: `fn(image_array, prompt) -> int` returns the 'true' count."""
    name = "mock"

    def __init__(self, fn: Callable):
        self.fn = fn

    def ask(self, image, prompt):
        return f"[[{self.fn(np.asarray(to_pil(image)), prompt)}]]"


@dataclass
class CountResult:
    value: Optional[int]
    attempts: int
    failures: int
    raw: List[str] = field(default_factory=list)


def count_with_retry(lvlm: LVLM, image, question: str, *, zero_aware=True, cot=False,
                     max_retries: int = 10) -> CountResult:
    prompt = build_prompt(question, zero_aware, cot)
    fails, raws = 0, []
    for attempt in range(1, max_retries + 1):
        try:
            txt = lvlm.ask(image, prompt)
        except Exception as e:  # API error counts as a failed attempt
            log.warning("LVLM call failed: %s", e)
            fails += 1
            continue
        raws.append(txt)
        try:
            return CountResult(parse_count(txt), attempt, fails, raws)
        except ParseError as e:
            log.warning("parse failure (%d/%d): %s", attempt, max_retries, e)
            fails += 1
    return CountResult(None, max_retries, fails, raws)


def extract_expression(lvlm: LVLM, question: str) -> str:
    """Stage 0: noun phrase E from a counting question (few-shot)."""
    prompt = EXPRESSION_PROMPT + f"\nQuestion: {question}\nPhrase:"
    return lvlm.ask_text(prompt).strip().strip("\"'. \n").split("\n")[0]


EXPRESSION_PROMPT = """Extract only the noun phrase describing the objects that must be counted in the question. Output the phrase and nothing else.
Question: How many brown eggs are in the image?
Phrase: brown eggs
Question: How many cows and persons are there?
Phrase: cows and persons
Question: Count the clocks showing the time two-thirty.
Phrase: clocks at time two-thirty
Question: How many people are wearing a red hat?
Phrase: people wearing a red hat
Question: What is the number of penguins visible in this photo?
Phrase: penguins"""
