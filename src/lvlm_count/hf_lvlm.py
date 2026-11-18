"""Open-source LVLM backbones through HuggingFace transformers (fp16, V100-friendly)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .counting import LVLM, to_pil


def prepare_image(im: Image.Image, min_side: int = 56, max_aspect: float = 8.0) -> Image.Image:
    """White-pad degenerate images so the vision tower accepts them (Qwen rejects aspect ratio >= 200 and
    needs >= 2 patches per side). Content is unchanged; applied identically to every method (DECISIONS.md #35)."""
    w, h = im.size
    W, H = max(w, min_side), max(h, min_side)
    if W / H > max_aspect: H = int(np.ceil(W / max_aspect))
    if H / W > max_aspect: W = int(np.ceil(H / max_aspect))
    if (W, H) == (w, h): return im
    canvas = Image.new("RGB", (W, H), (255, 255, 255)); canvas.paste(im, (0, 0)); return canvas


class HFVLM(LVLM):
    """Qwen2-VL / Qwen2.5-VL (any AutoModelForImageTextToText chat model).

    gpus: list of physical-visible device indices; >1 shards the model with device_map='auto'.
    Sampling: temperature=1.0, top_p=1.0 (the authors' defaults, no greedy forcing).
    """

    def __init__(self, path: str, gpus=(0,), max_pixels: int = 1024 * 28 * 28, min_pixels: int = 16 * 28 * 28,
                 temperature: float = 1.0, top_p: float = 1.0, max_new_tokens: int = 64, sampling: str = "default"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.path, self.name = path, path.rstrip("/").split("/")[-1]
        self.temperature, self.top_p, self.max_new_tokens, self.sampling = temperature, top_p, max_new_tokens, sampling
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        kw = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
        if len(gpus) == 1:
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kw).to(f"cuda:{gpus[0]}")
        else:
            mm = {i: "30GiB" for i in gpus}
            self.model = AutoModelForImageTextToText.from_pretrained(path, device_map="auto", max_memory=mm, **kw)
        self.model.eval()
        self.dev = next(self.model.parameters()).device

    @torch.no_grad()
    def ask_batch(self, images, prompts, seed: int = 0):
        texts, imgs = [], []
        for im, p in zip(images, prompts):
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            texts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True))
            imgs.append(prepare_image(to_pil(im)))
        inp = self.processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(self.dev)
        torch.manual_seed(seed)
        if self.sampling == "t1":      # pure sampling T=1.0 (authors' --model_temperature/--model_top_p), no truncation
            out = self.model.generate(**inp, do_sample=True, temperature=self.temperature, top_p=self.top_p,
                                      top_k=0, max_new_tokens=self.max_new_tokens)
        else:                          # the model's shipped generation_config (provider default)
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)

    def ask(self, image, prompt):
        return self.ask_batch([image], [prompt])[0]

    @torch.no_grad()
    def ask_text(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        t = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=[t], return_tensors="pt").to(self.dev)
        out = self.model.generate(**inp, do_sample=False, max_new_tokens=32)
        return self.processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
