import os
import time

import psutil
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from chronos import Chronos2Pipeline

CONTEXT_LENGTH = 1024

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

app = FastAPI()


class ForecastRequest(BaseModel):
    data: list[list[float]]
    target: int
    length: int


_proc = psutil.Process(os.getpid())


@app.post("/forecast")
def forecast(req: ForecastRequest):
    mem_before = _proc.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    data = [series[-CONTEXT_LENGTH:] for series in req.data]
    tensor = torch.tensor(data).float().unsqueeze(0)
    samples = pipeline.predict(tensor, prediction_length=req.length)
    median = torch.median(samples[0][req.target], dim=0).values

    elapsed = time.perf_counter() - t0
    mem_after = _proc.memory_info().rss / 1024**2
    print(f"time: {elapsed:.3f}s | RAM: {mem_before:.1f} → {mem_after:.1f} MB (Δ{mem_after - mem_before:+.1f} MB)")

    return {"forecast": median.tolist()}
