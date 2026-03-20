FROM python:3.12-slim

WORKDIR /app

# Install torch CPU-only first (avoids pulling the large CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Chronos2 model so it's baked into the image.
# Remove this step if you prefer to mount a HuggingFace cache volume instead.
RUN python -c "from chronos import Chronos2Pipeline; Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')"

COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
