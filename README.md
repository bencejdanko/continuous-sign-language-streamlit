# Continuous Sign Language Streamlit

This is a setup for a demonstration of continuous sign language translation in realtime.

```
pip install -r requirements.txt

python app.py
```

The server starts at `http://localhost:8000`. Set `CAM_INDEX` env var to change camera (default `0`).

To connect your inference server, set the `INFERENCE_SERVER_URL` env var and implement the handler in the `/ws/translate` endpoint in `app.py`.