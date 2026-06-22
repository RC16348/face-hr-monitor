"""
Agnes AI API Client
Base URL: https://apihub.agnes-ai.com/v1
兼容 OpenAI API 格式
"""
import time
import json
import subprocess
from typing import Optional

DEFAULT_API_KEY = "你的密钥"  # 请替换为你的 Agnes AI API 密钥
BASE_URL = "https://apihub.agnes-ai.com/v1"

def _request(method: str, path: str, data: Optional[dict] = None, api_key: str = None):
    api_key = api_key or DEFAULT_API_KEY
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    args = ['curl.exe', '-s', '-X', method]
    for k, v in headers.items():
        args.extend(['-H', f'{k}: {v}'])
    tmp = None
    if data is not None:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        args.extend(['-d', f'@{tmp}'])
    args.append(url)

    try:
        result = subprocess.run(args, capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        if result.returncode != 0:
            raise RuntimeError(stderr or f'curl exit code {result.returncode}')
        return json.loads(stdout)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except Exception:
                pass

def chat(
    messages: list,
    model: str = "agnes-2.0-flash",
    stream: bool = False,
    temperature: float = 0.7,
    **kwargs
):
    """文本对话 - 兼容 OpenAI chat completions"""
    data = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        **kwargs
    }
    return _request("POST", "/chat/completions", data)

