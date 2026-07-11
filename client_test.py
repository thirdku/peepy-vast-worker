"""One-shot test client for the peepy-forge serverless endpoint.

Usage (from any machine, NOT the worker):
    pip install "vastai-sdk>=0.3.0" requests
    export VAST_API_KEY=...        # PowerShell: $env:VAST_API_KEY="..."
    python client_test.py [endpoint-name]     # default: peepy-forge

Sends one fixed-seed peepy-style render (same payload as the worker's
benchmark) and writes test_out.png. Run it twice: identical bytes back
proves seed determinism on the worker.
"""

import asyncio
import base64
import json
import sys

from vastai import Serverless

from worker import make_benchmark_payload, workload_calculator

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "peepy-forge"


async def main() -> None:
    payload = make_benchmark_payload()
    payload["seed"] = 123456789  # fixed for reproducibility checks
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name=ENDPOINT)
        result = await endpoint.request(
            "/sdapi/v1/txt2img", payload, cost=workload_calculator(payload)
        )
    resp = result["response"]
    images = resp.get("images") or []
    if not images:
        print("NO IMAGE — raw response:")
        print(json.dumps(resp)[:2000])
        sys.exit(1)
    with open("test_out.png", "wb") as f:
        f.write(base64.b64decode(images[0]))
    info = resp.get("info", "")
    print("saved test_out.png")
    print("info:", info[:300])


asyncio.run(main())
