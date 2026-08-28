#!/usr/bin/env python3
import asyncio
import sys
import time
import uuid

import httpx

QUERY = "List all collections."

AGENTS = {
    "vector-agent-postgres": (9024, QUERY),
    "vector-agent-mongo": (9025, QUERY),
    "vector-agent-couchbase": (9026, QUERY),
    "vector-agent-qdrant": (9027, QUERY),
}


def _initial_request_payload(question):
    return {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "parts": [{"kind": "text", "text": question}],
                "messageId": str(uuid.uuid4()),
            }
        },
        "id": 1,
    }


async def _send_initial_request(client, name, url, question):
    print(f"[{name}] Sending request: '{question}'")
    payload = _initial_request_payload(question)
    resp = None
    for retry in range(15):
        try:
            resp = await client.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )
            break
        except (httpx.ConnectError, httpx.ReadError) as err:
            print(
                f"[{name}] Connection attempt {retry + 1} failed ({err}). Retrying in 5s..."
            )
            await asyncio.sleep(5)
    return resp


def _extract_task_id(name, resp):
    if resp is None:
        print(f"[{name}] \033[91mFAILED\033[0m: Could not connect after retries.")
        return None
    if resp.status_code != 200:
        print(
            f"[{name}] \033[91mFAILED\033[0m: Initial request returned {resp.status_code}"
        )
        return None
    data = resp.json()
    if "result" not in data or "id" not in data["result"]:
        print(f"[{name}] \033[91mFAILED\033[0m: No task ID in response")
        return None
    task_id = data["result"]["id"]
    print(f"[{name}] Task {task_id} submitted. Polling...")
    return task_id


def _text_from_message_parts(msg):
    parts_text = [
        part["text"] if "text" in part else part["content"]
        for part in msg.get("parts", [])
        if "text" in part or "content" in part
    ]
    return "\n".join(parts_text) if parts_text else None


def _extract_result_text(poll_result):
    history = poll_result.get("history")
    if not history:
        return "No text content found."
    for msg in reversed(history):
        if msg.get("role") == "user":
            continue
        if "parts" in msg:
            text = _text_from_message_parts(msg)
            if text:
                return text
        elif "content" in msg:
            return msg["content"]
    return "No text content found."


def _report_outcome(name, state, poll_result, start_time):
    duration = time.time() - start_time
    print(f"[{name}] Finished with state: {state}")
    result_text = _extract_result_text(poll_result)

    if state in ("completed", "done"):
        print(f"[{name}] \033[92mPASSED\033[0m")
        print(f"[{name}] Final Output:\n{result_text}\n")
        return True, duration
    if state in ("failed", "error"):
        print(f"[{name}] \033[91mFAILED\033[0m: Agent reported failure")
        if "error" in poll_result:
            print(f"[{name}] Error details: {poll_result['error']}")
        print(f"[{name}] Final Output (if any):\n{result_text}\n")
        return False, duration
    print(f"[{name}] \033[93mFINISHED (State: {state})\033[0m")
    print(f"[{name}] Final Output:\n{result_text}\n")
    return True, duration


async def _poll_task(client, name, url, task_id, start_time):
    attempts = 0
    max_attempts = 9600

    while attempts < max_attempts:
        await asyncio.sleep(2)
        attempts += 1

        poll_payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": 2,
        }
        poll_resp = await client.post(
            url, json=poll_payload, headers={"Content-Type": "application/json"}
        )

        if poll_resp.status_code != 200:
            print(f"[{name}] Polling failed: {poll_resp.status_code}")
            continue
        poll_data = poll_resp.json()
        if "result" not in poll_data:
            print(f"[{name}] Polling response missing 'result'")
            continue
        state = poll_data["result"]["status"]["state"]
        if state in ("submitted", "running", "working"):
            continue
        return _report_outcome(name, state, poll_data["result"], start_time)

    print(f"[{name}] \033[91mTIMEOUT\033[0m: Validation timed out after 15 minutes")
    return False, time.time() - start_time


async def validate_agent(name, port, question):
    url = f"http://127.0.0.1:{port}/a2a/"
    print(f"[{name}] Starting validation at {url}...")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await _send_initial_request(client, name, url, question)
            task_id = _extract_task_id(name, resp)
            if task_id is None:
                return False, 0
            return await _poll_task(client, name, url, task_id, start_time)
        except httpx.ConnectError:
            print(
                f"[{name}] \033[91mFAILED\033[0m: Connection refused (is the container running?)"
            )
            return False, 0
        except Exception as e:
            print(f"[{name}] \033[91mERROR\033[0m: {repr(e)}")
            return False, 0


async def main():
    print("Starting A2A Agent Validation...")
    print("--------------------------------")

    results = {}
    tasks = []

    for name, (port, question) in AGENTS.items():
        await asyncio.sleep(6)
        task = asyncio.create_task(validate_agent(name, port, question))
        tasks.append((name, task))

    print("\nAll tasks submitted. Waiting for results...\n")

    for name, task in tasks:
        results[name] = await task

    print("\n--------------------------------")
    print("Validation Summary:")
    passed = 0
    for name, (success, duration) in results.items():
        status = "\033[92mPASS\033[0m" if success else "\033[91mFAIL\033[0m"
        duration_str = f"{duration:.2f}s"
        print(f"{name:<25} {status} ({duration_str})")
        if success:
            passed += 1

    print(f"\nTotal: {passed}/{len(AGENTS)} Agents Passed")

    if passed < len(AGENTS):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
