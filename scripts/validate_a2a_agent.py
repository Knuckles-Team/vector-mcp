#!/usr/bin/env python3
import asyncio
import json
import uuid

import httpx

A2A_URL = "http://audio-transcriber-agent.arpa/a2a/"


def _last_non_user_message(history):
    for msg in reversed(history):
        if msg.get("role") != "user":
            return msg
    return None


def _print_message_parts(last_msg):
    print("\n--- Agent Response ---")
    for part in last_msg["parts"]:
        if "text" in part:
            print(part["text"])
        elif "content" in part:
            print(part["content"])


def _print_agent_response(poll_data):
    result = poll_data["result"]
    history = result.get("history")
    if not history:
        return

    last_msg = _last_non_user_message(history)
    if last_msg and "parts" in last_msg:
        _print_message_parts(last_msg)
    elif last_msg:
        print(f"Final Message (No parts): {last_msg}")
    else:
        print("\n--- No Agent Response Found in History ---")


async def _poll_task(client, url, task_id):
    while True:
        await asyncio.sleep(2)
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
            print(f"Polling Failed: {poll_resp.status_code}")
            print(f"Polling Error Details: {poll_resp.text}")
            return None

        poll_data = poll_resp.json()
        if "result" not in poll_data:
            print("Starting polling error key check...")
            if "error" in poll_data:
                print(f"Polling Error: {poll_data['error']}")
            return None

        state = poll_data["result"]["status"]["state"]
        print(f"Task State: {state}")
        if state in ("submitted", "running", "working"):
            continue
        print(f"\nTask Finished with state: {state}")
        return poll_data


async def _handle_task_submission(client, url, data):
    task_id = data["result"]["id"]
    print(f"\nTask Submitted with ID: {task_id}. Polling for result...")
    poll_data = await _poll_task(client, url, task_id)
    if poll_data is None:
        return
    _print_agent_response(poll_data)
    print(f"Full Result Debug:\n{json.dumps(poll_data, indent=2)}")


async def _handle_json_response(client, url, resp):
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"Response (Text):\n{resp.text}")
        return

    print(f"Response (JSON):\n{json.dumps(data, indent=2)}")
    if "result" in data and "id" in data["result"]:
        await _handle_task_submission(client, url, data)
    if "error" in data:
        print(f"JSON-RPC Error: {data['error']}")


async def _ask_question(client, url, question):
    print(f"\n\n\nUser: {question}")
    print("--- Sending Request ---")

    payload = {
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

    try:
        print(f"Trying POST {url} with JSON-RPC (message/send)...")
        resp = await client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            await _handle_json_response(client, url, resp)
        else:
            print(f"Error: {resp.status_code}")
            print(resp.text)
    except httpx.RequestError as e:
        print(f"Connection failed to {url}: {e}")


async def main():
    print(f"Validating A2A Agent at {A2A_URL}...")

    questions = [
        "What tools do you have available?",
    ]

    async with httpx.AsyncClient(timeout=10000.0) as client:
        for q in questions:
            await _ask_question(client, A2A_URL, q)


if __name__ == "__main__":
    asyncio.run(main())
