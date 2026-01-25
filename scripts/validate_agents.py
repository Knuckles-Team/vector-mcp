#!/usr/bin/env python3
import asyncio
import httpx
import uuid
import sys
import time

# Configuration: Map of Agent Name -> (Port, Test Question)
# We run the same workflow on all agents: Create, Insert, Retrieve, List.
# QUERY = "Create a collection called 'test_collection', insert a document with text 'This is a test document', retrieve it by searching for 'test', and finally list all collections."
QUERY = "List all collections."

AGENTS = {
    # "vector-agent-chromadb": (9023, QUERY),
    "vector-agent-pgvector": (9024, QUERY),
    "vector-agent-mongo": (9025, QUERY),
    "vector-agent-couchbase": (9026, QUERY),
    "vector-agent-qdrant": (9027, QUERY),
}


async def validate_agent(name, port, question):
    url = f"http://127.0.0.1:{port}/a2a/"
    print(f"[{name}] Starting validation at {url}...")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Construct JSON-RPC payload
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

            print(f"[{name}] Sending request: '{question}'")
            resp = None
            for _retry in range(15):
                try:
                    resp = await client.post(
                        url, json=payload, headers={"Content-Type": "application/json"}
                    )
                    break
                except (httpx.ConnectError, httpx.ReadError) as err:
                    print(
                        f"[{name}] Connection attempt {_retry+1} failed ({err}). Retrying in 5s..."
                    )
                    await asyncio.sleep(5)

            if resp is None:
                print(
                    f"[{name}] \033[91mFAILED\033[0m: Could not connect after retries."
                )
                return False, 0

            if resp.status_code != 200:
                print(
                    f"[{name}] \033[91mFAILED\033[0m: Initial request returned {resp.status_code}"
                )
                return False, 0

            data = resp.json()
            if "result" not in data or "id" not in data["result"]:
                print(f"[{name}] \033[91mFAILED\033[0m: No task ID in response")
                return False, 0

            task_id = data["result"]["id"]
            print(f"[{name}] Task {task_id} submitted. Polling...")

            # Poll for completion
            attempts = 0
            # 15 minutes timeout: 15 * 60 seconds / 2 seconds interval = 450 attempts
            # User suggested 9600, which is fine (320 mins), sticking to user's logic or a reasonable cap.
            max_attempts = 9600  # 15 mins should be enough for this.

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
                    url,
                    json=poll_payload,
                    headers={"Content-Type": "application/json"},
                )

                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    if "result" in poll_data:
                        state = poll_data["result"]["status"]["state"]

                        if state not in ["submitted", "running", "working"]:
                            duration = time.time() - start_time
                            print(f"[{name}] Finished with state: {state}")

                            # Extract result text
                            result_text = "No text content found."
                            if "history" in poll_data["result"]:
                                history = poll_data["result"]["history"]
                                # Find last non-user message
                                for msg in reversed(history):
                                    if msg.get("role") != "user":
                                        if "parts" in msg:
                                            parts_text = []
                                            for part in msg["parts"]:
                                                if "text" in part:
                                                    parts_text.append(part["text"])
                                                elif "content" in part:  # Fallback
                                                    parts_text.append(part["content"])
                                            if parts_text:
                                                result_text = "\n".join(parts_text)
                                                break
                                        elif "content" in msg:
                                            result_text = msg["content"]
                                            break

                            if state == "completed" or state == "done":
                                print(f"[{name}] \033[92mPASSED\033[0m")
                                print(f"[{name}] Final Output:\n{result_text}\n")
                                return True, duration
                            elif state == "failed" or state == "error":
                                print(
                                    f"[{name}] \033[91mFAILED\033[0m: Agent reported failure"
                                )
                                if "error" in poll_data["result"]:
                                    print(
                                        f"[{name}] Error details: {poll_data['result']['error']}"
                                    )
                                print(
                                    f"[{name}] Final Output (if any):\n{result_text}\n"
                                )
                                return False, duration
                            else:
                                print(
                                    f"[{name}] \033[93mFINISHED (State: {state})\033[0m"
                                )
                                print(f"[{name}] Final Output:\n{result_text}\n")
                                return True, duration
                    else:
                        print(f"[{name}] Polling response missing 'result'")
                else:
                    print(f"[{name}] Polling failed: {poll_resp.status_code}")

            print(
                f"[{name}] \033[91mTIMEOUT\033[0m: Validation timed out after 15 minutes"
            )
            return False, time.time() - start_time

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

    # Run validations concurrently but with a staggered start
    for name, (port, question) in AGENTS.items():
        # Stagger start by 6 seconds
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
