#!/usr/bin/env python3
"""
Test cases for Cal-AI pipeline: Qdrant collections, user profile, vision timeout.
Run: cd Cal-AI && ./venv/bin/python scripts/test_pipeline.py
"""
import asyncio
import httpx
import json
import os
import sys
import time

# Ensure Cal-AI modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_URL = "http://localhost:8001"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
results = []


def report(name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append(ok)
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


async def test_qdrant_collections():
    print("\n=== Test 1: Qdrant collection connectivity ===")
    try:
        from config.settings import settings
        from core.services.retrieval.qdrant_service import QdrantService
        qdrant = QdrantService()
        available = qdrant.available_collections()
        report("Qdrant reachable", True, f"{len(available)} collections")

        expected = ["food_nutrition_vectors_768", "food_common_vectors_768"]
        for col in expected:
            found = col in available
            report(f"Collection '{col}' exists", found)

        from core.agent.agentic_rag import GenericRAGAgent
        agent = GenericRAGAgent(qdrant=qdrant)
        text_cols = agent._text_collections()
        report("Text collections resolved", len(text_cols) > 0, f"{len(text_cols)} collections")
    except Exception as e:
        report("Qdrant connection", False, str(e))


async def test_text_query():
    print("\n=== Test 2: Text query (Agentic RAG) ===")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(f"{BASE_URL}/api/agent/query", json={
                "question": "How many calories in a banana?",
                "top_k": 4
            })
            report("HTTP status 200", res.status_code == 200, f"got {res.status_code}")
            data = res.json()
            answer = data.get("answer", "")
            report("Answer not empty", bool(answer), answer[:80])
            report("Has results", len(data.get("results", [])) > 0,
                   f"{len(data.get('results', []))} results")
            report("Intent classified", data.get("intent") == "nutrition_qa",
                   f"intent={data.get('intent')}")
    except Exception as e:
        report("Text query", False, str(e))


async def test_user_profile():
    print("\n=== Test 3: User profile passed to Cal-AI ===")
    try:
        profile = {
            "gender": "male",
            "age": 25,
            "height": 175,
            "weight": 70,
            "dailyCalories": 2200,
            "goal": "muscle_gain",
            "activityLevel": "moderate"
        }
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(f"{BASE_URL}/api/agent/query", json={
                "question": "Plan my lunch",
                "top_k": 6,
                "user_profile": profile
            })
            report("HTTP status 200", res.status_code == 200)
            data = res.json()

            trace_titles = [s.get("title") for s in data.get("trace", [])]
            has_profile_step = "User profile" in trace_titles
            report("Trace includes user profile step", has_profile_step,
                   f"trace steps: {trace_titles[:5]}")

            answer = data.get("answer", "")
            report("Answer not empty", bool(answer), answer[:80])

        # Test without profile
        res2 = await httpx.AsyncClient(timeout=120).post(
            f"{BASE_URL}/api/agent/query",
            json={"question": "Plan my lunch", "top_k": 6}
        )
        data2 = res2.json()
        trace_titles2 = [s.get("title") for s in data2.get("trace", [])]
        no_profile_step = "User profile" not in trace_titles2
        report("No profile step when profile absent", no_profile_step)

    except Exception as e:
        report("User profile test", False, str(e))


async def test_vision_timeout_handling():
    print("\n=== Test 4: Vision endpoint error handling ===")
    try:
        # Test with invalid image (should return 400, not hang)
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{BASE_URL}/api/food/analyze",
                files={"file": ("test.txt", b"not an image", "text/plain")}
            )
            report("Rejects non-image file", res.status_code == 400,
                   f"status={res.status_code}")

        # Test with a tiny valid JPEG (1x1 pixel)
        import struct
        # Minimal JPEG: SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI
        # Use a simpler approach: create via PIL if available
        from PIL import Image
        from io import BytesIO
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        start = time.time()
        async with httpx.AsyncClient(timeout=180) as client:
            res = await client.post(
                f"{BASE_URL}/api/food/analyze",
                files={"file": ("test_food.jpg", jpeg_bytes, "image/jpeg")},
                data={"question": "What is this?"}
            )
            elapsed = time.time() - start
            report("Vision endpoint responds", res.status_code == 200,
                   f"status={res.status_code}, {elapsed:.1f}s")
            if res.status_code == 200:
                data = res.json()
                report("Has dish_name field", "dish_name" in data,
                       f"dish={data.get('dish_name', 'N/A')}")
                report("Has confidence field", "confidence" in data)

    except httpx.ReadTimeout:
        report("Vision endpoint timed out", False, "exceeded 180s client timeout")
    except Exception as e:
        report("Vision error handling", False, str(e))


async def test_no_context_friendly_response():
    print("\n=== Test 5: Friendly response when no context found ===")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(f"{BASE_URL}/api/agent/query", json={
                "question": "xyznonexistentfood12345",
                "top_k": 4
            })
            data = res.json()
            answer = data.get("answer", "")
            is_friendly = len(answer) > 30 and not answer.startswith("{")
            report("Returns friendly text (not JSON/empty)", is_friendly, answer[:100])

            has_suggestion = any(word in answer for word in [
                "th\u1eed", "m\u00f4 t\u1ea3", "c\u1ee5 th\u1ec3",
                "ti\u1ebfng Anh", "English", "s\u1eb5n s\u00e0ng",
                "cung c\u1ea5p", "cho m\u00ecnh bi\u1ebft",
                "g\u1eedi l\u1ea1i", "b\u1ed5 sung",
            ])
            report("Contains helpful suggestions", has_suggestion)
    except Exception as e:
        report("Friendly response", False, str(e))


async def main():
    print("=" * 60)
    print("Cal-AI Pipeline Test Suite")
    print("=" * 60)

    # Check Cal-AI is running
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{BASE_URL}/docs")
            if res.status_code != 200:
                print(f"  [{FAIL}] Cal-AI not running on {BASE_URL}")
                sys.exit(1)
    except Exception:
        print(f"  [{FAIL}] Cal-AI not reachable at {BASE_URL}")
        print("  Start it first: cd Cal-AI && ./venv/bin/python -m uvicorn api.main:app --port 8001")
        sys.exit(1)

    await test_qdrant_collections()
    await test_text_query()
    await test_user_profile()
    await test_vision_timeout_handling()
    await test_no_context_friendly_response()

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print(f"[{PASS}] All tests passed!")
    else:
        print(f"[{FAIL}] {total - passed} test(s) failed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
