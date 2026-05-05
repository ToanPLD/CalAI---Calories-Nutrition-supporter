import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.agentic_rag import AgenticRAG


async def main():
    agent = AgenticRAG()
    result = await agent.run(
        "Gợi ý bữa trưa giàu protein, ít carb và trình bày dạng bảng",
        top_k=5,
        intent="meal_planning",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
