import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_case


def _json_default(value: Any) -> str:
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single retrieval query and print all returned case data."
    )
    parser.add_argument("query", help="Retrieval query string.")
    args = parser.parse_args()

    results = retrieve_case(args.query)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
