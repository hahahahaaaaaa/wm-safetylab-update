"""Write the API contract to web/openapi.json for type generation.

Usage: python scripts/export_openapi.py [output]
"""
import json
import sys
import tempfile
from pathlib import Path

from wms.app.api import create_app

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web/openapi.json"
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(storage_root=tmp, web_dist=Path(tmp) / "no-dist")
        schema = app.openapi()
    output.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{output}: {len(schema['paths'])} paths, {len(schema['components']['schemas'])} schemas")
