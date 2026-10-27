"""The catalog and shipped assets must agree, including task variants."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from wms.app.api import create_app
from wms.redteam.schema import EnvironmentPreview

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web/public"


def test_catalog_previews_are_shipped_and_reachable(tmp_path):
    # Serve the assets through the same StaticFiles mount as a production build.
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>")
    import shutil
    shutil.copytree(PUBLIC / "task-previews", web / "task-previews")
    with TestClient(create_app(storage_root=tmp_path / "store", web_dist=web)) as client:
        environments = client.get("/api/v1/catalog").json()["environments"]
        manifest = client.get("/task-previews/manifest.json").json()
        assert len(environments) == 12
        for env in environments:
            preview = env["preview"]
            assert preview and preview["caption"]
            response = client.get(preview["url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"
            assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
            item = manifest["environments"][env["id"]]
            assert item["url"] == preview["url"]
            assert item["sha256"] == hashlib.sha256(response.content).hexdigest()


def test_variant_preview_sources_are_explicit():
    meta = json.loads((ROOT / "src/wms/redteam/catalog_meta.json").read_text(encoding="utf-8"))
    for env, source in (("toy-pick-vector", "toy-pick"),
                        ("maniskill-pickcube-rgb-ee", "maniskill-pickcube-rgb"),
                        ("maniskill-pushcube-state", "maniskill-pushcube-state-independent")):
        assert meta["environments"][env]["preview"]["source_environment"] == source
    assert meta["environments"]["toy-pick"]["preview"]["kind"] == "illustration"


@pytest.mark.parametrize("url", ["https://example.com/image.png", "/task-previews/../private.png",
                                "/task-previews/a.svg", "//example.com/a.png"])
def test_preview_contract_rejects_external_or_nonimage_paths(url):
    with pytest.raises(ValidationError):
        EnvironmentPreview(url=url, kind="task_asset", caption="Example")
