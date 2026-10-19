"""One local HTTP API shared by the Web, CLI and Python client."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from wms.core.evaluation import compare_runs, evaluation_config, evaluations_for
from wms.core.registry import PROJECT_ROOT, Registry
from wms.core.runner import Runner
from wms.core.storage import Store


def request(method, path, body=None):
    import httpx
    base = os.environ.get("WMS_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
    local = base.startswith(("http://127.0.0.1", "http://localhost"))
    with httpx.Client(timeout=60, trust_env=not local) as client:
        response = client.request(method, base + "/api/v1" + path, json=body)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def create_app(storage_root=None, worker_python=None, web_dist=None, components=(), registries=(), native_configs=None):
    registry = Registry(paths=(*components, *registries), python=worker_python, native_configs=native_configs)
    root = Path(storage_root or os.environ.get("WMS_STORAGE_ROOT", PROJECT_ROOT / "data"))
    services = {}

    @asynccontextmanager
    async def lifespan(app):
        store = Store(root)
        runner = Runner(store)
        services.update(store=store, runner=runner)
        app.state.services = services
        try:
            yield
        finally:
            runner.shutdown()
            store.close()

    app = FastAPI(title="WM SafetyLab", version="0.3.0", lifespan=lifespan)

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc}"})

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "version": "0.3.0"}

    @app.get("/api/v1/recipes")
    def recipes():
        return {"items": registry.recipes()}

    @app.get("/api/v1/recipes/{recipe_id:path}")
    def recipe(recipe_id: str):
        return registry.recipe(recipe_id)

    @app.get("/api/v1/components")
    def components_list():
        return {key: values for key, values in registry.items.items() if key != "recipes"}

    @app.post("/api/v1/runs")
    def submit(config: dict = Body(...)):
        if config.get("task") == "evaluate":
            return evaluate(config["source_run"], config)
        return services["runner"].submit(registry.resolve(config))

    @app.get("/api/v1/runs")
    def runs():
        return {"items": services["store"].list_runs()}

    @app.get("/api/v1/runs/{run_id}")
    def run(run_id: str):
        return services["store"].get(run_id)

    @app.get("/api/v1/runs/{run_id}/logs")
    def logs(run_id: str, offset: int = 0):
        store = services["store"]
        store.get(run_id)
        path = store.run_dir(run_id) / "run.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        start = min(max(offset, 0), len(text))
        chunk = text[start:start + 200000]
        return {"text": chunk, "next_offset": start + len(chunk)}

    @app.get("/api/v1/runs/{run_id}/result")
    def result(run_id: str):
        return services["store"].get(run_id)["result"]

    @app.get("/api/v1/runs/{run_id}/files")
    def files(run_id: str):
        result = services["store"].get(run_id)["result"] or {}
        return {"items": result.get("outputs", [])}

    @app.get("/api/v1/runs/{run_id}/files/{path:path}")
    def file(run_id: str, path: str):
        store = services["store"]
        store.get(run_id)
        output = store.output_dir(run_id).resolve()
        target = (output / path).resolve()
        if not target.is_relative_to(output) or not target.is_file():
            raise HTTPException(404, "Output file not found")
        return FileResponse(target)

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return services["runner"].cancel(run_id)

    @app.post("/api/v1/runs/{run_id}/retry")
    def retry(run_id: str):
        return services["runner"].retry(run_id)

    @app.post("/api/v1/runs/{run_id}/evaluate")
    def evaluate(run_id: str, options: dict = Body(default={})):
        store = services["store"]
        runtime = options.get("runtime")
        if isinstance(runtime, dict) and runtime.get("config_file"):
            # Resolve the newly selected file before inheriting any source
            # fields, otherwise the source python would override that file.
            runtime = registry.resolve({"task": "evaluate", "runtime": runtime})["runtime"]
        config = evaluation_config(store, store.get(run_id), options.get("metrics"),
                                   runtime, options.get("timeout_s"))
        return services["runner"].submit(config)

    @app.get("/api/v1/runs/{run_id}/evaluations")
    def evaluations(run_id: str):
        return {"items": evaluations_for(services["store"], run_id)}

    @app.post("/api/v1/comparisons")
    def compare(options: dict = Body(...)):
        return compare_runs(services["store"], options["run_ids"], options.get("evaluations"))

    from wms.app.embodied_visual import create_embodied_visual_router
    app.include_router(create_embodied_visual_router(services))
    from wms.app.redteam_api import create_redteam_router
    app.include_router(create_redteam_router(services, registry))

    dist = Path(web_dist) if web_dist else PROJECT_ROOT / "web/dist"
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app
