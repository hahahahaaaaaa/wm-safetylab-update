"""Commands for the same API used by the Web and Python client."""
import json
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True, help="WM SafetyLab: submit configurations and inspect experiment outputs.")


def show(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command()
def serve(
    storage_root: Path = typer.Option(None),
    host: str = "127.0.0.1",
    port: int = 8765,
    worker_python: str = typer.Option(None),
    components: list[Path] = typer.Option(None, "--components", "--registry"),
    native_configs: Path = typer.Option(None, help="JSON mapping native engine names to runtime config paths"),
):
    import uvicorn
    from wms.app.api import create_app
    mapping = json.loads(native_configs.read_text(encoding="utf-8-sig")) if native_configs else None
    if mapping:
        base = native_configs.resolve().parent
        mapping = {key: str((base / value).resolve()) if not Path(value).is_absolute() else value
                   for key, value in mapping.items()}
    uvicorn.run(create_app(storage_root=storage_root, worker_python=worker_python,
                          components=components or (), native_configs=mapping), host=host, port=port)


@app.command()
def recipes():
    from wms.app.api import request
    show(request("GET", "/recipes"))


@app.command()
def components():
    from wms.app.api import request
    show(request("GET", "/components"))


@app.command()
def run(config: str, wait: bool = False):
    import wms
    path = Path(config)
    value = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {"recipe": config}
    record = wms.run(value)
    if wait:
        record = wms.wait(record["id"])
    show(record)
    if record["state"] in {"failed", "cancelled", "interrupted"}:
        raise typer.Exit(1)


@app.command("get-run")
def get_run(run_id: str):
    import wms
    show(wms.get_run(run_id))


@app.command()
def runs():
    from wms.app.api import request
    show(request("GET", "/runs"))


@app.command()
def logs(run_id: str):
    from wms.app.api import request
    typer.echo(request("GET", f"/runs/{run_id}/logs")["text"])


@app.command()
def cancel(run_id: str):
    import wms
    show(wms.cancel(run_id))


@app.command()
def retry(run_id: str):
    import wms
    show(wms.retry(run_id))


@app.command()
def evaluate(run_id: str, wait: bool = False, metrics: Path = typer.Option(None),
             runtime: Path = typer.Option(None), timeout_s: float = typer.Option(None)):
    import wms
    selected = json.loads(metrics.read_text(encoding="utf-8-sig")) if metrics else None
    environment = json.loads(runtime.read_text(encoding="utf-8-sig")) if runtime else None
    record = wms.evaluate(run_id, selected, runtime=environment, timeout_s=timeout_s)
    if wait:
        record = wms.wait(record["id"])
    show(record)
    if record["state"] in {"failed", "cancelled", "interrupted"} or record.get("evaluation_status") == "partial_failed":
        raise typer.Exit(1)


@app.command()
def compare(run_ids: list[str], evaluations: Path = typer.Option(None)):
    import wms
    selected = json.loads(evaluations.read_text(encoding="utf-8-sig")) if evaluations else None
    show(wms.compare(run_ids, evaluations=selected))


plugin_app = typer.Typer(no_args_is_help=True, help="S1-S6 plugins (wms.plugins).")
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("test")
def plugin_test(
    plugin: str = typer.Argument(..., help="registry ID or module:Class"),
    config: str = typer.Option("{}", help="JSON object, or a JSON file, merged over the registry config"),
    registry: list[Path] = typer.Option(None, "--registry", help="extra recipes.json / registry.json"),
    python: str = typer.Option(None, help="Python with numpy (default: native/toy/runtime.json python)"),
    output: Path = typer.Option(None),
):
    """Check a plugin against the contract on the CPU toy (python -m wms.plugins.conformance)."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[3]
    if python is None:
        runtime = root / "native/toy/runtime.json"
        python = json.loads(runtime.read_text(encoding="utf-8-sig")).get("python") if runtime.is_file() else sys.executable
    command = [python, "-m", "wms.plugins.conformance", plugin, "--config", config]
    for path in registry or []:
        command += ["--registry", str(path.resolve())]
    if output:
        command += ["--output", str(output.resolve())]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")]))}
    raise typer.Exit(subprocess.call(command, env=env))


if __name__ == "__main__":
    app()

