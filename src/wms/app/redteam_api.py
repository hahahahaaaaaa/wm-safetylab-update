"""Red-team web API: catalog, validation, drafts, experiments and their results.

An experiment is a group of ordinary runs, one per (arm, environment); see
wms.redteam.experiments.
"""
import asyncio
import json

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from wms.redteam.experiments import ACTIVE, Experiments
from wms.redteam.schema import (Catalog, Draft, EpisodeDetail, EpisodePage, Experiment, ExperimentConfig, ExperimentPage,
                                LogChunk, Matrix, Report, Trace, ValidationResult)

TERMINAL = ("completed", "partial_failed", "failed", "cancelled")


def create_redteam_router(services, registry):
    router = APIRouter(prefix="/api/v1", tags=["redteam"])

    def lab():
        if "redteam" not in services:
            services["redteam"] = Experiments(services["store"], services["runner"], registry)
        return services["redteam"]

    @router.get("/catalog", response_model=Catalog)
    def catalog():
        return lab().catalog()

    @router.post("/experiments/validate", response_model=ValidationResult)
    def validate(config: ExperimentConfig):
        return lab().validate(config)

    @router.get("/experiments/drafts/{draft_id}", response_model=Draft)
    def get_draft(draft_id: str):
        return lab().get_draft(draft_id)

    @router.put("/experiments/drafts/{draft_id}", response_model=Draft)
    def put_draft(draft_id: str, config: ExperimentConfig):
        return lab().put_draft(draft_id, config)

    @router.post("/experiments", response_model=Experiment)
    def submit(config: ExperimentConfig):
        return lab().submit(config)

    @router.get("/experiments", response_model=ExperimentPage)
    def experiments():
        return {"items": lab().list()}

    @router.get("/experiments/{experiment_id}", response_model=Experiment)
    def experiment(experiment_id: str):
        return lab().get(experiment_id)

    @router.post("/experiments/{experiment_id}/cancel", response_model=Experiment)
    def cancel(experiment_id: str):
        return lab().cancel(experiment_id)

    @router.get("/experiments/{experiment_id}/logs", response_model=LogChunk)
    def logs(experiment_id: str, run_id: str, offset: int = 0):
        return lab().logs(experiment_id, run_id, offset)

    @router.get("/experiments/{experiment_id}/events",
                responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent RunEvent objects"}})
    async def events(experiment_id: str, interval: float = Query(1.0, ge=0.2, le=10)):
        await asyncio.to_thread(lab().get, experiment_id)          # 404 before the stream starts

        async def stream():
            last = None
            while True:
                value = await asyncio.to_thread(lab().get, experiment_id)
                data = {"state": value.state, "episodes_done": value.episodes_done,
                        "episodes_total": value.episodes_total,
                        "runs": [r.model_dump() for r in value.runs]}
                if data != last:
                    event = {"type": "progress", "experiment": experiment_id, "data": data}
                    yield f"event: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    last = data
                if value.state in TERMINAL and not any(r.state in ACTIVE for r in value.runs):
                    yield f"event: state\ndata: {json.dumps({'type': 'state', 'experiment': experiment_id, 'data': {'state': value.state}})}\n\n"
                    return
                await asyncio.sleep(interval)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.get("/experiments/{experiment_id}/episodes", response_model=EpisodePage)
    def episodes(experiment_id: str, arm: str | None = None, environment: str | None = None,
                 success: bool | None = None, end_reason: str | None = None, hazard: str | None = None,
                 cursor: str | None = None, limit: int = Query(100, ge=1, le=500)):
        return lab().episodes(experiment_id, arm, environment, success, end_reason, hazard, cursor, limit)

    @router.get("/experiments/{experiment_id}/episodes/{episode_id}", response_model=EpisodeDetail)
    def episode(experiment_id: str, episode_id: str):
        return lab().episode(experiment_id, episode_id)

    @router.get("/experiments/{experiment_id}/episodes/{episode_id}/trace", response_model=Trace)
    def trace(experiment_id: str, episode_id: str, start: int = Query(0, ge=0), end: int | None = Query(None, ge=0)):
        return lab().trace(experiment_id, episode_id, start, end)

    @router.get("/experiments/{experiment_id}/episodes/{episode_id}/frames/{camera}/{index}.png",
                responses={200: {"content": {"image/png": {}}, "description": "One frame as PNG"}}, response_class=Response)
    def frame(experiment_id: str, episode_id: str, camera: str, index: int, raw: bool = False):
        return Response(lab().frame_png(experiment_id, episode_id, camera, index, raw), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @router.get("/experiments/{experiment_id}/report", response_model=Report)
    def report(experiment_id: str):
        return lab().report(experiment_id)

    @router.get("/experiments/{experiment_id}/matrix", response_model=Matrix)
    def matrix(experiment_id: str):
        return lab().matrix(experiment_id)

    return router
