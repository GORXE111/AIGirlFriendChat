"""HTTP 服务：聊天窗口 + agent API。

前端是单页，轮询 `/api/saves/{id}/poll` 取到点的消息 —— **延迟是设计的一部分**，
不是要消除的摩擦。调试时用 `DELAY_SCALE` 压缩时间。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..agent import Agent
from ..beats import get_beat, load_beats
from ..config import get_settings
from ..llm import DeepSeekProvider, LLMError
from ..memory import Reflector
from ..metrics import UsageRecorder
from ..persona import load_card
from ..presets import PRESETS, listing, seed
from ..schedule import ScheduleEngine
from ..state import STAGE_BEHAVIOR, EmotionState, Stage
from ..storage import Database, parse_ts
from ..timewindow import is_peak, now_beijing

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

_db: Database | None = None
_agent: Agent | None = None
_reflector: Reflector | None = None
_recorder: UsageRecorder | None = None
_provider: DeepSeekProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _agent, _reflector, _recorder, _provider
    s = get_settings()
    logging.basicConfig(level=s.log_level)

    _db = Database(s.db_path)
    _recorder = UsageRecorder(s.usage_log_path)
    _provider = DeepSeekProvider(s)
    _agent = Agent(
        _db, _provider, _recorder, ScheduleEngine(tz=s.story_timezone),
        delay_scale=s.delay_scale,
        max_delay_seconds=s.max_delay_seconds,
    )
    _reflector = Reflector(_db, _provider, _recorder)

    if s.delay_scale != 1.0:
        log.warning("DELAY_SCALE=%s —— 时间被压缩，仅供调试", s.delay_scale)

    try:
        yield
    finally:
        await _provider.aclose()


app = FastAPI(title="GirlFriendAgent", version="0.2.0", lifespan=lifespan)


def deps() -> tuple[Database, Agent, Reflector]:
    if _db is None or _agent is None or _reflector is None:
        raise HTTPException(503, "服务未就绪")
    return _db, _agent, _reflector


# ---------------- 模型 ----------------


class CreateSave(BaseModel):
    name: str = Field("新存档", max_length=40)
    surname: str = Field("", max_length=4)
    given: str = Field("", max_length=8)
    character_id: str = "h01"
    preset: str = Field("s0", max_length=16, description="测试预设，见 /api/presets")


class Choose(BaseModel):
    index: int = Field(..., ge=0, le=9)


class StartBeat(BaseModel):
    beat_id: str = Field(..., max_length=64)


def _turn_payload(r) -> dict[str, Any]:
    d = r.completion
    return {
        "queued": [
            {"content": t, "deliver_at": w.isoformat()} for t, w in r.scheduled
        ],
        "options": [o.as_dict() for o in r.options],
        "topics": [t.as_dict() for t in r.topics],
        "beat": {
            "id": r.beat_id, "title": r.beat_title,
            "turn": r.beat_turn, "finished": r.beat_finished,
        },
        "stage": r.stage.value,
        "affinity": round(r.affinity, 1),
        "emotion_note": r.emotion_note,
        "delay_seconds": r.delay_seconds,
        "diagnostics": {
            "raw": r.raw_text,
            "violations": r.violations,
            "cleaned": r.cleaned,
            "retries": r.retries,
            "used_fallback": r.used_fallback,
            "latency_ms": d.latency_ms if d else None,
            "cache_hit_rate": round(d.usage.cache_hit_rate, 3) if d else None,
            "cost_cny": round(d.cost.total_cny, 6) if d else None,
        },
    }


# ---------------- 页面 ----------------


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    story_now = _agent.schedule.now_local() if _agent else None
    return {
        "ok": True,
        # 她的时区 —— 日程按这个走
        "story_timezone": s.story_timezone,
        "story_time": story_now.isoformat() if story_now else None,
        "window": _agent.schedule.state().window.name if _agent else None,
        # DeepSeek 计费时区，恒为北京，与上面无关
        "billing_beijing_time": now_beijing().isoformat(),
        "peak_pricing": is_peak(),
        "api_key_configured": bool(s.deepseek_api_key),
        "delay_scale": s.delay_scale,
    }


@app.get("/usage")
async def usage() -> dict[str, Any]:
    if _recorder is None:
        raise HTTPException(503, "服务未就绪")
    return _recorder.summary()


@app.get("/debug/beats")
async def debug_beats(character_id: str = "h01") -> list[dict[str, Any]]:
    """检查桥段是否都能解析。编剧改完 md 之后看这里。"""
    return [
        {
            "id": b.id, "title": b.title, "kind": b.kind.value,
            "priority": b.priority, "once": b.once,
            "turns": [b.min_turns, b.max_turns],
            "outcomes": [o.id for o in b.outcomes],
            "has_hidden": bool(b.hidden),
            "source": b.source,
        }
        for b in load_beats(character_id)
    ]


@app.get("/debug/card")
async def debug_card(character_id: str = "h01") -> dict[str, Any]:
    """检查人设卡装配结果。稳定前缀的体量直接决定缓存成本。"""
    try:
        card = load_card(character_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    persona, lexicon = card.stable_text()
    return {
        "character_id": card.character_id,
        "approx_tokens": card.approx_tokens,
        "chars": {"persona": len(persona), "lexicon": len(lexicon)},
        "persona": persona,
        "lexicon": lexicon,
    }


# ---------------- 存档 ----------------


@app.get("/api/saves")
async def list_saves() -> list[dict[str, Any]]:
    db, _, _ = deps()
    out = []
    for s in db.list_saves():
        stage = Stage(s["stage"])
        out.append({
            "id": s["id"], "name": s["name"],
            "player": f"{s['player_surname']}{s['player_given']}",
            "stage": stage.value, "stage_label": stage.label,
            "affinity": s["affinity"], "updated_at": s["updated_at"],
        })
    return out


@app.get("/api/presets")
async def presets() -> list[dict[str, Any]]:
    return listing()


@app.post("/api/saves")
async def create_save(body: CreateSave) -> dict[str, Any]:
    db, _, _ = deps()
    if body.preset not in PRESETS:
        raise HTTPException(400, f"没有这个预设：{body.preset}")
    save_id = db.create_save(
        body.name, character_id=body.character_id,
        surname=body.surname.strip(), given=body.given.strip(),
    )
    seed(db, save_id, body.preset)
    return {"id": save_id, "preset": body.preset}


@app.delete("/api/saves/{save_id}")
async def delete_save(save_id: int) -> dict[str, bool]:
    db, _, _ = deps()
    db.delete_save(save_id)
    return {"ok": True}


@app.get("/api/saves/{save_id}/export")
async def export_save(save_id: int) -> dict[str, Any]:
    db, _, _ = deps()
    try:
        return db.export_save(save_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------- 状态 ----------------


@app.get("/api/saves/{save_id}/state")
async def get_state(save_id: int) -> dict[str, Any]:
    db, agent, _ = deps()
    save = db.get_save(save_id)
    if save is None:
        raise HTTPException(404, "存档不存在")

    stage = Stage(save["stage"])
    emotions = EmotionState.from_json(save["emotions"])
    sched = agent.schedule.state()
    behavior = STAGE_BEHAVIOR[stage]

    progress = agent._progress(save)
    beat = get_beat(progress.beat_id, save["character_id"]) if progress.beat_id else None

    return {
        "id": save_id,
        "name": save["name"],
        "player": f"{save['player_surname']}{save['player_given']}",
        "stage": stage.value,
        "stage_label": stage.label,
        "affinity": round(save["affinity"], 1),
        "emotions": {e.value: round(v, 2) for e, v in emotions.active().items()},
        "emotion_note": emotions.describe(),
        "options": [o.as_dict() for o in agent.current_options(save_id)],
        "topics": [t.as_dict() for t in agent._pending_topics(save)],
        "beat": ({"id": beat.id, "title": beat.title, "turn": progress.turn}
                 if beat else None),
        "available_beats": [
            {"id": b.id, "title": b.title} for b in agent.available_beats(save_id)
        ],
        "flags": sorted(agent._flags(save)),
        "schedule": {
            "window": sched.window.name,
            "pace": sched.pace.value,
            "can_initiate": sched.can_initiate,
            "local_time": sched.local_time.isoformat(),
            "note": sched.note,
        },
        "mother_night_shift": agent.schedule.is_mother_night_shift(),
        "farewell": behavior.farewell,
        "pending": agent.pending_count(save_id),
        "facts": [f["content"] for f in db.get_facts(save_id, 20)],
        "insights": [
            {"content": i["content"], "kind": i["kind"], "weight": i["weight"]}
            for i in db.get_insights(save_id, limit=12)
        ],
        "episodes": [
            {"summary": e["summary"],
             "happened_at": parse_ts(e["happened_at"]).strftime("%m-%d")}
            for e in db.get_episodes(save_id, 12)
        ],
    }


# ---------------- 消息 ----------------


@app.get("/api/saves/{save_id}/messages")
async def get_messages(save_id: int, limit: int = 200) -> list[dict[str, Any]]:
    db, _, _ = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    return [
        {"id": m["id"], "role": m["role"], "content": m["content"],
         "at": m["created_at"], "proactive": bool(m["proactive"])}
        for m in db.recent_messages(save_id, limit=limit)
    ]


@app.post("/api/saves/{save_id}/open")
async def open_chat(save_id: int) -> dict[str, Any]:
    """打开聊天。她可能主动发起一场戏。"""
    db, agent, _ = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    try:
        return _turn_payload(await agent.open_chat(save_id))
    except LLMError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/saves/{save_id}/choose")
async def choose(
    save_id: int, body: Choose, background: BackgroundTasks
) -> dict[str, Any]:
    db, agent, reflector = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    try:
        result = await agent.choose(save_id, body.index)
    except IndexError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc

    if get_settings().auto_reflect and reflector.should_run(save_id):
        background.add_task(_safe_reflect, reflector, save_id)
    return _turn_payload(result)


@app.post("/api/saves/{save_id}/topic")
async def choose_topic(
    save_id: int, body: Choose, background: BackgroundTasks
) -> dict[str, Any]:
    """玩家选了今天聊什么。"""
    db, agent, reflector = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    try:
        result = await agent.choose_topic(save_id, body.index)
    except IndexError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc

    if get_settings().auto_reflect and reflector.should_run(save_id):
        background.add_task(_safe_reflect, reflector, save_id)
    return _turn_payload(result)


@app.post("/api/saves/{save_id}/topics")
async def refresh_topics(save_id: int) -> dict[str, Any]:
    """换个话题。当前这场戏就此打住。"""
    db, agent, _ = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    try:
        return _turn_payload(await agent.refresh_topics(save_id))
    except LLMError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/saves/{save_id}/beat")
async def start_beat(save_id: int, body: StartBeat) -> dict[str, Any]:
    """玩家主动开启一场戏。"""
    db, agent, _ = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    try:
        return _turn_payload(await agent.start_beat(save_id, body.beat_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/saves/{save_id}/poll")
async def poll(save_id: int) -> dict[str, Any]:
    """取到点该送达的消息。前端定时调用。"""
    db, agent, _ = deps()
    if db.get_save(save_id) is None:
        raise HTTPException(404, "存档不存在")
    due = agent.collect_due(save_id)
    return {
        "messages": [
            {"id": m["id"], "content": m["content"], "at": m["created_at"],
             "proactive": bool(m["proactive"])}
            for m in due
        ],
        "pending": agent.pending_count(save_id),
    }


@app.post("/api/saves/{save_id}/reflect")
async def force_reflect(save_id: int) -> dict[str, Any]:
    _, _, reflector = deps()
    r = await reflector.run(save_id, force=True)
    return {
        "facts_added": r.facts_added,
        "episodes_added": r.episodes_added,
        "insights_added": r.insights_added,
        "skipped": r.skipped,
        "error": r.error,
    }


async def _safe_reflect(reflector: Reflector, save_id: int) -> None:
    try:
        await reflector.run(save_id)
    except Exception:
        log.exception("save=%s 后台归档异常", save_id)


# 让 uvicorn --reload 下的 asyncio 警告安静一点
asyncio.get_event_loop_policy()
