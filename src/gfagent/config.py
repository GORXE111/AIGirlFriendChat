from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = "https://api.deepseek.com"

    llm_timeout_seconds: float = 60.0
    llm_connect_timeout_seconds: float = 10.0
    llm_max_retries: int = 3
    llm_max_concurrency: int = 32
    """单进程对 provider 的并发上限。三个女主同时活跃时这个值会被吃满。"""

    log_level: str = "INFO"

    usage_log_path: str | None = ".usage/usage.jsonl"
    """每次调用的 usage/cost 落盘路径。设为 None 关闭。"""

    db_path: str = "data/saves.db"

    story_timezone: str = "Asia/Shanghai"
    """**她所在的时区** —— 她是中国高中生，日程按她的当地时间走。

    这跟开发者／玩家在哪无关。新加坡、马来西亚、菲律宾都是 UTC+8，
    读数与上海一致，不需要改。

    ⚠️ 与 `timewindow.py` 的峰谷计费时区是两回事 —— 那个由 DeepSeek 决定，
    永远是北京时间，不受这里影响。
    """

    delay_scale: float = 1.0
    """时间压缩。1.0 ＝ 真实延迟；调试时设 0.01，40 分钟压成 24 秒。

    ⚠️ 上线必须是 1.0 —— 延迟本身就是活人感，压缩掉就只剩一个秒回的机器人。
    """

    max_delay_seconds: int = 5
    """单次回复的延迟上限（真实秒数，施加 delay_scale 之前）。

    日程决定"快还是慢"，这个上限决定"最慢能到多慢"。

    ⚠️ **5 秒是测试值，等于关掉了延迟机制。** 所有时段都会秒回，
    「她什么时候方便回」这层活人感完全看不到 —— 现在就是个普通聊天机器人。
    正式跑改回 `600`（10 分钟），或在 `.env` 里设 `MAX_DELAY_SECONDS=600`。
    """

    auto_reflect: bool = True
    """对话累积到阈值后自动跑慢回路归档。"""


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """测试用。"""
    global _settings
    _settings = None
