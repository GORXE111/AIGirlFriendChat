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

    `debug=False` 时由 `check_production()` 强制校验，不会带着测试值上线。
    """

    auto_reflect: bool = True
    """对话累积到阈值后自动跑慢回路归档。"""

    debug: bool = True
    """开发模式。**上线必须设 False**（`.env` 里 `DEBUG=false`）。

    为 False 时启动会跑 `check_production()`，把调试用的数值挡在门外。
    默认 True 是故意的 —— 忘了配置只会让本地跑不起来，不会让线上带着
    测试值静悄悄地跑。
    """


# 只在生产有意义的下限。测试值和生产值差一个数量级以上的都该登记在这。
PRODUCTION_FLOORS: tuple[tuple[str, float, str], ...] = (
    ("max_delay_seconds", 60, "延迟上限太小，等于关掉了延迟机制，她会全时段秒回"),
    ("delay_scale", 1.0, "时间被压缩了，延迟本身就是活人感"),
)


class UnsafeProductionConfig(RuntimeError):
    """带着调试值要上生产。"""


def check_production(settings: Settings) -> None:
    """`debug=False` 时校验那些「调试时故意调小」的值。

    这类数字靠记是记不住的 —— `max_delay_seconds=5` 在这个项目里已经躺了
    很久，每次讨论都说「上线记得改」。改成启动就崩。
    """
    if settings.debug:
        return
    bad = [
        f"{name}={getattr(settings, name)!r}（应 ≥ {floor}）：{why}"
        for name, floor, why in PRODUCTION_FLOORS
        if getattr(settings, name) < floor
    ]
    if bad:
        raise UnsafeProductionConfig(
            "DEBUG=false 但配置里还是调试值：\n  " + "\n  ".join(bad)
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        settings = Settings()
        check_production(settings)
        _settings = settings
    return _settings


def reset_settings() -> None:
    """测试用。"""
    global _settings
    _settings = None
