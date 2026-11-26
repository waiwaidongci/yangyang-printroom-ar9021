"""晨检数据和运行参数的公共约定。"""

SPEC = {
    "columns": ["id", "date", "humidity_pct", "reading_type"],
    "fields": {
        "id": "台账编号",
        "date": "晨检日期",
        "humidity_pct": "相对湿度（%RH）",
        "reading_type": "记录口径",
    },
    "settings": [
        {"key": "lag_order", "label": "差分滞后阶数", "default": 2, "min": 1, "max": 7, "step": 1},
        {"key": "backtest_days", "label": "回测日历天数", "default": 30, "min": 7, "max": 60, "step": 1},
        {"key": "forecast_days", "label": "预报天数", "default": 5, "min": 1, "max": 14, "step": 1},
        {"key": "wet_threshold", "label": "优先晨检阈值（%RH）", "default": 65, "min": 55, "max": 85, "step": 1},
    ],
    "modules": {
        "datasets": "晨检台账",
        "runs": "差分预报运行",
        "results": "版房晨报",
        "reviews": "交班复核",
    },
    "limitation": "仅适用于同版房、同测点、同晨检口径的每日读数。回测使用定稿台账评估一步预测，补测何时可见需核对原始流水。多日预报递推使用预测值，误差可能累积。",
    "dataset_name": "北版房_晨检湿度台账_2026-11-24.csv",
    "recommendation": "按未来晨检表安排优先实测，按缺测区间核对原始台账，并在取得新读数后重新运行。",
}


def normalize_settings(settings: dict) -> dict[str, int]:
    allowed = {item["key"] for item in SPEC["settings"]}
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"未知运行参数：{', '.join(sorted(map(str, unknown)))}。")

    values = {}
    for item in SPEC["settings"]:
        key = item["key"]
        value = settings.get(key, item["default"])
        if type(value) not in (int, float) or not item["min"] <= value <= item["max"]:
            raise ValueError(f"{item['label']}须在 {item['min']} 至 {item['max']} 之间。")
        if (value - item["min"]) % item["step"] != 0:
            raise ValueError(f"{item['label']}须按步长 {item['step']} 设置。")
        values[key] = int(value)
    return values
