import json
from pathlib import Path

# Đường dẫn registry.json (quy ước chuẩn)
REGISTRY_PATH = Path("models/registry.json")


def load_registry():
    """
    Load registry.json. Nếu không có thì tạo mới.
    """
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    else:
        return {
            "current_version": None,
            "versions": {}
        }


def save_registry(reg):
    """
    Ghi registry.json ra file.
    """
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=4)


def update_current_version(version: str, model_path: str):
    """
    Cập nhật current_version trong registry.
    """
    reg = load_registry()

    reg["current_version"] = version
    reg.setdefault("versions", {})
    reg["versions"][version] = {"path": model_path}

    save_registry(reg)


def get_current_version_meta(allow_missing: bool = True):
    """Trả về metadata của version hiện tại.

    Nếu ``current_version`` không tồn tại hoặc không có trong ``versions``:
    - Nếu vẫn còn version khác, tự động fallback sang version mới nhất (sorted)
      và cập nhật registry để tránh lỗi lặp lại.
    - Nếu không có version nào và ``allow_missing=True`` (mặc định),
      trả về dict rỗng dạng {"current_version": None, "path": None} thay vì
      raise exception (giúp dashboard / consumer không bị crash khi khởi động).
    """

    reg = load_registry()
    versions = reg.get("versions", {}) or {}
    cur = reg.get("current_version")

    if cur and cur in versions:
        return {"current_version": cur, "path": versions[cur]["path"]}

    # Nếu current_version không hợp lệ nhưng registry có versions, fallback sang version mới nhất
    if versions:
        latest = sorted(versions.keys())[-1]
        reg["current_version"] = latest
        save_registry(reg)
        return {"current_version": latest, "path": versions[latest]["path"], "fallback_from": cur}

    if allow_missing:
        return {"current_version": None, "path": None}

    raise ValueError("No current_version found and registry is empty")
