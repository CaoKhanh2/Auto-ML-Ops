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


def get_current_version_meta():
    """
    Trả về metadata của version hiện tại.
    """
    reg = load_registry()

    cur = reg.get("current_version")
    if not cur:
        raise ValueError("No current_version found in registry.json")

    versions = reg.get("versions", {})
    if cur not in versions:
        raise KeyError(f"Version '{cur}' không tồn tại trong registry.")

    return {
        "current_version": cur,
        "path": versions[cur]["path"]
    }
