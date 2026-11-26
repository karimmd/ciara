from __future__ import annotations

from typing import Any, Dict, List


def _is_type(v: Any, t: str) -> bool:
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    if t == "null":
        return v is None
    return True


def validate(instance: Any, schema: Dict[str, Any], strict_additional: bool = True, path: str = "$") -> List[str]:
    errs: List[str] = []
    if "anyOf" in schema:
        if not any(not validate(instance, sub, strict_additional, path) for sub in schema["anyOf"]):
            return [f"{path}: matches none of anyOf"]
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(instance, x) for x in types):
            return [f"{path}: expected {t}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: above maximum")
    if isinstance(instance, list) and "items" in schema:
        for i, x in enumerate(instance):
            errs.extend(validate(x, schema["items"], strict_additional, f"{path}[{i}]"))
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in instance:
                errs.append(f"{path}: missing required {r}")
        for k, v in instance.items():
            if k in props:
                errs.extend(validate(v, props[k], strict_additional, f"{path}.{k}"))
            elif "properties" in schema and (strict_additional or schema.get("additionalProperties") is False):
                errs.append(f"{path}: unknown property {k}")
    return errs
