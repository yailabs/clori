#!/usr/bin/env python3
"""Validate the canonical operator registry and emit immutable C descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


SCHEMA = "yvex.operator.registry.v1"
DISCOVERY_SCHEMA = "yvex.command.discovery.v1"
LANES = {
    "runtime-client",
    "offline-engine",
    "daemon-entrypoint",
    "REPL-local",
    "API-only",
    "test-only",
}
VISIBILITIES = {
    "product-default",
    "product-advanced",
    "engineering",
    "automation",
    "API-only",
    "test-only",
    "removed",
}
PLANES = {"Compile", "Run", "Execute", "Integrate", "Inspect", "System", "Profile"}
TTY_POLICIES = {"required", "optional", "forbidden"}
VALUE_TYPES = {"boolean", "u64", "number", "enum", "path", "name", "text", "delegated"}
REQUIREMENTS = {
    "daemon": {"absent", "optional", "required"},
    "model": {"none", "selected-config", "registry-model", "runtime-model"},
    "artifact": {"none", "descriptor", "admitted-artifact", "runtime-binding"},
    "backend": {"none", "optional", "required", "runtime"},
}
OLD_EXECUTABLES = {"yvex-dev", "yvex-openai"}
FORBIDDEN_TOP_LEVEL = {"dev", "evidence", "graph", "quant", "source", "tensor", "tokenizer", "integrate", "eval", "bench"}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
FLAG = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
WORD = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
TOP_LEVEL_KEYS = {
    "catalogs",
    "discovery_schema",
    "flag_sets",
    "global_flag_sets",
    "operation_defaults",
    "operations",
    "removed_paths",
    "schema",
    "schema_version",
}
CATALOG_KEYS = {
    "adapters",
    "completion_providers",
    "default_providers",
    "protocol_operations",
    "renderers",
    "validators",
}
ADAPTER_CATALOG_KEYS = {
    "runtime-client",
    "offline-engine",
    "daemon-entrypoint",
    "REPL-local",
}
OPERATION_KEYS = {
    "CLI_projection",
    "TTY_policy",
    "adapter_argv",
    "adapter_id",
    "aliases",
    "architectural_plane",
    "arguments",
    "artifact_requirement",
    "backend_requirement",
    "command_path",
    "completion_provider",
    "daemon_requirement",
    "default_providers",
    "deprecation_state",
    "documentation_owner",
    "flag_sets",
    "flags",
    "input_schema",
    "lane",
    "model_requirement",
    "operation_id",
    "protocol_operation",
    "renderer_id",
    "result_schema",
    "schema_version",
    "side_effects",
    "slash_arguments",
    "slash_aliases",
    "slash_projection",
    "summary",
    "superseded_by",
    "test_owner",
    "validator_ids",
    "visibility",
}
FLAG_KEYS = {
    "aliases",
    "config",
    "conflicts",
    "default_provider",
    "dependencies",
    "deprecation",
    "enum_values",
    "environment",
    "multiplicity",
    "name",
    "output_interaction",
    "protocol_field",
    "range",
    "required",
    "takes_value",
    "validator",
    "value_type",
}
ARGUMENT_KEYS = {
    "completion_provider",
    "enum_values",
    "multiplicity",
    "name",
    "range",
    "required",
    "sensitive_display",
    "validator",
    "value_type",
}


class RegistryError(ValueError):
    """One exact registry field is invalid."""


def fail(where: str, message: str) -> None:
    raise RegistryError(f"{where}: {message}")


def reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        fail(where, f"unknown field {unknown[0]!r}")


def boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        fail(where, "must be boolean")
    return value


def load_registry(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            registry = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(str(path), f"cannot read strict UTF-8 JSON: {exc}")
    if not isinstance(registry, dict):
        fail("registry", "top level must be an object")
    return registry


def text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        fail(where, "must be a non-empty string")
    return value


def string_list(value: Any, where: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        fail(where, "must be a string array")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(text(item, f"{where}[{index}]", allow_empty=False))
    return result


def catalog(registry: dict[str, Any], name: str) -> set[str]:
    owner: Any = registry.get("catalogs", {})
    for component in name.split("."):
        owner = owner.get(component) if isinstance(owner, dict) else None
    values = string_list(owner, f"catalogs.{name}", allow_empty=False)
    if len(values) != len(set(values)):
        fail(f"catalogs.{name}", "contains duplicate values")
    return set(values)


def expand_flag_sets(registry: dict[str, Any], operation: dict[str, Any], where: str) -> list[dict[str, Any]]:
    sets = registry.get("flag_sets")
    if not isinstance(sets, dict):
        fail("flag_sets", "must be an object")
    result: list[dict[str, Any]] = []
    names = string_list(registry.get("global_flag_sets", []), "global_flag_sets")
    names.extend(string_list(operation.get("flag_sets", []), f"{where}.flag_sets"))
    for name in names:
        rows = sets.get(name)
        if isinstance(rows, dict):
            expanded: list[dict[str, Any]] = []
            for flag_name in string_list(rows.get("values", []), f"flag_sets.{name}.values"):
                expanded.append({"name": flag_name, "value_type": "delegated", "takes_value": True})
            for flag_name in string_list(rows.get("booleans", []), f"flag_sets.{name}.booleans"):
                expanded.append({"name": flag_name, "value_type": "boolean", "takes_value": False})
            for flag_name in string_list(rows.get("repeatable_values", []), f"flag_sets.{name}.repeatable_values"):
                expanded.append({"name": flag_name, "value_type": "delegated", "takes_value": True,
                                 "multiplicity": "repeatable"})
            rows = expanded
        if not isinstance(rows, list):
            fail(f"{where}.flag_sets", f"unknown flag set {name!r}")
        result.extend(rows)
    rows = operation.get("flags", [])
    if not isinstance(rows, list):
        fail(f"{where}.flags", "must be an array")
    result.extend(rows)
    return result


def validate_flag(flag: Any, where: str, defaults: set[str], validators: set[str]) -> dict[str, Any]:
    if not isinstance(flag, dict):
        fail(where, "must be an object")
    reject_unknown(flag, FLAG_KEYS, where)
    name = text(flag.get("name"), f"{where}.name")
    if not FLAG.fullmatch(name):
        fail(f"{where}.name", "must be one canonical long option")
    aliases = string_list(flag.get("aliases", []), f"{where}.aliases")
    for alias in aliases:
        if alias != "-h" and not FLAG.fullmatch(alias):
            fail(f"{where}.aliases", f"invalid flag alias {alias!r}")
    value_type = text(flag.get("value_type", "delegated"), f"{where}.value_type")
    if value_type not in VALUE_TYPES:
        fail(f"{where}.value_type", f"unknown value type {value_type!r}")
    multiplicity = text(flag.get("multiplicity", "single"), f"{where}.multiplicity")
    if multiplicity not in {"single", "repeatable"}:
        fail(f"{where}.multiplicity", "must be single or repeatable")
    default_provider = text(flag.get("default_provider", "none"), f"{where}.default_provider")
    if default_provider not in defaults:
        fail(f"{where}.default_provider", f"unknown default provider {default_provider!r}")
    validator = text(flag.get("validator", "syntax.delegated"), f"{where}.validator")
    if validator not in validators:
        fail(f"{where}.validator", f"unknown validator {validator!r}")
    conflicts = string_list(flag.get("conflicts", []), f"{where}.conflicts")
    dependencies = string_list(flag.get("dependencies", []), f"{where}.dependencies")
    takes_value = boolean(
        flag.get("takes_value", value_type != "boolean"),
        f"{where}.takes_value",
    )
    if value_type == "boolean" and takes_value:
        fail(where, "boolean flag cannot take a value")
    enum_values = string_list(flag.get("enum_values", []), f"{where}.enum_values")
    if len(enum_values) != len(set(enum_values)):
        fail(f"{where}.enum_values", "contains duplicate values")
    if value_type == "enum" and not enum_values:
        fail(f"{where}.enum_values", "enum flag requires admitted values")
    return {
        "name": name,
        "aliases": aliases,
        "value_type": value_type,
        "takes_value": takes_value,
        "multiplicity": multiplicity,
        "required": boolean(flag.get("required", False), f"{where}.required"),
        "default_provider": default_provider,
        "range": text(flag.get("range", "delegated"), f"{where}.range"),
        "enum_values": enum_values,
        "conflicts": conflicts,
        "dependencies": dependencies,
        "environment": text(flag.get("environment", "none"), f"{where}.environment"),
        "config": text(flag.get("config", "none"), f"{where}.config"),
        "protocol_field": text(flag.get("protocol_field", "none"), f"{where}.protocol_field"),
        "output_interaction": text(flag.get("output_interaction", "none"), f"{where}.output_interaction"),
        "deprecation": text(flag.get("deprecation", "current"), f"{where}.deprecation"),
        "validator": validator,
    }


def validate_argument(
    argument: Any,
    where: str,
    validators: set[str],
    completion_providers: set[str],
) -> dict[str, Any]:
    if not isinstance(argument, dict):
        fail(where, "must be an object")
    reject_unknown(argument, ARGUMENT_KEYS, where)
    name = text(argument.get("name"), f"{where}.name")
    validator = text(argument.get("validator", "syntax.delegated"), f"{where}.validator")
    if validator not in validators:
        fail(f"{where}.validator", f"unknown validator {validator!r}")
    multiplicity = text(argument.get("multiplicity", "one"), f"{where}.multiplicity")
    if multiplicity not in {"one", "optional", "many"}:
        fail(f"{where}.multiplicity", "must be one, optional, or many")
    completion_provider = text(
        argument.get("completion_provider", "none"),
        f"{where}.completion_provider",
    )
    if completion_provider not in completion_providers:
        fail(
            f"{where}.completion_provider",
            f"unknown completion provider {completion_provider!r}",
        )
    value_type = text(argument.get("value_type", "text"), f"{where}.value_type")
    if value_type not in VALUE_TYPES - {"boolean"}:
        fail(f"{where}.value_type", f"unknown value type {value_type!r}")
    enum_values = string_list(argument.get("enum_values", []), f"{where}.enum_values")
    if len(enum_values) != len(set(enum_values)):
        fail(f"{where}.enum_values", "contains duplicate values")
    if value_type == "enum" and not enum_values:
        fail(f"{where}.enum_values", "enum argument requires admitted values")
    required = boolean(
        argument.get("required", multiplicity == "one"),
        f"{where}.required",
    )
    if required != (multiplicity == "one"):
        fail(
            f"{where}.required",
            "must be true only for one-value arguments",
        )
    return {
        "name": name,
        "value_type": value_type,
        "required": required,
        "multiplicity": multiplicity,
        "range": text(argument.get("range", "delegated"), f"{where}.range"),
        "enum_values": enum_values,
        "completion_provider": completion_provider,
        "sensitive_display": text(argument.get("sensitive_display", "ordinary"), f"{where}.sensitive_display"),
        "validator": validator,
    }


def validate_arguments(
    value: Any,
    where: str,
    validators: set[str],
    completion_providers: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        fail(where, "must be an array")
    result = [
        validate_argument(row, f"{where}[{index}]", validators, completion_providers)
        for index, row in enumerate(value)
    ]
    optional_seen = False
    many_seen = False
    for index, argument in enumerate(result):
        if many_seen:
            fail(f"{where}[{index}]", "cannot follow a many-value argument")
        if argument["multiplicity"] == "many":
            many_seen = True
        elif argument["multiplicity"] == "optional":
            optional_seen = True
        elif optional_seen:
            fail(f"{where}[{index}]", "required argument cannot follow an optional argument")
    return result


def operation_defaults(registry: dict[str, Any]) -> dict[str, Any]:
    defaults = registry.get("operation_defaults")
    if not isinstance(defaults, dict):
        fail("operation_defaults", "must be an object")
    return defaults


def validate_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    reject_unknown(registry, TOP_LEVEL_KEYS, "registry")
    if registry.get("schema") != SCHEMA or registry.get("schema_version") != 1:
        fail("registry.schema", f"expected {SCHEMA!r} version 1")
    if registry.get("discovery_schema") != DISCOVERY_SCHEMA:
        fail("registry.discovery_schema", f"expected {DISCOVERY_SCHEMA!r}")
    catalogs = registry.get("catalogs")
    if not isinstance(catalogs, dict):
        fail("catalogs", "must be an object")
    reject_unknown(catalogs, CATALOG_KEYS, "catalogs")
    adapter_catalogs = catalogs.get("adapters")
    if not isinstance(adapter_catalogs, dict):
        fail("catalogs.adapters", "must be an object")
    reject_unknown(adapter_catalogs, ADAPTER_CATALOG_KEYS, "catalogs.adapters")
    flag_sets = registry.get("flag_sets")
    if not isinstance(flag_sets, dict):
        fail("flag_sets", "must be an object")
    for name, rows in flag_sets.items():
        if isinstance(rows, dict):
            reject_unknown(
                rows,
                {"values", "booleans", "repeatable_values"},
                f"flag_sets.{name}",
            )
        elif not isinstance(rows, list):
            fail(f"flag_sets.{name}", "must be an array or shorthand object")
    adapters = {
        lane: catalog(registry, f"adapters.{lane}")
        for lane in ("runtime-client", "offline-engine", "daemon-entrypoint", "REPL-local")
    }
    validators = catalog(registry, "validators")
    renderers = catalog(registry, "renderers")
    protocols = catalog(registry, "protocol_operations")
    defaults = catalog(registry, "default_providers")
    completion_providers = catalog(registry, "completion_providers")
    source = registry.get("operations")
    if not isinstance(source, list) or not source:
        fail("operations", "must be a non-empty array")
    common = operation_defaults(registry)
    reject_unknown(common, OPERATION_KEYS, "operation_defaults")
    operations: list[dict[str, Any]] = []
    operation_ids: set[str] = set()
    paths: dict[tuple[str, ...], str] = {}
    aliases: dict[tuple[str, ...], str] = {}
    for index, raw in enumerate(source):
        where = f"operations[{index}]"
        if not isinstance(raw, dict):
            fail(where, "must be an object")
        reject_unknown(raw, OPERATION_KEYS, where)
        operation = dict(common)
        operation.update(raw)
        operation_id = text(operation.get("operation_id"), f"{where}.operation_id")
        if not IDENTIFIER.fullmatch(operation_id):
            fail(f"{where}.operation_id", "has invalid stable identity syntax")
        if operation_id in operation_ids:
            fail(f"{where}.operation_id", f"duplicate operation ID {operation_id!r}")
        operation_ids.add(operation_id)
        lane = text(operation.get("lane"), f"{where}.lane")
        if lane not in LANES:
            fail(f"{where}.lane", f"unknown lane {lane!r}")
        visibility = text(operation.get("visibility"), f"{where}.visibility")
        if visibility not in VISIBILITIES:
            fail(f"{where}.visibility", f"unknown visibility {visibility!r}")
        plane = text(operation.get("architectural_plane"), f"{where}.architectural_plane")
        if plane not in PLANES:
            fail(f"{where}.architectural_plane", f"unknown plane {plane!r}")
        command_path = string_list(operation.get("command_path", []), f"{where}.command_path")
        for word in command_path:
            if not WORD.fullmatch(word):
                fail(f"{where}.command_path", f"invalid word {word!r}")
        projection = boolean(
            operation.get("CLI_projection", bool(command_path)),
            f"{where}.CLI_projection",
        )
        path_key = tuple(command_path)
        if projection:
            if not command_path:
                fail(f"{where}.command_path", "CLI projection needs a canonical path")
            if command_path[0] in FORBIDDEN_TOP_LEVEL:
                fail(f"{where}.command_path", f"forbidden top-level namespace {command_path[0]!r}")
            if path_key in paths:
                fail(f"{where}.command_path", f"duplicate canonical path owned by {paths[path_key]}")
            paths[path_key] = operation_id
        alias_rows = operation.get("aliases", [])
        if not isinstance(alias_rows, list):
            fail(f"{where}.aliases", "must be an array")
        normalized_aliases: list[dict[str, Any]] = []
        for alias_index, alias in enumerate(alias_rows):
            alias_where = f"{where}.aliases[{alias_index}]"
            if not isinstance(alias, dict):
                fail(alias_where, "must be an object")
            reject_unknown(alias, {"path", "deprecation"}, alias_where)
            alias_path = string_list(alias.get("path"), f"{alias_where}.path")
            for word in alias_path:
                if not WORD.fullmatch(word) and not (
                    len(alias_path) == 1 and
                    (word == "-h" or FLAG.fullmatch(word))
                ):
                    fail(f"{alias_where}.path", f"invalid word {word!r}")
            alias_key = tuple(alias_path)
            if alias_key in paths or alias_key in aliases:
                fail(alias_where, "alias collides with a canonical path or alias")
            if alias_path and alias_path[0] in FORBIDDEN_TOP_LEVEL:
                fail(alias_where, f"forbidden top-level alias {alias_path[0]!r}")
            aliases[alias_key] = operation_id
            normalized_aliases.append({
                "path": alias_path,
                "deprecation": text(alias.get("deprecation", "current"), f"{alias_where}.deprecation"),
            })
        adapter = text(operation.get("adapter_id"), f"{where}.adapter_id")
        if lane in adapters and adapter not in adapters[lane]:
            fail(f"{where}.adapter_id", f"unknown {lane} adapter {adapter!r}")
        if lane == "runtime-client" and adapter in adapters["offline-engine"]:
            fail(f"{where}.adapter_id", "runtime-client command maps to an offline adapter")
        if lane == "offline-engine" and operation.get("daemon_requirement") == "required":
            fail(f"{where}.daemon_requirement", "offline command cannot be a daemon operation")
        renderer = text(operation.get("renderer_id"), f"{where}.renderer_id")
        if renderer not in renderers:
            fail(f"{where}.renderer_id", f"unknown renderer {renderer!r}")
        protocol = text(operation.get("protocol_operation", "none"), f"{where}.protocol_operation")
        if protocol not in protocols:
            fail(f"{where}.protocol_operation", f"unknown protocol operation {protocol!r}")
        validator_ids = string_list(operation.get("validator_ids", []), f"{where}.validator_ids")
        unknown_validators = sorted(set(validator_ids) - validators)
        if unknown_validators:
            fail(f"{where}.validator_ids", f"unknown validator {unknown_validators[0]!r}")
        default_providers = string_list(operation.get("default_providers", []), f"{where}.default_providers")
        unknown_defaults = sorted(set(default_providers) - defaults)
        if unknown_defaults:
            fail(f"{where}.default_providers", f"unknown default provider {unknown_defaults[0]!r}")
        flags = [
            validate_flag(row, f"{where}.flags[{flag_index}]", defaults, validators)
            for flag_index, row in enumerate(expand_flag_sets(registry, operation, where))
        ]
        flag_names: dict[str, dict[str, Any]] = {}
        for flag in flags:
            for spelling in [flag["name"], *flag["aliases"]]:
                if spelling in flag_names:
                    existing = flag_names[spelling]
                    if (
                        existing["value_type"] != flag["value_type"]
                        or existing["takes_value"] != flag["takes_value"]
                        or existing["default_provider"] != flag["default_provider"]
                    ):
                        fail(
                            f"{where}.flags",
                            f"conflicting flag types/defaults for {spelling!r}",
                        )
                    fail(f"{where}.flags", f"duplicate flag {spelling!r}")
                flag_names[spelling] = flag
        for flag in flags:
            for relation in [*flag["conflicts"], *flag["dependencies"]]:
                if relation not in flag_names:
                    fail(f"{where}.flags.{flag['name']}", f"unknown related flag {relation!r}")
        arguments = validate_arguments(
            operation.get("arguments", []),
            f"{where}.arguments",
            validators,
            completion_providers,
        )
        slash_argument_source = operation.get(
            "slash_arguments", operation.get("arguments", [])
        )
        if not isinstance(slash_argument_source, list):
            fail(f"{where}.slash_arguments", "must be an array")
        slash_arguments = validate_arguments(
            slash_argument_source,
            f"{where}.slash_arguments",
            validators,
            completion_providers,
        )
        completion_provider = text(operation.get("completion_provider", "none"), f"{where}.completion_provider")
        if completion_provider not in completion_providers:
            fail(f"{where}.completion_provider", f"unknown completion provider {completion_provider!r}")
        for field in ("test_owner", "documentation_owner", "input_schema", "result_schema", "summary", "side_effects"):
            text(operation.get(field), f"{where}.{field}")
        if projection and visibility not in {"API-only", "test-only", "removed"}:
            if operation["test_owner"] == "none" or operation["documentation_owner"] == "none":
                fail(where, "product command requires test and documentation owners")
        tty_policy = text(operation.get("TTY_policy"), f"{where}.TTY_policy")
        if tty_policy not in TTY_POLICIES:
            fail(f"{where}.TTY_policy", f"unknown TTY policy {tty_policy!r}")
        requirements: dict[str, str] = {}
        for field, admitted in REQUIREMENTS.items():
            value = text(operation.get(f"{field}_requirement"), f"{where}.{field}_requirement")
            if value not in admitted:
                fail(f"{where}.{field}_requirement", f"unknown requirement {value!r}")
            requirements[field] = value
        slash = text(operation.get("slash_projection", "none"), f"{where}.slash_projection")
        if slash != "none" and (not slash.startswith("/") or " " in slash):
            fail(f"{where}.slash_projection", "must be one slash command spelling")
        slash_aliases = string_list(
            operation.get("slash_aliases", []), f"{where}.slash_aliases"
        )
        for alias in slash_aliases:
            if not alias.startswith("/") or " " in alias:
                fail(f"{where}.slash_aliases", "must contain slash command spellings")
        if slash != "none" and lane not in {"runtime-client", "REPL-local"}:
            fail(
                f"{where}.slash_projection",
                "slash projection requires a client or REPL lane",
            )
        adapter_argv = string_list(operation.get("adapter_argv", []), f"{where}.adapter_argv")
        deprecation = text(operation.get("deprecation_state", "current"), f"{where}.deprecation_state")
        superseded_by = string_list(
            operation.get("superseded_by", []), f"{where}.superseded_by"
        )
        if (deprecation == "removed") != (visibility == "removed"):
            fail(
                where,
                "removed visibility and deprecation state must be declared together",
            )
        if deprecation == "removed" and not superseded_by:
            fail(f"{where}.superseded_by", "removed operation needs an explicit successor")
        if deprecation != "removed" and superseded_by:
            fail(f"{where}.superseded_by", "current operation cannot name a successor")
        operations.append({
            **operation,
            "operation_id": operation_id,
            "command_path": command_path,
            "aliases": normalized_aliases,
            "lane": lane,
            "visibility": visibility,
            "architectural_plane": plane,
            "CLI_projection": projection,
            "adapter_id": adapter,
            "renderer_id": renderer,
            "protocol_operation": protocol,
            "validator_ids": validator_ids,
            "default_providers": default_providers,
            "flags": flags,
            "arguments": arguments,
            "slash_arguments": slash_arguments,
            "completion_provider": completion_provider,
            "requirements": requirements,
            "slash_projection": slash,
            "slash_aliases": slash_aliases,
            "adapter_argv": adapter_argv,
            "deprecation_state": deprecation,
            "superseded_by": superseded_by,
            "TTY_policy": tty_policy,
        })
    for alias_path, owner in aliases.items():
        if alias_path in paths:
            fail(f"operation {owner}", "alias collides with canonical path")
    slash_owners: dict[str, str] = {}
    for operation in operations:
        spellings = ([] if operation["slash_projection"] == "none"
                     else [operation["slash_projection"]]) + operation["slash_aliases"]
        for slash in spellings:
            if slash in slash_owners:
                fail(
                    f"operation {operation['operation_id']}.slash_aliases",
                    f"collides with {slash_owners[slash]!r}",
                )
            slash_owners[slash] = operation["operation_id"]
        for successor in operation["superseded_by"]:
            if successor not in operation_ids:
                fail(
                    f"operation {operation['operation_id']}.superseded_by",
                    f"unknown successor {successor!r}",
                )
            if successor == operation["operation_id"]:
                fail(
                    f"operation {operation['operation_id']}.superseded_by",
                    "operation cannot supersede itself",
                )
    used_flag_sets = set(
        string_list(registry.get("global_flag_sets", []), "global_flag_sets")
    )
    for operation in source:
        used_flag_sets.update(
            string_list(operation.get("flag_sets", []), "operation.flag_sets")
        )
    declared_flag_sets = set(registry.get("flag_sets", {}))
    unused_flag_sets = sorted(declared_flag_sets - used_flag_sets)
    if unused_flag_sets:
        fail("flag_sets", f"orphan flag set {unused_flag_sets[0]!r}")
    removed = registry.get("removed_paths")
    if not isinstance(removed, list):
        fail("removed_paths", "must be an array")
    removed_keys: set[tuple[str, ...]] = set()
    for index, row in enumerate(removed):
        where = f"removed_paths[{index}]"
        if not isinstance(row, dict):
            fail(where, "must be an object")
        reject_unknown(row, {"path", "hint"}, where)
        key = tuple(string_list(row.get("path"), f"{where}.path", allow_empty=False))
        text(row.get("hint"), f"{where}.hint")
        if key in removed_keys or key in paths or key in aliases:
            fail(where, "removed path collides or is duplicated")
        removed_keys.add(key)
    encoded = json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for name in OLD_EXECUTABLES:
        if name in encoded:
            fail("registry", f"references retired executable {name!r}")
    operations.sort(key=lambda row: (row["command_path"], row["operation_id"]))
    return operations


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def c_enum(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).upper().strip("_")


def registry_identity(registry: dict[str, Any]) -> str:
    normalized = json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def render_header(registry: dict[str, Any]) -> str:
    adapter_catalog = registry["catalogs"]["adapters"]
    lines = [
        "/* Generated by tools/generate_operator_registry.py; do not edit. */",
        "#ifndef YVEX_GENERATED_OPERATOR_REGISTRY_H",
        "#define YVEX_GENERATED_OPERATOR_REGISTRY_H",
        "#include <stddef.h>",
        "#define YVEX_OPERATOR_REGISTRY_SCHEMA \"yvex.operator.registry.v1\"",
        "#define YVEX_COMMAND_DISCOVERY_SCHEMA \"yvex.command.discovery.v1\"",
        "typedef enum {",
    ]
    for index, lane in enumerate(sorted(LANES)):
        lines.append(f"    YVEX_OPERATOR_LANE_{c_enum(lane)} = {index},")
    lines.extend(["} yvex_operator_lane;", "typedef enum {"])
    for index, value in enumerate(sorted(VISIBILITIES)):
        lines.append(f"    YVEX_OPERATOR_VISIBILITY_{c_enum(value)} = {index},")
    lines.extend(["} yvex_operator_visibility;", "typedef enum {"])
    for index, value in enumerate(sorted(PLANES)):
        lines.append(f"    YVEX_OPERATOR_PLANE_{c_enum(value)} = {index},")
    lines.append("} yvex_operator_plane;")
    for lane, type_name in (("runtime-client", "runtime"), ("offline-engine", "offline"), ("REPL-local", "repl"), ("daemon-entrypoint", "daemon")):
        lines.append("typedef enum {")
        for index, adapter in enumerate(adapter_catalog[lane]):
            lines.append(f"    YVEX_OPERATOR_{type_name.upper()}_{c_enum(adapter)} = {index},")
        lines.append(f"    YVEX_OPERATOR_{type_name.upper()}_COUNT = {len(adapter_catalog[lane])}")
        lines.append(f"}} yvex_operator_{type_name}_adapter;")
    lines.extend([
        "typedef struct {",
        "    const char *name, *value_type, *aliases, *multiplicity;",
        "    const char *default_provider, *range, *enum_values;",
        "    const char *conflicts, *dependencies, *environment, *config;",
        "    const char *protocol_field, *output_interaction, *deprecation, *validator;",
        "    int takes_value, required;",
        "} yvex_operator_flag_descriptor;",
        "typedef struct {",
        "    const char *name, *value_type, *multiplicity, *range, *enum_values;",
        "    const char *completion_provider, *sensitive_display, *validator;",
        "    int required;",
        "} yvex_operator_argument_descriptor;",
        "typedef struct yvex_operator_descriptor {",
        "    unsigned int schema_version;",
        "    const char *operation_id, *command_path, *aliases, *deprecation_state;",
        "    const char *superseded_by;",
        "    const char *summary, *input_schema, *result_schema, *side_effects;",
        "    const char *protocol_operation, *adapter_id, *renderer_id;",
        "    const char *slash_projection, *slash_aliases, *completion_provider;",
        "    const char *test_owner, *documentation_owner, *default_providers, *validator_ids;",
        "    const char *daemon_requirement, *model_requirement, *artifact_requirement, *backend_requirement;",
        "    const char *tty_policy;",
        "    yvex_operator_lane lane;",
        "    yvex_operator_visibility visibility;",
        "    yvex_operator_plane plane;",
        "    yvex_operator_runtime_adapter runtime_adapter;",
        "    yvex_operator_offline_adapter offline_adapter;",
        "    yvex_operator_repl_adapter repl_adapter;",
        "    yvex_operator_daemon_adapter daemon_adapter;",
        "    const char *const *command_words;",
        "    size_t command_word_count;",
        "    const char *const *adapter_argv;",
        "    size_t adapter_argc;",
        "    const yvex_operator_argument_descriptor *arguments;",
        "    size_t argument_count;",
        "    const yvex_operator_argument_descriptor *slash_arguments;",
        "    size_t slash_argument_count;",
        "    const yvex_operator_flag_descriptor *flags;",
        "    size_t flag_count;",
        "    int cli_projection;",
        "} yvex_operator_descriptor;",
        "typedef struct {",
        "    const char *path, *deprecation;",
        "    const char *const *words;",
        "    size_t word_count, operation_index;",
        "} yvex_operator_alias_descriptor;",
        "typedef struct { const char *path, *hint; } yvex_operator_removed_path;",
        "extern const char yvex_operator_registry_identity[];",
        "extern const yvex_operator_descriptor yvex_operator_descriptors[];",
        "extern const size_t yvex_operator_descriptor_count;",
        "extern const yvex_operator_alias_descriptor yvex_operator_aliases[];",
        "extern const size_t yvex_operator_alias_count;",
        "extern const yvex_operator_removed_path yvex_operator_removed_paths[];",
        "extern const size_t yvex_operator_removed_path_count;",
        "#endif",
        "",
    ])
    return "\n".join(lines)


def joined(values: list[str]) -> str:
    return "|".join(values) if values else "none"


def render_source(registry: dict[str, Any], operations: list[dict[str, Any]], identity: str) -> str:
    lines = [
        "/* Generated by tools/generate_operator_registry.py; do not edit. */",
        '#include "operator/registry.h"',
        f"const char yvex_operator_registry_identity[] = {c_string(identity)};",
    ]
    for index, operation in enumerate(operations):
        if operation["command_path"]:
            words = ", ".join(c_string(word) for word in operation["command_path"])
            lines.append(f"static const char *const command_{index}[] = {{{words}}};")
        for alias_index, alias in enumerate(operation["aliases"]):
            if alias["path"]:
                words = ", ".join(c_string(word) for word in alias["path"])
                lines.append(
                    f"static const char *const alias_{index}_{alias_index}[] = {{{words}}};"
                )
        if operation["adapter_argv"]:
            words = ", ".join(c_string(word) for word in operation["adapter_argv"])
            lines.append(f"static const char *const adapter_{index}[] = {{{words}}};")
        if operation["arguments"]:
            lines.append(f"static const yvex_operator_argument_descriptor arguments_{index}[] = {{")
            for argument in operation["arguments"]:
                lines.append("    {" + ", ".join([
                    c_string(argument["name"]), c_string(argument["value_type"]),
                    c_string(argument["multiplicity"]), c_string(argument["range"]),
                    c_string(joined(argument["enum_values"])), c_string(argument["completion_provider"]),
                    c_string(argument["sensitive_display"]), c_string(argument["validator"]),
                    "1" if argument["required"] else "0",
                ]) + "},")
            lines.append("};")
        if operation["slash_arguments"]:
            lines.append(f"static const yvex_operator_argument_descriptor slash_arguments_{index}[] = {{")
            for argument in operation["slash_arguments"]:
                lines.append("    {" + ", ".join([
                    c_string(argument["name"]), c_string(argument["value_type"]),
                    c_string(argument["multiplicity"]), c_string(argument["range"]),
                    c_string(joined(argument["enum_values"])), c_string(argument["completion_provider"]),
                    c_string(argument["sensitive_display"]), c_string(argument["validator"]),
                    "1" if argument["required"] else "0",
                ]) + "},")
            lines.append("};")
        if operation["flags"]:
            lines.append(f"static const yvex_operator_flag_descriptor flags_{index}[] = {{")
            for flag in operation["flags"]:
                lines.append("    {" + ", ".join([
                    c_string(flag["name"]), c_string(flag["value_type"]), c_string(joined(flag["aliases"])),
                    c_string(flag["multiplicity"]), c_string(flag["default_provider"]), c_string(flag["range"]),
                    c_string(joined(flag["enum_values"])), c_string(joined(flag["conflicts"])),
                    c_string(joined(flag["dependencies"])), c_string(flag["environment"]),
                    c_string(flag["config"]), c_string(flag["protocol_field"]),
                    c_string(flag["output_interaction"]), c_string(flag["deprecation"]),
                    c_string(flag["validator"]), "1" if flag["takes_value"] else "0",
                    "1" if flag["required"] else "0",
                ]) + "},")
            lines.append("};")
    lines.append("const yvex_operator_descriptor yvex_operator_descriptors[] = {")
    adapter_catalog = registry["catalogs"]["adapters"]
    for index, operation in enumerate(operations):
        lane = operation["lane"]
        runtime_adapter = "YVEX_OPERATOR_RUNTIME_COUNT"
        offline_adapter = "YVEX_OPERATOR_OFFLINE_COUNT"
        repl_adapter = "YVEX_OPERATOR_REPL_COUNT"
        daemon_adapter = "YVEX_OPERATOR_DAEMON_COUNT"
        if lane == "runtime-client":
            runtime_adapter = f"YVEX_OPERATOR_RUNTIME_{c_enum(operation['adapter_id'])}"
        elif lane == "offline-engine":
            offline_adapter = f"YVEX_OPERATOR_OFFLINE_{c_enum(operation['adapter_id'])}"
        elif lane == "REPL-local":
            repl_adapter = f"YVEX_OPERATOR_REPL_{c_enum(operation['adapter_id'])}"
        elif lane == "daemon-entrypoint":
            daemon_adapter = f"YVEX_OPERATOR_DAEMON_{c_enum(operation['adapter_id'])}"
        alias_text = ",".join(" ".join(row["path"]) for row in operation["aliases"]) or "none"
        slash_alias_text = ",".join(operation["slash_aliases"]) or "none"
        command_words = f"command_{index}" if operation["command_path"] else "NULL"
        adapter_words = f"adapter_{index}" if operation["adapter_argv"] else "NULL"
        argument_rows = f"arguments_{index}" if operation["arguments"] else "NULL"
        slash_argument_rows = (
            f"slash_arguments_{index}" if operation["slash_arguments"] else "NULL"
        )
        flag_rows = f"flags_{index}" if operation["flags"] else "NULL"
        lines.extend([
            "    {",
            f"        1u, {c_string(operation['operation_id'])}, {c_string(' '.join(operation['command_path']))},",
            f"        {c_string(alias_text)}, {c_string(operation['deprecation_state'])},",
            f"        {c_string(joined(operation['superseded_by']))},",
            f"        {c_string(operation['summary'])}, {c_string(operation['input_schema'])},",
            f"        {c_string(operation['result_schema'])}, {c_string(operation['side_effects'])},",
            f"        {c_string(operation['protocol_operation'])}, {c_string(operation['adapter_id'])},",
            f"        {c_string(operation['renderer_id'])}, {c_string(operation['slash_projection'])},",
            f"        {c_string(slash_alias_text)},",
            f"        {c_string(operation['completion_provider'])},",
            f"        {c_string(operation['test_owner'])}, {c_string(operation['documentation_owner'])},",
            f"        {c_string(joined(operation['default_providers']))}, {c_string(joined(operation['validator_ids']))},",
            f"        {c_string(operation['requirements']['daemon'])}, {c_string(operation['requirements']['model'])},",
            f"        {c_string(operation['requirements']['artifact'])}, {c_string(operation['requirements']['backend'])},",
            f"        {c_string(operation['TTY_policy'])}, YVEX_OPERATOR_LANE_{c_enum(lane)},",
            f"        YVEX_OPERATOR_VISIBILITY_{c_enum(operation['visibility'])},",
            f"        YVEX_OPERATOR_PLANE_{c_enum(operation['architectural_plane'])},",
            f"        {runtime_adapter}, {offline_adapter}, {repl_adapter}, {daemon_adapter},",
            f"        {command_words}, {len(operation['command_path'])}u, {adapter_words}, {len(operation['adapter_argv'])}u,",
            f"        {argument_rows}, {len(operation['arguments'])}u,",
            f"        {slash_argument_rows}, {len(operation['slash_arguments'])}u,",
            f"        {flag_rows}, {len(operation['flags'])}u,",
            f"        {1 if operation['CLI_projection'] else 0}",
            "    },",
        ])
    lines.extend([
        "};",
        "const size_t yvex_operator_descriptor_count =",
        "    sizeof(yvex_operator_descriptors) / sizeof(yvex_operator_descriptors[0]);",
        "const yvex_operator_alias_descriptor yvex_operator_aliases[] = {",
    ])
    for operation_index, operation in enumerate(operations):
        for alias_index, alias in enumerate(operation["aliases"]):
            words = (
                f"alias_{operation_index}_{alias_index}"
                if alias["path"]
                else "NULL"
            )
            lines.append(
                "    {"
                + ", ".join(
                    [
                        c_string(" ".join(alias["path"])),
                        c_string(alias["deprecation"]),
                        words,
                        f"{len(alias['path'])}u",
                        f"{operation_index}u",
                    ]
                )
                + "},"
            )
    lines.extend([
        "};",
        "const size_t yvex_operator_alias_count =",
        "    sizeof(yvex_operator_aliases) / sizeof(yvex_operator_aliases[0]);",
        "const yvex_operator_removed_path yvex_operator_removed_paths[] = {",
    ])
    for row in registry["removed_paths"]:
        lines.append(f"    {{{c_string(' '.join(row['path']))}, {c_string(row['hint'])}}},")
    lines.extend([
        "};",
        "const size_t yvex_operator_removed_path_count =",
        "    sizeof(yvex_operator_removed_paths) / sizeof(yvex_operator_removed_paths[0]);",
        "",
    ])
    return "\n".join(lines)


def write_if_changed(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def generated(registry_path: pathlib.Path, output: pathlib.Path) -> tuple[str, str, str]:
    registry = load_registry(registry_path)
    operations = validate_registry(registry)
    identity = registry_identity(registry)
    return render_header(registry), render_source(registry, operations, identity), identity + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        header, source, identity = generated(arguments.registry, arguments.output)
    except RegistryError as exc:
        print(f"operator registry: {exc}", file=sys.stderr)
        return 2
    products = {
        arguments.output / "registry.h": header,
        arguments.output / "registry.c": source,
        arguments.output / "registry.sha256": identity,
    }
    if arguments.check:
        stale = [str(path) for path, content in products.items() if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            print("operator registry: stale generated products: " + ", ".join(stale), file=sys.stderr)
            return 1
    else:
        for path, content in products.items():
            write_if_changed(path, content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
