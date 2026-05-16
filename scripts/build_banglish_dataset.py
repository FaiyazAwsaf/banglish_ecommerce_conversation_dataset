#!/usr/bin/env python3
"""Build anonymized Banglish e-commerce chat datasets from Meta HTML exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
DOMAIN_URL_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|bd|co|io|me|info|shop|store)\S*",
    re.I,
)
DOWNLOAD_ATTACHMENT_RE = re.compile(r"\bDownload\s+(?:file|video|audio):\s*\S+", re.I)
PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
BD_PHONE_RE = re.compile(r"(?<!\d)(?:\+?88[\s().-]*)?0?1[3-9](?:[\s().-]*\d){8}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
MULTISPACE_RE = re.compile(r"[ \t\r\f\v]+")
LINE_START_NAME_RE = re.compile(
    r"(?im)^(\s*(?:name|nam|customer\s+name|\u09a8\u09be\u09ae)\s*[:：=-]\s*).+$"
)

PAYMENT_KEYWORDS = (
    "bkash",
    "bikash",
    "b-kash",
    "nagad",
    "rocket",
    "upay",
    "surecash",
    "trx",
    "transaction",
    "txn",
    "account",
    "ac no",
    "a/c",
    "card",
    "visa",
    "mastercard",
    "\u09ac\u09bf\u0995\u09be\u09b6",
    "\u09a8\u0997\u09a6",
    "\u09b0\u0995\u09c7\u099f",
    "\u099f\u09cd\u09b0\u09be\u09a8\u099c\u09c7\u0995\u09b6\u09a8",
)

ADDRESS_KEYWORDS = (
    "address",
    "addr",
    "thikana",
    "location",
    "house",
    "holding",
    "road",
    "rd",
    "lane",
    "block",
    "sector",
    "flat",
    "floor",
    "avenue",
    "village",
    "vill",
    "union",
    "upazila",
    "thana",
    "post office",
    "postcode",
    "postal",
    "ward",
    "district",
    "zila",
    "\u09a0\u09bf\u0995\u09be\u09a8\u09be",
    "\u09a5\u09bf\u0995\u09be\u09a8\u09be",
    "\u09ac\u09be\u09b8\u09be",
    "\u09ac\u09be\u09a1\u09bc\u09bf",
    "\u09ac\u09be\u09dc\u09bf",
    "\u09b0\u09cb\u09a1",
    "\u09ab\u09cd\u09b2\u09cd\u09af\u09be\u099f",
    "\u09ac\u09cd\u09b2\u0995",
    "\u09b8\u09c7\u0995\u09cd\u099f\u09b0",
    "\u09a5\u09be\u09a8\u09be",
    "\u09aa\u09cb\u09b8\u09cd\u099f",
    "\u099c\u09c7\u09b2\u09be",
    "\u0989\u09aa\u099c\u09c7\u09b2\u09be",
    "\u0993\u09df\u09be\u09b0\u09cd\u09a1",
)

LOCATION_HINTS = (
    "dhaka",
    "mirpur",
    "uttara",
    "banani",
    "gulshan",
    "badda",
    "bashundhara",
    "jatrabari",
    "dhanmondi",
    "mohammadpur",
    "shahjahanpur",
    "azimpur",
    "ashkona",
    "khilkhet",
    "chittagong",
    "ctg",
    "sylhet",
    "khulna",
    "rajshahi",
    "barishal",
    "rangpur",
    "mymensingh",
    "cumilla",
    "narayanganj",
    "gazipur",
)

DIGIT_TRANSLATION = str.maketrans(
    {
        "\u09e6": "0",
        "\u09e7": "1",
        "\u09e8": "2",
        "\u09e9": "3",
        "\u09ea": "4",
        "\u09eb": "5",
        "\u09ec": "6",
        "\u09ed": "7",
        "\u09ee": "8",
        "\u09ef": "9",
        "\u0660": "0",
        "\u0661": "1",
        "\u0662": "2",
        "\u0663": "3",
        "\u0664": "4",
        "\u0665": "5",
        "\u0666": "6",
        "\u0667": "7",
        "\u0668": "8",
        "\u0669": "9",
        "\u06f0": "0",
        "\u06f1": "1",
        "\u06f2": "2",
        "\u06f3": "3",
        "\u06f4": "4",
        "\u06f5": "5",
        "\u06f6": "6",
        "\u06f7": "7",
        "\u06f8": "8",
        "\u06f9": "9",
    }
)


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {k.lower(): v or "" for k, v in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {k.lower(): v or "" for k, v in attrs})
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)


@dataclass
class RawMessage:
    sender: str
    text: str
    image_refs: list[str]
    timestamp: str


@dataclass
class DatasetRow:
    conversation_id: str
    msg_id: int
    role: str
    text_clean: str
    has_image: int = 0
    has_image_context: bool = False
    image_context_position: str = ""
    refers_to_image: bool = False
    intent: str = ""
    language: str = ""
    tone: str = ""


def iter_nodes(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from iter_nodes(child)


def class_tokens(node: Node) -> set[str]:
    return set(node.attrs.get("class", "").split())


def has_class(node: Node, token: str) -> bool:
    return token in class_tokens(node)


def find_first(node: Node, tag: str | None = None, class_token: str | None = None) -> Node | None:
    for candidate in iter_nodes(node):
        if tag is not None and candidate.tag != tag:
            continue
        if class_token is not None and not has_class(candidate, class_token):
            continue
        return candidate
    return None


def text_content(node: Node | str) -> str:
    if isinstance(node, str):
        return node
    if node.tag in {"script", "style"}:
        return ""
    if node.tag == "br":
        return "\n"
    parts: list[str] = []
    for child in node.children:
        parts.append(text_content(child))
    return "".join(parts)


def normalize_text(value: str) -> str:
    value = html.unescape(value or "")
    value = value.translate(DIGIT_TRANSLATION)
    value = ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\xa0", " ")
    value = value.replace("\u00ad", "")
    lines = [MULTISPACE_RE.sub(" ", line).strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def is_image_ref(value: str) -> bool:
    if not value:
        return False
    lower = value.lower().split("?", 1)[0]
    path = Path(lower)
    if "/photos/" in lower or lower.startswith("photos/") or "\\photos\\" in lower:
        return True
    if "stickers_used" in lower:
        return False
    return path.suffix in IMAGE_EXTENSIONS


def collect_image_refs(node: Node) -> list[str]:
    refs: list[str] = []
    for child in iter_nodes(node):
        if child.tag in {"img", "a"}:
            for attr in ("src", "href"):
                value = child.attrs.get(attr, "")
                if is_image_ref(value):
                    refs.append(value)
    deduped = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def parse_message_file(path: Path) -> list[RawMessage]:
    parser = TreeBuilder()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    messages: list[RawMessage] = []
    for section in iter_nodes(parser.root):
        if section.tag != "section" or not has_class(section, "_a6-g"):
            continue
        sender_node = find_first(section, "h2", "_a6-h")
        body_node = find_first(section, "div", "_a6-p")
        footer_node = find_first(section, "footer", "_a6-o")
        if sender_node is None or body_node is None:
            continue
        sender = normalize_text(text_content(sender_node))
        body = normalize_text(text_content(body_node))
        timestamp = normalize_text(text_content(footer_node)) if footer_node else ""
        image_refs = collect_image_refs(body_node)
        if sender:
            messages.append(RawMessage(sender=sender, text=body, image_refs=image_refs, timestamp=timestamp))
    return messages


def discover_message_files(inbox_dir: Path) -> list[Path]:
    return sorted(inbox_dir.glob("*/message_*.html"), key=lambda p: (p.parent.name, p.name))


def infer_business_sender(conversations: dict[Path, list[RawMessage]], override: str | None) -> str:
    if override:
        return override
    counts = Counter(msg.sender for messages in conversations.values() for msg in messages if msg.sender)
    if not counts:
        raise ValueError("Could not infer a business sender; no message senders were found.")
    return counts.most_common(1)[0][0]


def derive_business_aliases(inbox_dir: Path, business_sender: str) -> set[str]:
    aliases = {business_sender}
    for part in inbox_dir.parts:
        match = re.match(r"facebook-(.+?)-\d{4}-\d{2}-\d{2}", part)
        if not match:
            continue
        store_slug = match.group(1)
        lower_slug = store_slug.lower()
        aliases.add(store_slug)
        aliases.add(store_slug.replace("-", " "))
        if lower_slug.endswith("bd") and len(store_slug) > 2:
            base = store_slug[:-2]
            aliases.add(base)
            aliases.add(f"{base} BD")
            aliases.add(f"{base} Bangladesh")
            aliases.add(f"{base}Bangladesh")
            aliases.add(f"{base} Shop")
            aliases.add(f"{base}Shop")
            aliases.add(f"{base}s")
            aliases.add(f"{base}e")
        if lower_slug.endswith("bangladesh") and len(store_slug) > len("bangladesh"):
            base = store_slug[: -len("bangladesh")]
            aliases.add(base)
            aliases.add(f"{base} Bangladesh")
            aliases.add(f"{base}Bangladesh")
            aliases.add(f"{base} BD")
            aliases.add(f"{base}BD")
            aliases.add(f"{base}Ba")
    return {alias for alias in aliases if normalize_text(alias)}


def make_conversation_id(business_id: str, source_key: str) -> str:
    digest = hashlib.sha256(f"{business_id}:{source_key}".encode("utf-8")).hexdigest()[:12]
    return f"{business_id}_conv_{digest}"


def normalize_for_boundary(value: str) -> str:
    return re.escape(value).replace(r"\ ", r"\s+")


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def compile_name_patterns(names: Iterable[str], ascii_boundaries: bool = True) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    ascii_names: list[str] = []
    non_ascii_names: list[str] = []
    for name in sorted({normalize_text(n) for n in names if normalize_text(n)}, key=len, reverse=True):
        if len(name) < 3:
            continue
        escaped = normalize_for_boundary(name)
        if all(ord(ch) < 128 for ch in name):
            ascii_names.append(escaped)
        else:
            non_ascii_names.append(escaped)
    for group in chunked(ascii_names, 250):
        joined = "|".join(group)
        if ascii_boundaries:
            patterns.append(re.compile(rf"(?<![A-Za-z0-9_])(?:{joined})(?![A-Za-z0-9_])", re.I))
        else:
            patterns.append(re.compile(rf"(?:{joined})", re.I))
    for group in chunked(non_ascii_names, 250):
        patterns.append(re.compile(rf"(?:{'|'.join(group)})"))
    return patterns


def replace_phone_candidate(match: re.Match[str]) -> str:
    token = match.group(0)
    digits = re.sub(r"\D", "", token.translate(DIGIT_TRANSLATION))
    if digits.startswith("8801") and len(digits) == 13:
        return "[PHONE]"
    if digits.startswith("01") and len(digits) == 11:
        return "[PHONE]"
    if digits.startswith("1") and len(digits) == 10:
        return "[PHONE]"
    for idx in range(0, max(0, len(digits) - 10)):
        chunk = digits[idx : idx + 11]
        if chunk.startswith("01") and len(chunk) == 11 and chunk[2] in "3456789":
            return "[PHONE]"
    return token


def replace_long_numeric_candidate(match: re.Match[str]) -> str:
    phone_replacement = replace_phone_candidate(match)
    if phone_replacement == "[PHONE]":
        return "[PHONE]"
    return "[PAYMENT_INFO]"


def has_phone_like_value(text: str) -> bool:
    value = text.translate(DIGIT_TRANSLATION)
    if BD_PHONE_RE.search(value):
        return True
    for match in PHONE_CANDIDATE_RE.finditer(value):
        if replace_phone_candidate(match) == "[PHONE]":
            return True
    return False


def looks_like_payment_line(line: str) -> bool:
    lower = line.casefold()
    digit_count = sum(ch.isdigit() for ch in line)
    return digit_count >= 4 and any(keyword in lower for keyword in PAYMENT_KEYWORDS)


def looks_like_address_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 14:
        return False
    lower = stripped.casefold()
    has_address_keyword = any(keyword in lower for keyword in ADDRESS_KEYWORDS)
    has_location_hint = any(keyword in lower for keyword in LOCATION_HINTS)
    if not has_address_keyword and not has_location_hint:
        return False
    digit_count = sum(ch.isdigit() for ch in stripped)
    commaish = stripped.count(",") + stripped.count(";")
    has_address_marker = any(
        marker in lower
        for marker in (
            "address",
            "addr",
            "thikana",
            "\u09a0\u09bf\u0995\u09be\u09a8\u09be",
            "\u09a5\u09bf\u0995\u09be\u09a8\u09be",
        )
    )
    has_location_structure = digit_count >= 1 or commaish >= 2 or "/" in stripped or "-" in stripped
    has_phone_context = has_phone_like_value(stripped) and (commaish >= 1 or has_address_keyword or has_location_hint)
    asks_for_address = "?" in stripped and digit_count == 0 and commaish == 0
    return (has_address_marker or (has_address_keyword and has_location_structure) or has_phone_context) and not asks_for_address


def redact_sensitive_lines(text: str) -> str:
    redacted_lines: list[str] = []
    for line in text.splitlines() or [text]:
        stripped = line.strip()
        if not stripped:
            continue
        if looks_like_payment_line(stripped):
            redacted_lines.append("[PAYMENT_INFO]")
        elif looks_like_address_line(stripped):
            redacted_lines.append("[ADDRESS]")
        else:
            redacted_lines.append(stripped)
    return "\n".join(redacted_lines)


def anonymize_text(
    text: str,
    customer_name_patterns: list[re.Pattern[str]],
    business_name_patterns: list[re.Pattern[str]],
) -> str:
    value = normalize_text(text)
    if not value:
        return ""
    value = DOWNLOAD_ATTACHMENT_RE.sub("[FILE_ATTACHMENT]", value)
    value = EMAIL_RE.sub("[EMAIL]", value)
    value = URL_RE.sub("[URL]", value)
    value = DOMAIN_URL_RE.sub("[URL]", value)
    value = redact_sensitive_lines(value)
    value = CARD_RE.sub(replace_long_numeric_candidate, value)
    value = BD_PHONE_RE.sub("[PHONE]", value)
    value = PHONE_CANDIDATE_RE.sub(replace_phone_candidate, value)
    value = LINE_START_NAME_RE.sub(r"\1[NAME]", value)
    for pattern in business_name_patterns:
        value = pattern.sub("[BUSINESS]", value)
    for pattern in customer_name_patterns:
        value = pattern.sub("[NAME]", value)
    value = URL_RE.sub("[URL]", value)
    value = DOMAIN_URL_RE.sub("[URL]", value)
    value = CARD_RE.sub(replace_long_numeric_candidate, value)
    value = BD_PHONE_RE.sub("[PHONE]", value)
    value = PHONE_CANDIDATE_RE.sub(replace_phone_candidate, value)
    value = normalize_text(value)
    value = value.replace("\n", " ")
    value = MULTISPACE_RE.sub(" ", value).strip()
    return value


def final_pii_sanitize(text: str) -> str:
    marker = "[IMAGE_ATTACHMENT]"
    had_image_marker = text.startswith(marker)
    value = normalize_text(text)
    if had_image_marker:
        value = value[len(marker) :].strip()
    value = DOWNLOAD_ATTACHMENT_RE.sub("[FILE_ATTACHMENT]", value)
    value = EMAIL_RE.sub("[EMAIL]", value)
    value = URL_RE.sub("[URL]", value)
    value = DOMAIN_URL_RE.sub("[URL]", value)
    value = redact_sensitive_lines(value)
    value = CARD_RE.sub(replace_long_numeric_candidate, value)
    value = BD_PHONE_RE.sub("[PHONE]", value)
    value = PHONE_CANDIDATE_RE.sub(replace_phone_candidate, value)
    value = CARD_RE.sub(replace_long_numeric_candidate, value)
    value = value.replace("\n", " ")
    value = MULTISPACE_RE.sub(" ", value).strip()
    if had_image_marker:
        value = f"{marker} {value}".strip()
    return value


def prefix_image_marker(text: str) -> str:
    marker = "[IMAGE_ATTACHMENT]"
    if text.startswith(marker):
        return text
    return f"{marker} {text}".strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def row_to_csv_dict(row: DatasetRow) -> dict[str, str | int]:
    return {
        "conversation_id": row.conversation_id,
        "msg_id": row.msg_id,
        "role": row.role,
        "text_clean": row.text_clean,
        "has_image": row.has_image,
        "has_image_context": bool_text(row.has_image_context),
        "image_context_position": row.image_context_position,
        "refers_to_image": bool_text(row.refers_to_image),
        "intent": row.intent,
        "language": row.language,
        "tone": row.tone,
    }


def row_to_json_message(row: DatasetRow) -> dict[str, object]:
    message: dict[str, object] = {
        "msg_id": row.msg_id,
        "role": row.role,
        "content": row.text_clean,
        "has_image": bool(row.has_image),
        "has_image_context": row.has_image_context,
        "refers_to_image": row.refers_to_image,
    }
    if row.image_context_position:
        message["image_context_position"] = row.image_context_position
    if row.role == "user":
        message["intent"] = None
        message["language"] = None
        message["tone"] = None
    return message


def build_rows_for_conversation(
    conv_id: str,
    raw_messages_newest_first: list[RawMessage],
    business_sender: str,
    customer_name_patterns: list[re.Pattern[str]],
    business_name_patterns: list[re.Pattern[str]],
) -> tuple[list[DatasetRow], dict[str, int]]:
    raw_messages = list(reversed(raw_messages_newest_first))
    role_by_index: dict[int, str] = {}
    row_by_raw_index: dict[int, DatasetRow] = {}
    rows_in_order: list[tuple[int, DatasetRow]] = []
    stats = defaultdict(int)

    for idx, raw in enumerate(raw_messages):
        role = "assistant" if raw.sender == business_sender else "user"
        role_by_index[idx] = role
        text_clean = anonymize_text(raw.text, customer_name_patterns, business_name_patterns)
        has_image = bool(raw.image_refs)
        if text_clean:
            row = DatasetRow(
                conversation_id=conv_id,
                msg_id=0,
                role=role,
                text_clean=text_clean,
                has_image=1 if has_image else 0,
                has_image_context=has_image,
                refers_to_image=has_image,
            )
            if has_image:
                row.text_clean = prefix_image_marker(row.text_clean)
                stats["text_messages_with_own_image"] += 1
            row_by_raw_index[idx] = row
            rows_in_order.append((idx, row))
        elif has_image:
            stats["image_only_messages"] += 1
        else:
            stats["skipped_empty_or_non_image_messages"] += 1

    text_indices_by_role: dict[str, list[int]] = defaultdict(list)
    for raw_idx, row in rows_in_order:
        text_indices_by_role[row.role].append(raw_idx)

    for idx, raw in enumerate(raw_messages):
        if raw.text.strip() or not raw.image_refs:
            continue
        role = role_by_index[idx]
        candidates = text_indices_by_role.get(role, [])
        if not candidates:
            stats["unlinked_image_only_messages"] += 1
            continue
        nearest_idx = min(candidates, key=lambda candidate: (abs(candidate - idx), 0 if candidate > idx else 1))
        row = row_by_raw_index[nearest_idx]
        position = "prev" if idx < nearest_idx else "next"
        row.text_clean = prefix_image_marker(row.text_clean)
        row.has_image = 1
        row.has_image_context = True
        row.refers_to_image = True
        if not row.image_context_position:
            row.image_context_position = position
        elif row.image_context_position != position:
            row.image_context_position = "prev" if row.image_context_position == "prev" else position
        stats["linked_image_only_messages"] += 1

    final_rows = [row for _, row in sorted(rows_in_order, key=lambda item: item[0])]
    for msg_id, row in enumerate(final_rows, start=1):
        row.text_clean = final_pii_sanitize(row.text_clean)
        row.msg_id = msg_id
    stats["rows"] = len(final_rows)
    stats["user_rows"] = sum(1 for row in final_rows if row.role == "user")
    stats["assistant_rows"] = sum(1 for row in final_rows if row.role == "assistant")
    return final_rows, dict(stats)


def scan_pii(rows: list[DatasetRow]) -> dict[str, int]:
    phone_count = 0
    email_count = 0
    url_count = 0
    card_like_count = 0
    price_list_like_count = 0
    for row in rows:
        text = row.text_clean
        for match in PHONE_CANDIDATE_RE.finditer(text):
            if replace_phone_candidate(match) == "[PHONE]":
                phone_count += 1
        email_count += len(EMAIL_RE.findall(text))
        url_count += len(URL_RE.findall(text)) + len(DOMAIN_URL_RE.findall(text))
        for match in CARD_RE.finditer(text):
            numbers = [int(number) for number in re.findall(r"\d+", match.group(0))]
            if len(numbers) >= 4 and all(100 <= number <= 50000 for number in numbers):
                price_list_like_count += 1
            else:
                card_like_count += 1
    return {
        "phone_like_remaining": phone_count,
        "email_like_remaining": email_count,
        "url_like_remaining": url_count,
        "card_like_remaining": card_like_count,
        "price_list_numeric_sequences_remaining": price_list_like_count,
    }


def write_outputs(
    output_dir: Path,
    business_id: str,
    rows_by_conversation: dict[str, list[DatasetRow]],
    manifest: dict[str, object],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{business_id}_messages.csv"
    jsonl_path = output_dir / f"{business_id}_conversations.jsonl"
    manifest_path = output_dir / f"{business_id}_manifest.json"

    fieldnames = [
        "conversation_id",
        "msg_id",
        "role",
        "text_clean",
        "has_image",
        "has_image_context",
        "image_context_position",
        "refers_to_image",
        "intent",
        "language",
        "tone",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for conv_id in sorted(rows_by_conversation):
            for row in rows_by_conversation[conv_id]:
                writer.writerow(row_to_csv_dict(row))

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for conv_id in sorted(rows_by_conversation):
            record = {
                "conversation_id": conv_id,
                "messages": [row_to_json_message(row) for row in rows_by_conversation[conv_id]],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return csv_path, jsonl_path, manifest_path


def load_conversations(message_files: list[Path]) -> dict[Path, list[RawMessage]]:
    conversations: dict[Path, list[RawMessage]] = {}
    for index, path in enumerate(message_files, start=1):
        try:
            conversations[path.parent] = parse_message_file(path)
        except Exception as exc:  # pragma: no cover - defensive CLI behavior.
            source_hash = hashlib.sha256(str(path.parent).encode("utf-8")).hexdigest()[:12]
            print(f"warning: failed to parse source {source_hash}: {exc}", file=sys.stderr)
            conversations[path.parent] = []
        if index % 500 == 0:
            print(f"parsed {index}/{len(message_files)} message files", file=sys.stderr)
    return conversations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to a Meta messages/inbox directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV, JSONL, and manifest output.")
    parser.add_argument("--business-id", default="store_01", help="Anonymized business id prefix for conversation ids.")
    parser.add_argument("--business-sender", default=None, help="Optional exact sender name for assistant messages.")
    args = parser.parse_args()

    inbox_dir = args.input
    if not inbox_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {inbox_dir}")

    message_files = discover_message_files(inbox_dir)
    if not message_files:
        raise ValueError(f"No message_*.html files found under {inbox_dir}")

    conversations = load_conversations(message_files)
    business_sender = infer_business_sender(conversations, args.business_sender)
    all_senders = {msg.sender for messages in conversations.values() for msg in messages if msg.sender}
    customer_senders = all_senders - {business_sender}
    business_aliases = derive_business_aliases(inbox_dir, business_sender)
    customer_name_patterns = compile_name_patterns(customer_senders)
    business_name_patterns = compile_name_patterns(business_aliases, ascii_boundaries=False)

    rows_by_conversation: dict[str, list[DatasetRow]] = {}
    aggregate = defaultdict(int)
    for source_dir in sorted(conversations, key=lambda p: p.name):
        conv_id = make_conversation_id(args.business_id, source_dir.name)
        rows, stats = build_rows_for_conversation(
            conv_id,
            conversations[source_dir],
            business_sender,
            customer_name_patterns,
            business_name_patterns,
        )
        if rows:
            rows_by_conversation[conv_id] = rows
        for key, value in stats.items():
            aggregate[key] += value

    all_rows = [row for rows in rows_by_conversation.values() for row in rows]
    pii_scan = scan_pii(all_rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_inbox_dir_hash": hashlib.sha256(str(inbox_dir).encode("utf-8")).hexdigest()[:12],
            "message_html_files": len(message_files),
        },
        "business_id": args.business_id,
        "role_inference": {
            "method": "dominant_sender" if not args.business_sender else "explicit_sender",
            "assistant_sender_message_count": sum(
                1 for messages in conversations.values() for msg in messages if msg.sender == business_sender
            ),
        },
        "schema": {
            "row_level_csv": [
                "conversation_id",
                "msg_id",
                "role",
                "text_clean",
                "has_image",
                "has_image_context",
                "image_context_position",
                "refers_to_image",
                "intent",
                "language",
                "tone",
            ],
            "annotation_status": "intent/language/tone left blank/null as requested",
        },
        "anonymization": {
            "placeholders": ["[NAME]", "[BUSINESS]", "[PHONE]", "[ADDRESS]", "[PAYMENT_INFO]", "[EMAIL]", "[URL]"],
            "customer_sender_names_in_replacement_dictionary": len(customer_senders),
            "business_sender_names_in_replacement_dictionary": len(business_aliases),
        },
        "counts": {
            "conversations_with_rows": len(rows_by_conversation),
            "rows": len(all_rows),
            "user_rows": sum(1 for row in all_rows if row.role == "user"),
            "assistant_rows": sum(1 for row in all_rows if row.role == "assistant"),
            "rows_with_image_context": sum(1 for row in all_rows if row.has_image_context),
            **dict(sorted(aggregate.items())),
        },
        "pii_scan": pii_scan,
    }

    csv_path, jsonl_path, manifest_path = write_outputs(args.output_dir, args.business_id, rows_by_conversation, manifest)
    print(json.dumps({"csv": str(csv_path), "jsonl": str(jsonl_path), "manifest": str(manifest_path), **manifest["counts"], "pii_scan": pii_scan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
