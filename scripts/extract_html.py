"""scripts/extract_html.py

Minimal extractor for Facebook Messenger HTML exports (Meta Business Suite style).

What it does:
- Extracts message texts (plain text) + role (customer/assistant) + stable random conversation_id.
- Does NOT mask or remove PII (per project workflow: do anonymization separately if needed).
- Does NOT store attachments; it only emits metadata flags/counts (has_image/has_link).
- Timestamps are only used to order messages chronologically and are not written to CSV.

Usage:
    python3 scripts/extract_html.py \
        --input_dir data/data_raw \
        --output data/data_processed/cleaned_messages.csv \
        --id_map data/data_processed/conversation_id_map.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


# -----------------------------
# HTML parsing + extraction
# -----------------------------


@dataclass
class ParsedMessage:
    sender_name: str
    text: str
    timestamp: Optional[datetime]
    has_image: bool
    image_count: int
    has_link: bool
    link_count: int


_TS_FORMAT = "%b %d, %Y %I:%M:%S %p"  # e.g., Jan 11, 2026 9:02:58 pm


def _safe_parse_timestamp(raw: str) -> Optional[datetime]:
    raw = (raw or "").strip()
    if not raw:
        return None
    raw = " ".join(raw.split())
    raw = re.sub(r"\s+(am|pm)\s*$", lambda m: " " + m.group(1).upper(), raw, flags=re.IGNORECASE)
    try:
        return datetime.strptime(raw, _TS_FORMAT)
    except Exception:
        return None


def _extract_message_text_and_meta(section) -> Tuple[str, bool, int, bool, int]:
    body = section.select_one("div._a6-p")
    if body is None:
        return "", False, 0, False, 0

    imgs = body.select("img")
    image_count = len(imgs)
    has_image = image_count > 0

    all_links = body.select("a[href]")
    link_count = 0
    for a in all_links:
        if a.select_one("img") is not None:
            continue
        link_count += 1
    has_link = link_count > 0

    # Remove reactions
    for ul in body.select("ul._a6-q"):
        ul.decompose()

    # Remove images and their wrapping anchors (we don't store attachment paths)
    for img in body.select("img"):
        parent_a = img.find_parent("a")
        if parent_a is not None:
            parent_a.decompose()
        else:
            img.decompose()

    # Remove remaining href targets but keep anchor text (if any)
    for a in body.select("a[href]"):
        a.replace_with(a.get_text(" ", strip=True))

    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, has_image, image_count, has_link, link_count


def parse_messenger_html(html_path: Path) -> Tuple[str, List[ParsedMessage]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")

    # In these exports, header h1 is the conversation title (customer name in 1:1 business chats)
    h1 = soup.select_one("header h1")
    customer_name = h1.get_text(strip=True) if h1 else ""

    messages: List[ParsedMessage] = []
    for section in soup.select("main section._a6-g"):
        sender_h2 = section.select_one("h2")
        sender_name = sender_h2.get_text(strip=True) if sender_h2 else ""

        text, has_image, image_count, has_link, link_count = _extract_message_text_and_meta(section)

        ts_raw_el = section.select_one("footer ._a72d")
        ts_raw = ts_raw_el.get_text(strip=True) if ts_raw_el else ""
        ts = _safe_parse_timestamp(ts_raw)

        messages.append(
            ParsedMessage(
                sender_name=sender_name,
                text=text,
                timestamp=ts,
                has_image=has_image,
                image_count=image_count,
                has_link=has_link,
                link_count=link_count,
            )
        )

    return customer_name, messages


# -----------------------------
# Dataset extraction
# -----------------------------


def find_message_htmls(input_dir: Path) -> List[Path]:
    """Find Messenger message HTML files under input_dir (recursively).

    Meta exports typically name files message_1.html, message_2.html, ...
    We intentionally ignore attachment folders like photos/.
    """
    files: List[Path] = []
    for fp in input_dir.rglob("message_*.html"):
        # Skip any attachment subtrees; they can contain non-message files.
        if "photos" in fp.parts:
            continue
        files.append(fp)
    return sorted(files)


def thread_key_for_file(html_path: Path) -> str:
    folder = str(html_path.parent.resolve())
    return hashlib.sha256(folder.encode("utf-8")).hexdigest()


def load_id_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_id_map(path: Path, id_map: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")


def get_or_create_conversation_id(id_map: Dict[str, str], thread_key: str) -> str:
    if thread_key in id_map:
        return id_map[thread_key]
    cid = str(uuid.uuid4())
    id_map[thread_key] = cid
    return cid


def infer_role(sender_name: str, customer_name: str) -> str:
    if sender_name and customer_name and sender_name.strip().lower() == customer_name.strip().lower():
        return "customer"
    return "assistant"


class MessengerHTMLExtractor:
    def __init__(self, id_map_path: Path):
        self.id_map_path = id_map_path
        self.id_map = load_id_map(id_map_path)
        self.rows: List[Dict[str, object]] = []

    def process_directory(self, input_dir: Path) -> None:
        html_files = find_message_htmls(input_dir)
        print(f"Found {len(html_files)} HTML files")

        threads: Dict[str, List[Path]] = {}
        for hp in html_files:
            threads.setdefault(thread_key_for_file(hp), []).append(hp)

        total_messages = 0
        for thread_key, files in sorted(threads.items(), key=lambda x: x[0]):
            conversation_id = get_or_create_conversation_id(self.id_map, thread_key)

            customer_name_final = ""
            all_msgs: List[ParsedMessage] = []
            for fp in sorted(files):
                try:
                    customer_name, msgs = parse_messenger_html(fp)
                except Exception as e:
                    print(f"Error processing {fp}: {e}")
                    continue

                if customer_name and not customer_name_final:
                    customer_name_final = customer_name
                all_msgs.extend(msgs)

            if any(m.timestamp is not None for m in all_msgs):
                all_msgs.sort(key=lambda m: (m.timestamp is None, m.timestamp or datetime.min))

            for msg_index, m in enumerate(all_msgs, start=1):
                role = infer_role(m.sender_name, customer_name_final)
                text_clean = (m.text or "").strip()
                is_media_only = int((not text_clean) and (m.has_image or m.has_link))

                self.rows.append(
                    {
                        "conversation_id": conversation_id,
                        "msg_id": msg_index,
                        "role": role,
                        "text": text_clean,
                        "has_image": int(bool(m.has_image)),
                        "image_count": int(m.image_count),
                        "has_link": int(bool(m.has_link)),
                        "link_count": int(m.link_count),
                        "is_media_only": is_media_only,
                    }
                )
                total_messages += 1

        save_id_map(self.id_map_path, self.id_map)
        print(f"Extracted {total_messages} messages from {len(threads)} conversations")

    def save_to_csv(self, output_path: Path) -> None:
        if not self.rows:
            print("No messages to save!")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "conversation_id",
            "msg_id",
            "role",
            "text",
            "has_image",
            "image_count",
            "has_link",
            "link_count",
            "is_media_only",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"✅ Saved {len(self.rows)} messages to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and anonymize Messenger HTML exports")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory to scan recursively for message_*.html files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/data_processed/cleaned_messages.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--id_map",
        type=str,
        default="data/data_processed/conversation_id_map.json",
        help="Where to persist stable random conversation IDs",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    id_map_path = Path(args.id_map)

    if not input_dir.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        return 2

    extractor = MessengerHTMLExtractor(id_map_path=id_map_path)
    extractor.process_directory(input_dir)
    extractor.save_to_csv(output_path)

    if extractor.rows:
        print(f"Extracted {len(extractor.rows)} messages")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())