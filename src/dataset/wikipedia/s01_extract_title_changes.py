#!/usr/bin/env python3

import os
import glob
import csv
import json

from tap import Tap
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime
from bisect import bisect_right
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

from dataset.wikipedia.misc.load_wiki_sql_tables import load_wiki_page_title_to_wiki_page_id_improved_chatgpt

import re

MW_NS = '{http://www.mediawiki.org/xml/export-0.11/}'

# Known MediaWiki namespace prefixes (English Wikipedia).
# Used to filter out non-main-namespace pages while preserving main-namespace
# articles that contain colons (e.g., "Batman: The Animated Series").
KNOWN_NS_PREFIXES = {
    'talk', 'user', 'user talk', 'wikipedia', 'wikipedia talk', 'file', 'file talk',
    'mediawiki', 'mediawiki talk', 'template', 'template talk', 'help', 'help talk',
    'category', 'category talk', 'portal', 'portal talk', 'draft', 'draft talk',
    'module', 'module talk', 'timedtext', 'timedtext talk', 'gadget', 'gadget talk',
    'gadget definition', 'gadget definition talk', 'special', 'media',
}


def _is_non_main_namespace(title: str) -> bool:
    """Return True if the title belongs to a known non-main namespace."""
    if ':' not in title:
        return False
    prefix = title.split(':', 1)[0].lower()
    return prefix in KNOWN_NS_PREFIXES


class Args(Tap):
    # JSON config
    config_file: str

    # Config-driven parameters (all optional on CLI)
    wiki_dump_dir: str | None = None
    path_wikipedia_page_info: str | None = None
    path_wikipedia_page_logs: str | None = None
    cache_dir: str | None = None
    output_dir_data: str | None = None



_RE_TARGET = re.compile(r's:\d+:"4::target";s:\d+:"([^"]+)"')

def extract_new_title_from_params(params_text: str) -> str | None:
    """
    Extract the new page title from MediaWiki <params>.

    Handles:
      - old format: <params>New Title</params>
      - new format: PHP-serialized params with key '4::target'
    """
    if not params_text:
        return None

    # Newer dumps: PHP-serialized params
    if params_text.startswith('a:'):
        m = _RE_TARGET.search(params_text)
        if m:
            return m.group(1)
        return None

    # Older dumps: params is already the title
    return params_text

def _parse_ts(ts: str) -> int:
    # '2005-07-03T06:06:25Z' -> unix seconds
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    return int(dt.timestamp())


def _next_move_after(
        moves_from: Dict[str, List[Tuple[int, str]]],
        title: str,
        t: int
) -> Optional[Tuple[int, str]]:
    """
    Given a title and time t, return the next move (ts, new_title)
    where this title appears as old_title with ts > t.
    """
    lst = moves_from.get(title)
    if not lst:
        return None
    # lst is sorted by ts
    idx = bisect_right(lst, (t, '\uffff'))
    if idx >= len(lst):
        return None
    return lst[idx]


def resolve_final_title(
        moves_from: Dict[str, List[Tuple[int, str]]],
        start_title: str,
        start_time: int,
        max_hops: int = 1000
) -> str:
    """
    Follow move chain forward in time starting from start_title at start_time.
    Returns the final title at the end of the log timeline.
    """
    title = start_title
    t = start_time
    hops = 0

    while hops < max_hops:
        nxt = _next_move_after(moves_from, title, t)
        if nxt is None:
            break
        t, title = nxt
        hops += 1

    return title


def resolve_page_id_from_title(
        title: str,
        title_to_id: Dict[str, int]
) -> int | None:
    """
    Resolve page_id from a title using conservative normalization.

    Tries:
      1) underscore form (MediaWiki canonical)
      2) space form (human-readable)
    """

    # Preferred: canonical MediaWiki form
    key_underscore = title.replace(' ', '_')
    page_id = title_to_id.get(key_underscore)
    if page_id is not None:
        return page_id

    # Fallback: space form (defensive)
    page_id = title_to_id.get(title)
    if page_id is not None:
        return page_id

    return None


def extract_page_moves_from_logging_dump_two_phase(
        logging_xml_gz_path: str,
        csv_writer,
        wikipedia_page_title_to_wikipedia_page_id: Dict[str, int],
        wikipedia_page_id_to_wikipedia_page_title: dict[int, str],
        log_every: int = 200_000
):
    """
    Two-phase extraction:
      1) read all move events and index future moves
      2) resolve page_id using final title and write TSV rows

    TSV schema:
      page_id, old_title, new_title, unix_timestamp, date
    """

    # Phase 1: collect events and build index: old_title -> [(ts, new_title), ...]
    moves_from: Dict[str, List[Tuple[int, str]]] = {}
    events: List[Tuple[int, str, str, str]] = []  # (ts, old_title, new_title, iso_ts)

    nr_logitems_seen = 0
    nr_move_events = 0

    with gzip.open(logging_xml_gz_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))
        for _, elem in context:
            if elem.tag != f'{MW_NS}logitem':
                continue

            nr_logitems_seen += 1

            log_type = elem.findtext(f'{MW_NS}type')
            action = elem.findtext(f'{MW_NS}action')
            if log_type != 'move' or action != 'move':
                elem.clear()
                continue

            old_title = elem.findtext(f'{MW_NS}logtitle')
            # new_title = elem.findtext(f'{MW_NS}params')
            raw_params = elem.findtext(f'{MW_NS}params')
            new_title = extract_new_title_from_params(raw_params)
            iso_ts = elem.findtext(f'{MW_NS}timestamp')

            if not old_title or not new_title or not iso_ts:
                elem.clear()
                continue

            # main namespace only
            if _is_non_main_namespace(old_title):
                elem.clear()
                continue

            ts = _parse_ts(iso_ts)

            events.append((ts, old_title, new_title, iso_ts))
            moves_from.setdefault(old_title, []).append((ts, new_title))
            nr_move_events += 1

            if nr_move_events % log_every == 0:
                print(f'[page-moves] phase1: collected {nr_move_events:,} moves '
                      f'({nr_logitems_seen:,} logitems scanned) from {logging_xml_gz_path}')

            elem.clear()

    # Sort per-title move lists by time (needed for bisect)
    for title, lst in moves_from.items():
        lst.sort(key=lambda x: x[0])

    print(f'[page-moves] phase1 DONE: collected {nr_move_events:,} moves '
          f'from {logging_xml_gz_path}. Resolving IDs and writing TSV...')

    # Phase 2: resolve ids and write
    written = 0
    unresolved = 0

    for ts, old_title, new_title, iso_ts in events:
        # final_title = resolve_final_title(moves_from, old_title, ts)
        final_title = resolve_final_title(moves_from, new_title, ts)

        # page_id = wikipedia_page_title_to_wikipedia_page_id.get(final_title)
        page_id = resolve_page_id_from_title(title=final_title,
                                             title_to_id=wikipedia_page_title_to_wikipedia_page_id)
        if page_id is None:
            print(f'unresolved: old_title: {old_title} -- '
                  f'new_title: {new_title} -- '
                  f'final_title: {final_title}')
            unresolved += 1
            continue

        dt = datetime.utcfromtimestamp(ts).date()

        csv_writer.writerow([
            page_id,
            old_title,
            new_title,
            ts,
            dt
        ])

        written += 1
        if written % log_every == 0:
            print(f'[page-moves] phase2: wrote {written:,} moves '
                  f'(unresolved so far: {unresolved:,}) from {logging_xml_gz_path}')

    print(f'[page-moves] DONE {logging_xml_gz_path}: wrote {written:,} moves '
          f'(unresolved: {unresolved:,}, total moves: {nr_move_events:,})')


# ---------------------------------------------------------------------
# Move extraction
# ---------------------------------------------------------------------
def extract_page_moves_from_logging_dump(
        logging_xml_gz_path: str,
        csv_writer: csv.writer,
        wikipedia_page_title_to_wikipedia_page_id: Dict[str, int],
        log_every: int = 100_000,
):
    '''
    Stream-parse MediaWiki pages-logging.xml.gz and write page move events:

        page_id, old_title, new_title, unix_timestamp, date
    '''

    nr_logitems_seen = 0
    nr_moves_written = 0

    with gzip.open(logging_xml_gz_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))

        for _, elem in context:
            if elem.tag != f'{MW_NS}logitem':
                continue

            nr_logitems_seen += 1

            log_type = elem.findtext(f'{MW_NS}type')
            action = elem.findtext(f'{MW_NS}action')

            if log_type != 'move' or action != 'move':
                elem.clear()
                continue

            old_title = elem.findtext(f'{MW_NS}logtitle')
            new_title = elem.findtext(f'{MW_NS}params')
            timestamp = elem.findtext(f'{MW_NS}timestamp')

            if not old_title or not new_title or not timestamp:
                elem.clear()
                continue

            # main namespace only
            if _is_non_main_namespace(old_title):
                elem.clear()
                continue

            page_id = wikipedia_page_title_to_wikipedia_page_id.get(new_title)
            if page_id is None:
                elem.clear()
                continue

            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            csv_writer.writerow([
                page_id,
                old_title,
                new_title,
                int(dt.timestamp()),
                dt.date()
            ])

            nr_moves_written += 1

            if nr_moves_written % log_every == 0:
                print(
                    f'[page-moves] {nr_moves_written:,} moves written '
                    f'({nr_logitems_seen:,} logitems scanned) '
                    f'from {logging_xml_gz_path}'
                )

            elem.clear()

    print(
        f'[page-moves] DONE {logging_xml_gz_path}: '
        f'{nr_moves_written:,} moves written '
        f'({nr_logitems_seen:,} logitems scanned)'
    )


def main():
    # first parse only --config (if present)
    arguments_main = Args().parse_args(known_only=True)
    print('After parse_args():', arguments_main.config_file)

    # Step 2: Load JSON config if provided
    if arguments_main.config_file:
        print(f'Loading JSON config from {arguments_main.config_file}')
        with open(arguments_main.config_file) as f:
            config_data = json.load(f)

        # Step 3: Merge JSON config into arguments (but keep CLI priority)
        # Only update fields not set via CLI
        for key, value in config_data.items():
            if (getattr(arguments_main, key, None) == Args().get_default(key)
                    or getattr(arguments_main, key) is None):
                setattr(arguments_main, key, value)

    # --------------------------------------------------------------
    # Step 2: resolve paths
    # --------------------------------------------------------------
    wiki_dump_dir = arguments_main.wiki_dump_dir
    page_sql_path = os.path.join(wiki_dump_dir, arguments_main.path_wikipedia_page_info)
    logs_dir = os.path.join(wiki_dump_dir, arguments_main.path_wikipedia_page_logs)

    os.makedirs(arguments_main.cache_dir, exist_ok=True)
    os.makedirs(arguments_main.output_dir_data, exist_ok=True)

    cache_title_to_id = os.path.join(
        arguments_main.cache_dir, 'wikipedia_page_title_to_page_id.pkl'
    )
    cache_id_to_title = os.path.join(
        arguments_main.cache_dir, 'wikipedia_page_id_to_page_title.pkl'
    )

    output_tsv = os.path.join(
        arguments_main.output_dir_data, 'page_title_changes.tsv'
    )

    # --------------------------------------------------------------
    # Step 3: load page title ↔ page id mapping (with cache)
    # --------------------------------------------------------------
    wikipedia_page_title_to_wikipedia_page_id, \
        wikipedia_page_id_to_wikipedia_page_title = (
        load_wiki_page_title_to_wiki_page_id_improved_chatgpt(
            path_cache_wikipedia_page_title_to_wikipedia_page_id=cache_title_to_id,
            path_cache_wikipedia_page_id_to_wikipedia_page_title=cache_id_to_title,
            path_wikipedia_page_info=page_sql_path
        )
    )

    if not wikipedia_page_title_to_wikipedia_page_id:
        raise RuntimeError('Failed to load page title → page_id mapping')

    # --------------------------------------------------------------
    # Step 4: extract moves from ALL logging dumps
    # --------------------------------------------------------------
    logging_files = sorted(
        glob.glob(os.path.join(logs_dir, '*.xml.gz'))
    )

    if not logging_files:
        raise RuntimeError(f'No logging XML files found in {logs_dir}')

    with open(output_tsv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t')

        for logging_xml in tqdm(logging_files):
            extract_page_moves_from_logging_dump_two_phase(
                logging_xml_gz_path=logging_xml,
                csv_writer=writer,
                wikipedia_page_title_to_wikipedia_page_id=wikipedia_page_title_to_wikipedia_page_id,
                wikipedia_page_id_to_wikipedia_page_title=wikipedia_page_id_to_wikipedia_page_title
            )

    print('BYE_BYE_I_AM_DONE')

if __name__ == '__main__':
    main()
