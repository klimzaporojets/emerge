"""Reading the Wikidata history files without losing statements (step 02b: pass 1, sort, pass 2).

Regression: the 2025 run read the step-01 output with `csv.reader` (default quoting). A string value with a line
break made the next physical line start with '"'; the reader took it as an open quoted field and swallowed every
following line to the end of the file. The statements of whole id ranges (Eritrea, Bavaria, European route E6, ...)
vanished from the history, the graph and every KG snapshot, and nothing was logged.

<1 s, CPU only, synthetic files.
"""
import csv
import dataclasses
import sys
from pathlib import Path

import pytest

from dataset.wikidata.python.misc.history_tsv import (MalformedHistoryRecord, ReadStats, iter_raw_history_records,
                                                      sanitize_value, split_history)
from dataset.wikidata.python.s02_normalize_history_graph import normalize_history_file, process_history_files

REPO = Path(__file__).resolve().parents[1]


def _entity_lines(heads):
    return ''.join(f'Q{h}\tP31\t5\twikibase-entityid\t1500000000000:A,\n' for h in heads)


# a value ending with a line break, then a value whose second line starts with a double quote (as in the 2025 files)
BROKEN_VALUES = ('Q960\tP48\tTooruktug Dolgai Tangdym.ogg\n\tstring\t1368145691000:A,\n'
                 'Q961\tP1448\tsome text\n"quoted continuation\tmonolingualtext\t1400000000000:A,\n')


def test_csv_quoting_loses_the_statements_after_a_quote_and_the_new_reader_keeps_them(tmp_path):
    path = tmp_path / 'history.csv'
    path.write_text(BROKEN_VALUES + _entity_lines(range(100, 150)), encoding='utf-8')

    csv.field_size_limit(sys.maxsize)                       # as the 2025 run did
    with open(path, encoding='utf-8') as f:
        old_rows = [r for r in csv.reader(f, delimiter='\t') if len(r) >= 5 and r[3] == 'wikibase-entityid']
    assert old_rows == []                                   # all 50 statements swallowed by the quote

    stats = ReadStats()
    records = list(iter_raw_history_records(str(path), stats))
    assert len(records) == 52
    assert records[0] == (1, ['Q960', 'P48', 'Tooruktug Dolgai Tangdym.ogg\n', 'string', '1368145691000:A,'])
    assert records[1][1][2] == 'some text\n"quoted continuation'
    assert [r[1][0] for r in records[2:]] == [f'Q{h}' for h in range(100, 150)]
    assert stats.nr_records == 52 and stats.nr_records_with_line_breaks == 2


def test_tab_inside_a_value_is_kept_in_the_value(tmp_path):
    path = tmp_path / 'history.csv'
    path.write_text('Q1011\tP474\t "\t+238\tstring\t1367618492000:A,1367618501000:D,\n', encoding='utf-8')
    stats = ReadStats()
    [(_, record)] = list(iter_raw_history_records(str(path), stats))
    assert record == ['Q1011', 'P474', ' "\t+238', 'string', '1367618492000:A,1367618501000:D,']
    assert stats.nr_records_with_tabs_in_value == 1


def test_history_fragments_without_property_are_dropped_and_counted():
    stats = ReadStats()
    assert split_history('1603721000000:A,1603721344000:D,:9,', stats, 'f:1') == (
        ['1603721000000:A', '1603721344000:D'], [])
    assert split_history('1500000000000:A,P580:Y2015MM3D22:11,', stats, 'f:2') == (
        ['1500000000000:A'], ['P580:Y2015MM3D22:11'])
    assert stats.nr_dropped_history_fragments == 1


@pytest.mark.parametrize('content, where', [
    ('\tstring\t1368145691000:A,\n' + _entity_lines([1]), ':1:'),          # file starts inside a value
    ('Q1\tP31\tfoo\n' + _entity_lines([2]), ':1:'),                        # statement without type and history
    ('Q1\tP31\t5\tnot-a-type\t1500000000000:A,\n', ':1:'),                 # unknown value type
    ('Q1\tP31\t5\t\t1500000000000:A,\n', ':1:'),                           # empty type but a value
])
def test_a_statement_that_cannot_be_delimited_stops_the_run(tmp_path, content, where):
    path = tmp_path / 'history.csv'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(MalformedHistoryRecord, match=where):
        list(iter_raw_history_records(str(path), ReadStats()))


def test_bad_type_and_statements_without_value_are_read_and_counted(tmp_path):
    # both occur in the Feb-2025 dump (2026-03-11 extraction): 42 coordinates typed 'bad' (Q712 below is the first
    # statement found in the 2025 extraction) and 9,585 statements with an empty value and an empty type
    path = tmp_path / 'history.csv'
    path.write_text('Q712\tP625\tlat-18:lon179:\tbad\t1377552865000:A,1377627344000:D,\n'
                    'Q851\tP610\t\t\t1411992867000:A,1411992905000:D,\n' + _entity_lines([852]), encoding='utf-8')
    stats = ReadStats()
    records = [r for _, r in iter_raw_history_records(str(path), stats)]
    assert records[0] == ['Q712', 'P625', 'lat-18:lon179:', 'bad', '1377552865000:A,1377627344000:D,']
    assert records[1] == ['Q851', 'P610', '', '', '1411992867000:A,1411992905000:D,']
    assert records[2][0] == 'Q852'
    assert stats.nr_records == 3 and stats.nr_records_type_bad == 1 and stats.nr_records_without_value == 1


def test_sanitize_value_removes_the_separators():
    assert sanitize_value('a\tb\nc\r"d"') == 'a b c "d"'


def _read_output(path):
    with open(path, encoding='utf-8', newline='') as f:
        lines = f.read().splitlines()
    return [line.split('\t') for line in lines]


def test_pass1_sort_pass2_keeps_every_statement_and_merges_redirects(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    # file A: a broken string value right before entity statements (the 2025 trigger); Q4 redirects to Q1
    (raw / 'a.csv').write_text('Q1\tP31\t5\twikibase-entityid\t1500000000000:A,\n' + BROKEN_VALUES +
                               _entity_lines([2, 3]), encoding='utf-8')
    # file B also holds the two rare kinds of the Feb-2025 dump ('bad' type; empty value + type): kept, not entities
    (raw / 'b.csv').write_text('Q4\tP31\t5\twikibase-entityid\t1500000000000:A,\n'
                               'Q712\tP625\tlat-18:lon179:\tbad\t1377552865000:A,\n'
                               'Q851\tP610\t\t\t1411992867000:A,\n', encoding='utf-8')
    redirects = {4: 1}
    pass1 = tmp_path / 'pass1'
    pass1.mkdir()
    stats = ReadStats()
    for name in ('a.csv', 'b.csv'):
        normalize_history_file(str(raw / name), str(pass1 / name), 'raw_extraction', redirects, 168,
                               'milliseconds', stats)
    assert not list(pass1.glob('*.partial'))

    rows = _read_output(pass1 / 'a.csv')
    assert all(len(r) == 6 for r in rows)                   # one statement per physical line
    assert ['960', 'P48', 'Tooruktug Dolgai Tangdym.ogg ', 'string'] in [r[:4] for r in rows]

    # s02_cat_sort.sh: cat + sort in byte order (Python's sort of str = code-point order)
    lines = [line for name in ('a.csv', 'b.csv') for line in (pass1 / name).read_text(encoding='utf-8').splitlines()]
    cat = tmp_path / 'cat'
    cat.mkdir()
    (cat / 'merged_file_sorted.csv').write_text(''.join(f'{line}\r\n' for line in sorted(lines)), encoding='utf-8')

    out = tmp_path / 'pass2.csv'
    normalize_history_file(str(cat / 'merged_file_sorted.csv'), str(out), 'normalized', redirects, 168,
                           'milliseconds', ReadStats())
    entity_rows = [r for r in _read_output(out) if r[3] == 'wikibase-entityid']
    assert sorted((r[0], r[1], r[2]) for r in entity_rows) == [('1', 'P31', '5'), ('2', 'P31', '5'), ('3', 'P31', '5')]
    [q1] = [r for r in entity_rows if r[0] == '1']
    assert q1[4] == '1500000000000:A'                       # Q1 and its redirect Q4 merged into one history
    rows2 = _read_output(out)
    assert all(len(r) == 6 for r in rows2)
    assert ['712', 'P625', 'lat-18:lon179:', 'bad', '1377552865000:A'] in [r[:5] for r in rows2]
    assert ['851', 'P610', '', '', '1411992867000:A'] in [r[:5] for r in rows2]


def test_non_item_entity_value_is_skipped_and_counted_and_garbage_stops(tmp_path):
    # Q127448 P12861 E250 (an entity schema) occurs in the 2025 extraction; the 2025 code dropped it in a bare `except`
    raw = tmp_path / 'raw.csv'
    # and the invalid item Q-1 (23 statements of May 2013 in the Feb-2025 dump, all removed again)
    raw.write_text('Q127448\tP12861\tE250\twikibase-entityid\t1720076599000:A,\n'
                   'Q12404\tP47\tQ-1\twikibase-entityid\t1368102425000:A,\n' + _entity_lines([127449]),
                   encoding='utf-8')
    stats = ReadStats()
    normalize_history_file(str(raw), str(tmp_path / 'out.csv'), 'raw_extraction', {}, 168, 'milliseconds', stats)
    rows = _read_output(tmp_path / 'out.csv')
    assert [r[:3] for r in rows] == [['127449', 'P31', '5']]
    assert stats.nr_non_item_entity_values == 2 and stats.non_item_entity_prefixes == {'E': 1, 'Q-': 1}

    raw.write_text('Q1\tP31\tnot-an-id\twikibase-entityid\t1720076599000:A,\n', encoding='utf-8')
    with pytest.raises(MalformedHistoryRecord, match='not an entity id'):
        normalize_history_file(str(raw), str(tmp_path / 'out2.csv'), 'raw_extraction', {}, 168, 'milliseconds',
                               ReadStats())


def test_resumed_run_reports_the_statistics_of_every_file(tmp_path):
    raw, out, stats_dir = tmp_path / 'raw', tmp_path / 'out', tmp_path / 'stats'
    raw.mkdir()
    (raw / 'a.csv').write_text(BROKEN_VALUES + _entity_lines([1, 2]), encoding='utf-8')
    (raw / 'b.csv').write_text('Q127448\tP12861\tE250\twikibase-entityid\t1720076599000:A,\n' + _entity_lines([3]),
                               encoding='utf-8')
    full, nr_files, nr_skipped = process_history_files(str(raw), str(out), str(stats_dir), 'raw_extraction', {}, 168,
                                                       'milliseconds')
    assert (nr_files, nr_skipped) == (2, 0)
    assert full.nr_records == 6 and full.nr_records_with_line_breaks == 2 and full.non_item_entity_prefixes == {'E': 1}

    # a.csv finished before per-file statistics existed (output only) -> processed again; b.csv is skipped
    (stats_dir / 'a.csv.json').unlink()
    resumed, _, nr_skipped = process_history_files(str(raw), str(out), str(stats_dir), 'raw_extraction', {}, 168,
                                                   'milliseconds')
    assert nr_skipped == 1
    assert dataclasses.asdict(resumed) == dataclasses.asdict(full)
    assert sorted(p.name for p in out.iterdir()) == ['a.csv', 'b.csv']    # no stats or partial files in the output


def test_pass2_stops_if_a_record_spans_several_lines(tmp_path):
    path = tmp_path / 'merged_file_sorted.csv'
    with open(path, 'wt', encoding='utf-8', newline='') as f:
        csv.writer(f, delimiter='\t').writerow([1, 'P1448', 'multi\nline', 'monolingualtext', '1500000000000:A', '{1}'])
    with pytest.raises(MalformedHistoryRecord, match='one line with 6 fields'):
        normalize_history_file(str(path), str(tmp_path / 'out.csv'), 'normalized', {}, 168, 'milliseconds',
                               ReadStats())


def test_cat_sort_uses_byte_order():
    script = (REPO / 'scripts' / 'slurm' / 'wikidata' / 's02_cat_sort.sh').read_text()
    assert 'export LC_ALL=C' in script.split('sort -S')[0]
    # byte order keeps the (subject, property) groups contiguous; ignoring the TAB would interleave P107 and P1070
    lines = ['978\tP107\t0123', '978\tP1070\t5', '978\tP107\t1969448']
    assert [line.split('\t')[1] for line in sorted(lines)] == ['P107', 'P107', 'P1070']
