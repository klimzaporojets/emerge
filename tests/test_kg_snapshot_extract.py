"""extract_snapshot (s03_get_kg_snapshot.py) writes the KG snapshot files we ship.

Covers the failure modes that matter for the released snapshots:
  - activity rule at the boundaries: a triple is in the snapshot at t iff added < t and (removed > t or never
    removed); added exactly at t or removed exactly at t -> NOT in the snapshot;
  - a missing entity/property label falls back to the id itself (released format), never to '';
  - non-ASCII labels are written as UTF-8;
  - without labels only the 3 id columns are written.
"""
import csv
import gzip
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    import torch_geometric  # noqa: F401
except ImportError:
    # The code under test only uses PyG's `Data` as an attribute container (unset attributes are None).
    # Stub it so the test also runs in environments without PyG (e.g. the `emerge` env).
    class _Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            return None

    _tg = types.ModuleType('torch_geometric')
    _tg_data = types.ModuleType('torch_geometric.data')
    _tg_utils = types.ModuleType('torch_geometric.utils')
    _tg_data.Data = _Data
    _tg_utils.to_networkx = None
    sys.modules.update({'torch_geometric': _tg, 'torch_geometric.data': _tg_data,
                        'torch_geometric.utils': _tg_utils})

from dataset.wikidata.python.s03_get_deltas import load_into_pyg_dynamically_data  # noqa: E402
from dataset.wikidata.python.s03_get_kg_snapshot import check_config, check_graph, check_labels, extract_snapshot, \
    check_expected_graph_size, load_entity_filter_qids, \
    input_files_fingerprint, snapshot_epoch  # noqa: E402

T = 200
INDEX_TO_ENTITY = {0: 1, 1: 2, 2: 3, 3: 4}  # index -> numeric QID
INDEX_TO_RELATION = {0: 'P31', 1: 'P17', 2: 'P999'}
#            (head, tail, relation, added, removed)
EDGES = [(0, 1, 0, 100, 0),     # Q1 P31 Q2        active (never removed)
         (0, 2, 1, 100, 150),   # Q1 P17 Q3        removed before t
         (1, 3, 0, 250, 0),     # Q2 P31 Q4        added after t
         (2, 0, 2, 199, 300),   # Q3 P999 Q1       active; property without label
         (1, 2, 1, T, 0),       # Q2 P17 Q3        added exactly at t -> not active
         (2, 1, 0, 100, T),     # Q3 P31 Q2        removed exactly at t -> not active
         (1, 3, 1, 50, 0)]      # Q2 P17 Q4        active; Q4 has no label
LABELS = {'Q1': 'Ärzte ✓', 'Q2': 'two', 'Q3': 'three'}
PROPERTY_LABELS = {'P31': 'instance of', 'P17': 'country'}


def _graph():
    edge_index = torch.tensor([[e[0] for e in EDGES], [e[1] for e in EDGES]], dtype=torch.long)
    edge_attr = torch.tensor([[i, e[2]] for i, e in enumerate(EDGES)], dtype=torch.long)
    edge_timestamps = torch.tensor([[i, e[3], e[4]] for i, e in enumerate(EDGES)], dtype=torch.long)
    return SimpleNamespace(edge_index=edge_index, edge_attr=edge_attr), edge_timestamps


def _read(path):
    with open(path, 'rt', encoding='utf-8', newline='') as f:
        return sorted(tuple(r) for r in csv.reader(f, delimiter='\t'))


def test_with_labels_activity_rule_and_qid_fallback(tmp_path):
    graph, edge_timestamps = _graph()
    # 1 of 4 entities has no label: far above the 10% production guard, allowed here to test the fallback
    stats = extract_snapshot(edge_timestamps, graph, T, INDEX_TO_ENTITY, INDEX_TO_RELATION, str(tmp_path),
                             wikidata_qid_to_label=LABELS, property_qid_to_label=PROPERTY_LABELS,
                             output_with_labels=True, max_missing_label_share=1.0)
    assert stats == {'nr_active_triples': 3, 'nr_entities': 4, 'nr_entities_without_label': 1,
                     'nr_relations_without_label': 1, 'nr_overlapping_intervals': 0, 'nr_duplicate_triples': 0}
    assert _read(tmp_path / 'triples.tsv') == sorted([
        ('Q1', 'P31', 'Q2', 'Ärzte ✓', 'instance of', 'two'),
        ('Q3', 'P999', 'Q1', 'three', 'P999', 'Ärzte ✓'),
        ('Q2', 'P17', 'Q4', 'two', 'country', 'Q4'),
    ])


def test_too_many_missing_labels_fail(tmp_path):
    """Labels from a file that does not belong to the graph must stop the run, not write id-only labels."""
    graph, edge_timestamps = _graph()
    with pytest.raises(ValueError, match='no label'):
        extract_snapshot(edge_timestamps, graph, T, INDEX_TO_ENTITY, INDEX_TO_RELATION, str(tmp_path),
                         wikidata_qid_to_label=LABELS, property_qid_to_label=PROPERTY_LABELS,
                         output_with_labels=True)        # default guard: 25% > 10%


def test_empty_snapshot_fails(tmp_path):
    graph, edge_timestamps = _graph()
    with pytest.raises(ValueError, match='empty'):
        extract_snapshot(edge_timestamps, graph, 10, INDEX_TO_ENTITY, INDEX_TO_RELATION, str(tmp_path))


def test_duplicates_and_overlapping_intervals_are_counted(tmp_path):
    # edge 0 active through two overlapping intervals; edge 1 repeats edge 0's (head, relation, tail)
    edge_index = torch.tensor([[0, 0], [1, 1]])
    edge_attr = torch.tensor([[0, 0], [1, 0]])
    edge_timestamps = torch.tensor([[0, 100, 0], [0, 150, 0], [1, 120, 0]])
    stats = extract_snapshot(edge_timestamps, SimpleNamespace(edge_index=edge_index, edge_attr=edge_attr), T,
                             {0: 1, 1: 2}, {0: 'P31'}, str(tmp_path))
    assert (stats['nr_active_triples'], stats['nr_overlapping_intervals'], stats['nr_duplicate_triples']) == (3, 1, 2)


def test_without_labels_three_columns(tmp_path):
    graph, edge_timestamps = _graph()
    stats = extract_snapshot(edge_timestamps, graph, T, INDEX_TO_ENTITY, INDEX_TO_RELATION, str(tmp_path),
                             output_with_labels=False)
    assert stats['nr_active_triples'] == 3
    assert _read(tmp_path / 'triples.tsv') == sorted([('Q1', 'P31', 'Q2'), ('Q3', 'P999', 'Q1'),
                                                      ('Q2', 'P17', 'Q4')])


def test_cache_fingerprint_changes_with_input_contents(tmp_path):
    """The graph cache must not be reused when an input file changed under the same path (this produced stale released snapshots once)."""
    history_dir = tmp_path / 'history'
    history_dir.mkdir()
    (history_dir / 'merged_file_sorted.csv').write_text('a\n')
    creation = tmp_path / 'creation.csv'
    creation.write_text('x\n')
    paths = [str(history_dir), str(creation)]
    first = input_files_fingerprint(paths)
    assert input_files_fingerprint(paths) == first                    # stable when nothing changed
    (history_dir / 'merged_file_sorted.csv').write_text('a\nb\n')    # same path, new contents (file in a dir)
    second = input_files_fingerprint(paths)
    assert second != first
    (history_dir / 'extra.csv').write_text('c\n')                    # a new file in the history dir
    assert input_files_fingerprint(paths) != second


def test_loader_end_to_end_with_wikipedia_filter_and_qualifier(tmp_path):
    """The real graph loader (as called by s03_get_kg_snapshot) on a tiny history file, then a snapshot from it.

    Regression: the loader read `wikidata_qid_to_label` as a global that exists only when s03_get_deltas runs as a
    script, so the first qualifier crashed a snapshot rebuild with a NameError.
    """
    history_dir = tmp_path / 'history'
    history_dir.mkdir()
    (history_dir / 'merged_file_sorted.csv').write_text(
        '1\tP31\t2\twikibase-entityid\t1500000000000:A,\n'                          # Q1 P31 Q2   kept
        '1\tP17\t3\twikibase-entityid\t1500000000000:A,1600000000000:D,\n'          # Q1 P17 Q3   removed 1.6e9
        '2\tP54\t3\twikibase-entityid\t1500000000000:A,P580:Y2015MM3D22:11,\n'      # Q2 P54 Q3   + qualifier
        '4\tP31\t2\twikibase-entityid\t1500000000000:A,\n'                          # Q4 not in Wikipedia
        '1\tP569\t+1952-03-11\ttime\t1500000000000:A,\n')                           # literal tail: not loaded
    out_dir = tmp_path / 'out'
    config = {'heads_in_wikipedia': True, 'tails_in_wikipedia': True, 'output_dir_data': str(out_dir)}
    labels = {'Q1': 'one', 'Q2': 'two', 'Q3': 'three', 'Q4': 'four'}
    property_labels = {'P31': 'instance of', 'P17': 'country', 'P54': 'member of sports team', 'P580': 'start time'}

    (graph, edge_timestamps, _, index_to_entity, relation_to_index, _, _) = load_into_pyg_dynamically_data(
        input_triples_path=str(history_dir), debug_nr_triples=-1, entity_to_index={}, relation_to_index={},
        qualifier_to_details=dict(), wikidata_qid_to_wikipedia_page_id={'Q1': 10, 'Q2': 20, 'Q3': 30},
        config=config, kb_type='wikidata', property_qid_to_label=property_labels,
        wikidata_qid_to_label=labels, precision='milliseconds')

    assert graph.edge_index.shape[1] == 3                     # Q4 row filtered, literal row skipped
    qualifiers = [json.loads(l) for l in (out_dir / 'qualifiers.jsonl').read_text().splitlines()]
    assert [(q['triple'], q['triple_str']) for q in qualifiers] == [
        (['Q2', 'P54', 'Q3'], ['two', 'member of sports team', 'three'])]

    index_to_relation = {v: k for k, v in relation_to_index.items()}
    snap_dir = tmp_path / 'snapshot'
    stats = extract_snapshot(edge_timestamps, graph, 1650000000, index_to_entity, index_to_relation, str(snap_dir),
                          wikidata_qid_to_label=labels, property_qid_to_label=property_labels,
                          output_with_labels=True)
    assert stats["nr_active_triples"] == 2                   # Q1 P17 Q3 was removed before t (ms -> s)
    assert _read(snap_dir / 'triples.tsv') == sorted([
        ('Q1', 'P31', 'Q2', 'one', 'instance of', 'two'),
        ('Q2', 'P54', 'Q3', 'two', 'member of sports team', 'three')])


def test_loader_stops_if_a_record_spans_several_lines(tmp_path):
    """A quoted value over two lines means an unbalanced quote may have swallowed the lines after it (the 2025 cleaning
    step lost whole id ranges that way): the loader must stop instead of silently reading fewer statements."""
    history_dir = tmp_path / 'history'
    history_dir.mkdir()
    (history_dir / 'merged_file_sorted.csv').write_text(
        '1\tP31\t2\twikibase-entityid\t1500000000000:A,\t{1}\n'
        '1\tP1448\t"multi\nline"\tmonolingualtext\t1500000000000:A,\t{1}\n'
        '2\tP31\t3\twikibase-entityid\t1500000000000:A,\t{2}\n')
    config = {'heads_in_wikipedia': True, 'tails_in_wikipedia': True, 'output_dir_data': str(tmp_path / 'out')}
    with pytest.raises(ValueError, match='spans 2 lines'):
        load_into_pyg_dynamically_data(
            input_triples_path=str(history_dir), debug_nr_triples=-1, entity_to_index={}, relation_to_index={},
            qualifier_to_details=dict(), wikidata_qid_to_wikipedia_page_id={'Q1': 10, 'Q2': 20, 'Q3': 30},
            config=config, kb_type='wikidata', property_qid_to_label={'P31': 'instance of'},
            wikidata_qid_to_label={'Q1': 'one', 'Q2': 'two', 'Q3': 'three'}, precision='milliseconds')


def test_snapshot_epochs_are_cet_midnight_as_in_the_2025_build():
    """The labels and the 2025 snapshots cut at 00:00 Europe/Amsterdam; epochs from the 2025 s05 log."""
    expected = {'2019-01-01': 1546297200, '2020-01-01': 1577833200, '2021-01-01': 1609455600,
                '2022-01-01': 1640991600, '2023-01-01': 1672527600, '2024-01-01': 1704063600,
                '2025-01-01': 1735686000}
    assert {d: snapshot_epoch(d, 'Europe/Amsterdam') for d in expected} == expected
    assert snapshot_epoch('2019-01-01', 'UTC') == 1546300800          # 00:00 GMT would be one hour later


def _config(tmp_path, **overrides):
    history = tmp_path / 'history'
    history.mkdir(exist_ok=True)
    (history / 'merged_file_sorted.csv').write_text('x\n')
    for name in ('props.tsv', 'labels.tsv', 'wdata_dates.csv', 'wpedia_dates.csv'):
        (tmp_path / name).write_text('x\n')
    config = {'snapshot_timestamps': ['2019-01-01', '2020-01-01'], 'precision_wdata': 'milliseconds',
              'snapshot_timezone': 'Europe/Amsterdam', 'path_extracted_history_triples_wdata': str(history),
              'path_property_labels': str(tmp_path / 'props.tsv'), 'output_with_labels': True,
              'path_wikidata_labels': str(tmp_path / 'labels.tsv'), 'heads_in_wikipedia': True,
              'tails_in_wikipedia': True, 'wdata_entity_creation_date_path': str(tmp_path / 'wdata_dates.csv'),
              'wpedia_entity_creation_date_path': str(tmp_path / 'wpedia_dates.csv')}
    config.update(overrides)
    return config


def test_check_config_fails_before_the_long_load(tmp_path):
    check_config(_config(tmp_path))                                   # valid
    bad = {'unsorted dates': {'snapshot_timestamps': ['2020-01-01', '2019-01-01']},
           'duplicate date': {'snapshot_timestamps': ['2019-01-01', '2019-01-01']},
           'malformed date': {'snapshot_timestamps': ['2019-13-01']},
           'no dates': {'snapshot_timestamps': []},
           'bad precision': {'precision_wdata': 'ms'},
           'bad timezone': {'snapshot_timezone': 'Mars/Olympus'},
           'missing labels file': {'path_wikidata_labels': str(tmp_path / 'nope.tsv')},
           'missing creation dates': {'wpedia_entity_creation_date_path': str(tmp_path / 'nope.csv')}}
    for case, override in bad.items():
        # ValueError (dates, precision), FileNotFoundError (inputs), KeyError (ZoneInfoNotFoundError)
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            check_config(_config(tmp_path, **override))


def test_check_labels_rejects_line_breaking_characters():
    check_labels('entity labels', {'Q1': 'Paris', 'Q2': 'say "hi"'})  # a double quote is fine (csv quotes it)
    for bad in ('tab\there', 'new\nline', 'carriage\rreturn'):
        with pytest.raises(ValueError, match='TAB/CR/LF'):
            check_labels('entity labels', {'Q1': bad})


def _real_graph(added=1500000000, removed=0):
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0, 0], [1, 1]])
    edge_timestamps = torch.tensor([[0, added, removed], [1, 1500000000, 0]])
    return SimpleNamespace(edge_index=edge_index, edge_attr=edge_attr), edge_timestamps


def test_check_graph_invariants():
    entities, relations, wiki = {0: 1, 1: 2, 2: 3}, {0: 'P31', 1: 'P17'}, {'Q1': 1, 'Q2': 2, 'Q3': 3}
    graph, ts = _real_graph()
    assert check_graph(graph, ts, entities, relations, wiki, both_filters=True)['nr_edges'] == 2
    with pytest.raises(ValueError, match='milliseconds'):             # timestamps left in milliseconds
        check_graph(*_real_graph(added=1500000000000), entities, relations, wiki, both_filters=True)
    with pytest.raises(ValueError, match='removal'):                  # removed before added
        check_graph(*_real_graph(removed=1400000000), entities, relations, wiki, both_filters=True)
    with pytest.raises(ValueError, match='Wikipedia filter'):         # an entity outside the filter
        check_graph(graph, ts, entities, relations, {'Q1': 1, 'Q2': 2}, both_filters=True)
    with pytest.raises(ValueError, match='P<digits>'):                # a relation id that is not a property
        check_graph(graph, ts, entities, {0: 'P31', 1: 'Q17'}, wiki, both_filters=True)
    with pytest.raises(ValueError, match='outside'):                  # an edge pointing to an unknown entity
        check_graph(graph, ts, {0: 1, 1: 2}, relations, wiki, both_filters=True)


def test_load_entity_filter_qids(tmp_path):
    """The explicit entity filter (the exact entity set of the graph the labels came from)."""
    plain = tmp_path / 'qids.txt'
    plain.write_text('Q1\nQ22\nQ333\n')
    packed = tmp_path / 'qids.txt.gz'
    with gzip.open(packed, 'wt', encoding='utf-8') as f:
        f.write('Q1\nQ22\nQ333\n')
    assert load_entity_filter_qids(str(plain)) == load_entity_filter_qids(str(packed)) == dict.fromkeys(['Q1', 'Q22', 'Q333'])
    for bad, match in (('Q1\nP31\n', 'not a QID'), ('Q1\n\nQ2\n', 'not a QID'), ('Q1\nQ1\n', 'duplicate'),
                       ('', 'empty'), ('Q01\n', 'not a QID')):
        path = tmp_path / 'bad.txt'
        path.write_text(bad)
        with pytest.raises(ValueError, match=match):
            load_entity_filter_qids(str(path))


def test_check_config_with_entity_filter_list(tmp_path):
    base = _config(tmp_path)
    for key in ('wdata_entity_creation_date_path', 'wpedia_entity_creation_date_path'):
        del base[key]                                                  # not needed with an explicit list
    (tmp_path / 'qids.txt.gz').write_bytes(gzip.compress(b'Q1\n'))
    check_config({**base, 'path_entity_filter_qids': str(tmp_path / 'qids.txt.gz')})
    with pytest.raises(ValueError, match='heads AND tails'):
        check_config({**base, 'path_entity_filter_qids': str(tmp_path / 'qids.txt.gz'), 'tails_in_wikipedia': False})
    with pytest.raises(FileNotFoundError):
        check_config({**base, 'path_entity_filter_qids': str(tmp_path / 'missing.txt.gz')})
    with pytest.raises(KeyError):                                     # without a list the creation-date files are needed
        check_config(base)


def test_loader_with_entity_filter_list_keeps_exactly_the_listed_entities(tmp_path):
    history_dir = tmp_path / 'history'
    history_dir.mkdir()
    (history_dir / 'merged_file_sorted.csv').write_text(
        '1\tP31\t2\twikibase-entityid\t1500000000000:A,\n'                          # Q1-Q2 both listed: kept
        '2\tP54\t3\twikibase-entityid\t1500000000000:A,\n'                          # Q3 not listed: dropped
        '4\tP31\t1\twikibase-entityid\t1500000000000:A,\n')                         # Q4 not listed: dropped
    (tmp_path / 'qids.txt').write_text('Q1\nQ2\n')
    graph, *_ = load_into_pyg_dynamically_data(
        input_triples_path=str(history_dir), debug_nr_triples=-1, entity_to_index={}, relation_to_index={},
        qualifier_to_details=dict(), wikidata_qid_to_wikipedia_page_id=load_entity_filter_qids(str(tmp_path / 'qids.txt')),
        config={'heads_in_wikipedia': True, 'tails_in_wikipedia': True, 'output_dir_data': str(tmp_path / 'out')},
        kb_type='wikidata', property_qid_to_label={'P31': 'instance of', 'P54': 'member of'},
        wikidata_qid_to_label={}, precision='milliseconds')
    assert graph.edge_index.shape[1] == 1


def test_expected_graph_size_stops_a_wrong_graph():
    """A rebuild from later inputs gave 45,663,893 triples instead of the labels' 45,718,765 — must stop before writing."""
    check_expected_graph_size({}, {'nr_edges': 1, 'nr_entities': 1})                       # keys absent: no check
    good = {'expected_nr_graph_triples': 45718765, 'expected_nr_graph_entities': 6317075}
    check_expected_graph_size(good, {'nr_edges': 45718765, 'nr_entities': 6317075})
    with pytest.raises(ValueError, match='expected_nr_graph_triples'):
        check_expected_graph_size(good, {'nr_edges': 45663893, 'nr_entities': 6317075})
    with pytest.raises(ValueError, match='expected_nr_graph_entities'):
        check_expected_graph_size(good, {'nr_edges': 45718765, 'nr_entities': 6305878})
