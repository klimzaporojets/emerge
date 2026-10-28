"""Safe reading of the Wikidata triple-history files written by the step-01 Java extractor.

The extractor writes one statement per line, TAB-separated and WITHOUT any quoting:

    <head QID>  <property>  <value>  <value type>  <history: ts:A,ts:D,...,P580:...,>

String values are written as they are, so a value that contains a line break splits its statement over several
physical lines, and the next physical line may start with a double quote. Reading such a file with `csv.reader`
(default quoting) turns that quote into the start of a quoted field that swallows every following line until the next
quote -- in the 2025 build, to the end of the file, losing the statements of whole id ranges (Eritrea, Bavaria, ...).

This module never interprets quotes. A physical line that does not start with `<head>\t<property>\t` continues the
previous statement (the line break belonged to its value); a value may also contain TABs (the value is everything
between the property and the last two fields). A statement whose last two fields are not a known value type and a
history that starts with an action stops the run with the file and line number: nothing is skipped silently.
Two rare statement kinds of the Feb-2025 dump are read as they are and counted (neither is an entity value, so neither
reaches the KG): value type `bad` (Wikidata's marker for an unparsable value; 42 coordinates edited on 2013-08-26) and
an empty value with an empty type (9,585 statements whose value the extractor could not read). An entity value that is
not an item (e.g. the entity schema `E250`) is skipped and counted by s02 (the 2025 code dropped such rows in a bare
`except`, unlogged in the summary).
"""
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

RECORD_START = re.compile(r'Q?\d+\tP\d+\t')
# datavalue types of Wikidata main snaks (wikidata.misc.SnapshotParser writes `datavalue.type` as it is); 'bad' is
# Wikidata's own type for a value it could not parse
VALUE_TYPES = frozenset({'wikibase-entityid', 'string', 'quantity', 'time', 'monolingualtext', 'globecoordinate',
                         'bad'})
ACTION = re.compile(r'\d+:[AD]')
HISTORY_START = re.compile(r'\d+:[AD](,|$)')
# value of a `wikibase-entityid` statement: an item (the KG's entities) or another Wikidata entity kind, e.g. an
# entity schema `E250` (the extractor takes its id when the value has no entity-type), or the invalid item `Q-1`
# (23 statements of May 2013, all removed again); these cannot be KG edges (Feb-2025 dump: 66 E…, 23 Q-1)
ITEM_ID = re.compile(r'Q?\d+')
OTHER_ENTITY_ID = re.compile(r'[A-PR-Z]\d+(-[FS]\d+)?|Q-\d+')


class MalformedHistoryRecord(ValueError):
    pass


@dataclass
class ReadStats:
    nr_records: int = 0
    nr_records_with_line_breaks: int = 0     # statements re-assembled from several physical lines
    nr_records_with_tabs_in_value: int = 0
    nr_dropped_history_fragments: int = 0    # history elements that are neither an action nor a P-qualifier
    nr_records_type_bad: int = 0             # value type 'bad' (unparsable value in Wikidata)
    nr_records_without_value: int = 0        # empty value AND empty type (the extractor could not read the value)
    nr_non_item_entity_values: int = 0       # entity value that is not an item (e.g. E250): skipped, not a KG edge
    non_item_entity_prefixes: Dict[str, int] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)

    def add_example(self, text: str, max_examples: int = 20):
        if len(self.examples) < max_examples:
            self.examples.append(text)

    def add_non_item_entity_value(self, value: str, where: str):
        prefix = 'Q-' if value.startswith('Q-') else value[0]
        self.nr_non_item_entity_values += 1
        self.non_item_entity_prefixes[prefix] = (self.non_item_entity_prefixes[prefix] + 1
                                                 if prefix in self.non_item_entity_prefixes else 1)
        self.add_example(f'non-item entity value {value!r} skipped at {where}')

    def merge(self, other: 'ReadStats'):
        """Adds the counts of `other` (the statistics of one file) to these."""
        for f in dataclasses.fields(self):
            if f.name == 'examples':
                for example in other.examples:
                    self.add_example(example)
            elif f.name == 'non_item_entity_prefixes':
                for prefix, n in other.non_item_entity_prefixes.items():
                    self.non_item_entity_prefixes[prefix] = (self.non_item_entity_prefixes[prefix] + n
                                                             if prefix in self.non_item_entity_prefixes else n)
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def sanitize_value(value: str) -> str:
    """A value must not contain the separators of the history files (one statement = one physical line)."""
    return value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')


def split_history(history: str, stats: ReadStats, where: str) -> Tuple[List[str], List[str]]:
    """'ts:A,ts:D,P580:Y2015MM3D22:11,' -> (['ts:A', 'ts:D'], ['P580:Y2015MM3D22:11']). Elements that are neither an
    action nor a P-qualifier (e.g. ':9', a qualifier without its property) carry no usable information: they are
    dropped and counted, the statement is kept."""
    actions, qualifiers = list(), list()
    for elem in history.split(','):
        if elem == '':
            continue
        if ACTION.fullmatch(elem):
            actions.append(elem)
        elif elem.startswith('P'):
            qualifiers.append(elem)
        else:
            stats.nr_dropped_history_fragments += 1
            stats.add_example(f'dropped history fragment {elem!r} at {where}')
    return actions, qualifiers


def _parse_record(lines: List[str], path: str, line_no: int, stats: ReadStats) -> List[str]:
    text = '\n'.join(lines)
    fields = text.split('\t')
    where = f'{path}:{line_no}'
    if len(fields) < 5:
        raise MalformedHistoryRecord(f'{where}: statement with {len(fields)} fields: {text[:300]!r}')
    value_type, history = fields[-2], fields[-1]
    value = '\t'.join(fields[2:-2])
    without_value = value_type == '' and value == ''
    if (value_type not in VALUE_TYPES and not without_value) or not HISTORY_START.match(history):
        raise MalformedHistoryRecord(f'{where}: last two fields are not <value type> <history>: {text[:300]!r}')
    if value_type == 'bad':
        stats.nr_records_type_bad += 1
    if without_value:
        stats.nr_records_without_value += 1
    if len(lines) > 1:
        stats.nr_records_with_line_breaks += 1
        stats.add_example(f'statement over {len(lines)} lines at {where}: {text[:120]!r}')
    if len(fields) > 5:
        stats.nr_records_with_tabs_in_value += 1
        stats.add_example(f'TAB in value at {where}: {text[:120]!r}')
    stats.nr_records += 1
    return [fields[0], fields[1], value, value_type, history]


def iter_raw_history_records(path: str, stats: ReadStats) -> Iterator[Tuple[int, List[str]]]:
    """Yield (first line number, [head, property, value, value_type, history]) for every statement of an extractor
    output file. Lines are split on '\\n' only, so a carriage return inside a value stays inside it."""
    buffer, buffer_line_no = None, 0
    with open(path, 'rt', encoding='utf-8', newline='\n') as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line[:-1] if line.endswith('\n') else line
            if RECORD_START.match(line):
                if buffer is not None:
                    yield buffer_line_no, _parse_record(buffer, path, buffer_line_no, stats)
                buffer, buffer_line_no = [line], line_no
            elif buffer is None:
                raise MalformedHistoryRecord(f'{path}:{line_no}: file does not start with a statement: {line[:300]!r}')
            else:
                buffer.append(line)
    if buffer is not None:
        yield buffer_line_no, _parse_record(buffer, path, buffer_line_no, stats)
