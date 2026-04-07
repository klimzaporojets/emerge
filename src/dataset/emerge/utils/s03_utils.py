import bisect
import re
from typing import Dict, List, Set


# def split_text(
#         text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
# ):
def split_text(
        text: str, tokenizer, max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.

    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.

    Returns:
        List[str]: A list of text chunks.
    """
    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue

        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)

            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:

                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def obtain_revision_ids_to_extract_stub(
        timestamps: List[int],
        revision_history: List,
        min_stability_span_in_hs: int):
    timestamps_len = len(timestamps)

    revision_id_to_timestamps: Dict[int, Set[int]] = dict()
    timestamp_to_revision_id: Dict[int, int] = dict()
    for curr_timestamp in timestamps:
        revision_id_to_timestamps[revision_history[0][1]] = {curr_timestamp}
        timestamp_to_revision_id[curr_timestamp] = revision_history[0][1]
    assert timestamps_len == len(timestamp_to_revision_id)
    return timestamp_to_revision_id, revision_id_to_timestamps


def obtain_revision_ids_to_extract_binary(
        timestamps: List[int],
        revision_history: List,
        min_stability_span_in_hs: int):
    prev_revision_timestamp = None
    prev_revision_id = None
    min_stability_span_in_sec = min_stability_span_in_hs * 60 * 60
    revision_id_to_timestamps: Dict[int, Set[int]] = dict()
    timestamp_to_revision_id: Dict[int, int] = dict()
    idx_last_found_timestamp = 0
    timestamps_len = len(timestamps)

    sorted_list = [curr_revision[0] for curr_revision in revision_history]
    for curr_timestamp in timestamps:
        if curr_timestamp not in timestamp_to_revision_id:
            index = bisect.bisect_left(sorted_list, curr_timestamp)
            if index == 0:
                revision_id = revision_history[index][1]
            else:
                revision_id = revision_history[index - 1][1]
            timestamp_to_revision_id[curr_timestamp] = revision_id

            if revision_id not in revision_id_to_timestamps:
                revision_id_to_timestamps[revision_id] = set()
            revision_id_to_timestamps[revision_id].add(curr_timestamp)
    # for idx_revision, curr_revision in enumerate(revision_history):
    #     curr_revision_timestamp = curr_revision[0]
    #     curr_revision_id = curr_revision[1]
    #     curr_revision_margin = None
    #     if prev_revision_timestamp is not None:
    #         curr_revision_margin = curr_revision_timestamp - prev_revision_timestamp
    #
    #     for curr_timestamp_idx, curr_searched_timestamp in enumerate(timestamps[idx_last_found_timestamp:]):
    #         if curr_revision_timestamp > curr_searched_timestamp:
    #             if prev_revision_timestamp is None:
    #                 revision_id = curr_revision_id
    #             # elif curr_revision_margin is not None:
    #             else:
    #                 if curr_revision_margin > min_stability_span_in_sec:
    #                     revision_id = prev_revision_id
    #                 else:
    #                     revision_id = curr_revision_id
    #             if revision_id not in revision_id_to_timestamps:
    #                 revision_id_to_timestamps[revision_id] = set()
    #             revision_id_to_timestamps[revision_id].add(curr_searched_timestamp)
    #             if curr_searched_timestamp not in timestamp_to_revision_ids:
    #                 timestamp_to_revision_ids[curr_searched_timestamp] = set()
    #             timestamp_to_revision_ids[curr_searched_timestamp].add(revision_id)
    #             idx_last_found_timestamp += 1
    #         else:
    #             break
    #
    #     prev_revision_timestamp = curr_revision_timestamp
    #     prev_revision_id = curr_revision_id
    #     #
    #     if timestamps_len == len(timestamp_to_revision_ids):
    #         return timestamp_to_revision_ids, revision_id_to_timestamps
    # # pads with the last revision
    # assert prev_revision_id is not None
    # for curr_timestamp_idx, curr_searched_timestamp \
    #         in enumerate(timestamps[idx_last_found_timestamp:]):
    #     # adjusted_idx = curr_timestamp_idx + idx_last_found_timestamp
    #     if curr_searched_timestamp not in timestamp_to_revision_ids:
    #         timestamp_to_revision_ids[curr_searched_timestamp] = set()
    #     timestamp_to_revision_ids[curr_searched_timestamp].add(prev_revision_id)
    #     if prev_revision_id not in revision_id_to_timestamps:
    #         revision_id_to_timestamps[prev_revision_id] = set()
    #     revision_id_to_timestamps[prev_revision_id].add(curr_searched_timestamp)

    assert timestamps_len == len(timestamp_to_revision_id)
    return timestamp_to_revision_id, revision_id_to_timestamps


def obtain_revision_ids_to_extract(
        timestamps: List[int],
        revision_history: List,
        min_stability_span_in_hs: int):
    prev_revision_timestamp = None
    prev_revision_id = None
    prev_revision_margin = None
    min_stability_span_in_sec = min_stability_span_in_hs * 60 * 60
    revision_id_to_timestamps = dict()
    timestamp_to_revision_ids = dict()
    idx_last_found_timestamp = 0
    timestamps_len = len(timestamps)
    for idx_revision, curr_revision in enumerate(revision_history):
        curr_revision_timestamp = int(curr_revision[0])
        curr_revision_id = int(curr_revision[1])
        curr_revision_margin = None
        if prev_revision_timestamp is not None:
            curr_revision_margin = curr_revision_timestamp - prev_revision_timestamp

        for curr_timestamp_idx, curr_searched_timestamp in enumerate(timestamps[idx_last_found_timestamp:]):
            # adjusted_idx = curr_timestamp_idx + idx_last_found_timestamp
            # revision_id = None
            if curr_revision_timestamp > curr_searched_timestamp:
                if prev_revision_timestamp is None:
                    revision_id = curr_revision_id
                # elif curr_revision_margin is not None:
                else:
                    if curr_revision_margin > min_stability_span_in_sec:
                        revision_id = prev_revision_id
                    else:
                        revision_id = curr_revision_id
                if revision_id not in revision_id_to_timestamps:
                    revision_id_to_timestamps[revision_id] = set()
                revision_id_to_timestamps[revision_id].add(curr_searched_timestamp)
                if curr_searched_timestamp not in timestamp_to_revision_ids:
                    timestamp_to_revision_ids[curr_searched_timestamp] = set()
                timestamp_to_revision_ids[curr_searched_timestamp].add(revision_id)
                idx_last_found_timestamp += 1
            else:
                break

        prev_revision_timestamp = curr_revision_timestamp
        prev_revision_id = curr_revision_id
        # prev_revision_margin = curr_revision_margin
        #
        # if revision_id_to is not None and revision_id_from is not None:
        #     return revision_id_to, revision_id_from
        if timestamps_len == len(timestamp_to_revision_ids):
            return timestamp_to_revision_ids, revision_id_to_timestamps
    # pads with the last revision
    assert prev_revision_id is not None
    for curr_timestamp_idx, curr_searched_timestamp \
            in enumerate(timestamps[idx_last_found_timestamp:]):
        # adjusted_idx = curr_timestamp_idx + idx_last_found_timestamp
        if curr_searched_timestamp not in timestamp_to_revision_ids:
            timestamp_to_revision_ids[curr_searched_timestamp] = set()
        timestamp_to_revision_ids[curr_searched_timestamp].add(prev_revision_id)
        if prev_revision_id not in revision_id_to_timestamps:
            revision_id_to_timestamps[prev_revision_id] = set()
        revision_id_to_timestamps[prev_revision_id].add(curr_searched_timestamp)

    assert timestamps_len == len(timestamp_to_revision_ids)
    return timestamp_to_revision_ids, revision_id_to_timestamps


def get_page_id_to_revs(page_id: int,
                        timespans_2_tail_ids: Dict,
                        page_id_to_rev_to_tail_ids: Dict,
                        timestamps_revisions: List,
                        min_stability_span_in_hs: int,
                        page_id_to_rev_to_timestamps: Dict
                        # , page_id_to_main_page_id:Dict
                        ):
    # if page_id in head_id_2_timespans_2_tail_ids or \
    #         (page_id in page_id_to_main_page_id and
    #          page_id_to_main_page_id[page_id] in head_id_2_timespans_2_tail_ids):
    # if page_id in head_id_2_timespans_2_tail_ids:
    timestamps = timespans_2_tail_ids.keys()
    # timespans = list(set([item for sublist in timespans for item in sublist]))
    timestamps = list(set(timestamps))
    # timestamp_to_revision_ids, revision_id_to_timestamps = \
    #     obtain_revision_ids_to_extract_stub(
    #         timestamps=timespans,
    #         revision_history=timestamps_revisions,
    #         min_stability_span_in_hs=min_stability_span_in_hs
    #     )
    timestamp_to_revision_ids, revision_id_to_timestamps = \
        obtain_revision_ids_to_extract(
            timestamps=timestamps,
            revision_history=timestamps_revisions,
            min_stability_span_in_hs=min_stability_span_in_hs
        )
    # timestamp_to_revision_ids, revision_id_to_timestamps = \
    #     obtain_revision_ids_to_extract_binary(
    #         timestamps=timespans,
    #         revision_history=timestamps_revisions,
    #         min_stability_span_in_hs=min_stability_span_in_hs
    #     )
    page_id_to_rev_to_timestamps[page_id] = revision_id_to_timestamps
    page_id_to_rev_to_tail_ids[page_id] = dict()
    for curr_revision_id2, curr_timestamps in revision_id_to_timestamps.items():
        page_id_to_rev_to_tail_ids[page_id][curr_revision_id2] = set()
        for curr_timestamp2 in curr_timestamps:
            page_id_to_rev_to_tail_ids[page_id][curr_revision_id2] \
                .update(timespans_2_tail_ids[curr_timestamp2])

    return page_id_to_rev_to_tail_ids, page_id_to_rev_to_timestamps

def old_get_page_id_to_revs(page_id: int,
                        head_id_2_timespans_2_tail_ids: Dict,
                        page_id_to_rev_to_tail_ids: Dict,
                        timestamps_revisions: List,
                        min_stability_span_in_hs: int,
                        page_id_to_rev_to_timestamps: Dict
                        # , page_id_to_main_page_id:Dict
                        ):
    # if page_id in head_id_2_timespans_2_tail_ids or \
    #         (page_id in page_id_to_main_page_id and
    #          page_id_to_main_page_id[page_id] in head_id_2_timespans_2_tail_ids):
    if page_id in head_id_2_timespans_2_tail_ids:
        timestamps = head_id_2_timespans_2_tail_ids[page_id].keys()
        # timespans = list(set([item for sublist in timespans for item in sublist]))
        timestamps = list(set(timestamps))
        # timestamp_to_revision_ids, revision_id_to_timestamps = \
        #     obtain_revision_ids_to_extract_stub(
        #         timestamps=timespans,
        #         revision_history=timestamps_revisions,
        #         min_stability_span_in_hs=min_stability_span_in_hs
        #     )
        timestamp_to_revision_ids, revision_id_to_timestamps = \
            obtain_revision_ids_to_extract(
                timestamps=timestamps,
                revision_history=timestamps_revisions,
                min_stability_span_in_hs=min_stability_span_in_hs
            )
        # timestamp_to_revision_ids, revision_id_to_timestamps = \
        #     obtain_revision_ids_to_extract_binary(
        #         timestamps=timespans,
        #         revision_history=timestamps_revisions,
        #         min_stability_span_in_hs=min_stability_span_in_hs
        #     )
        page_id_to_rev_to_timestamps[page_id] = revision_id_to_timestamps
        page_id_to_rev_to_tail_ids[page_id] = dict()
        for curr_revision_id2, curr_timestamps in revision_id_to_timestamps.items():
            page_id_to_rev_to_tail_ids[page_id][curr_revision_id2] = set()
            for curr_timestamp2 in curr_timestamps:
                page_id_to_rev_to_tail_ids[page_id][curr_revision_id2] \
                    .update(head_id_2_timespans_2_tail_ids[page_id]
                            [curr_timestamp2])

    return page_id_to_rev_to_tail_ids, page_id_to_rev_to_timestamps
