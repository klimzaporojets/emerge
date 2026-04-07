import logging
import re

import mwparserfromhell

logger = logging.getLogger(__name__)


def clean_entity_links_from_text(wikimedia_text, compiled_mention_finder, source_title, filtered_date,
                                 anchor_wikidata_qid, compiled_country_in_link):
    to_ret = ''
    last_span_added_pos = 0
    for curr_found_mention in compiled_mention_finder.finditer(wikimedia_text):
        span_start = curr_found_mention.start()
        span_end = curr_found_mention.end()
        mention_text = curr_found_mention.group()
        mention_data = get_mention(wikimedia_text, mention_text, source_title,
                                   # filtered_date, anchor_wikidata_qid,
                                   compiled_country_in_link)
        mention_text = mention_data['mention']
        to_ret += wikimedia_text[last_span_added_pos:span_start]
        to_ret += ' ' + mention_text
        last_span_added_pos = span_end

    to_ret += wikimedia_text[last_span_added_pos:]
    return to_ret


def get_mention(source_text, mention_text, source_title,
                compiled_country_in_link):
    curr_found_mention_pr = mention_text[2:]
    if '|' in curr_found_mention_pr:
        index_pipe = curr_found_mention_pr.index('|')
        link = curr_found_mention_pr[:index_pipe]
        mention = curr_found_mention_pr[index_pipe + 1:]
        if mention[:mention.index(']]')] == '':
            # pipe trick
            found_parenthesis = re.search(' \(.*\)$', link)
            # first look for parenthesis like "Yours, Mine and Ours (1968 film)"
            if found_parenthesis is not None:
                mention = link[:found_parenthesis.start()].strip() + mention
            else:
                # if there is no parenthesis like in previous case, then look for the first comma like
                #  "Il Buono, il Brutto, il Cattivo"
                found_comma = re.search(', ', link)
                if found_comma is not None:
                    mention = link[:found_comma.start()].strip() + mention

            # if the mention is still empty, something wrong:
            if mention[:mention.index(']]')] == '':
                return {'mention': mention, 'link': link, 'has_to_ignore': True, 'has_to_count': True}
    else:
        right_index_link = mention_text.index(']]')
        link = mention_text[2:right_index_link]
        mention = mention_text[2:]

    link = link.strip()
    mention = mention.strip()

    if len(link) > 0:
        if compiled_country_in_link.search(link):
            return {'mention': '', 'link': '', 'has_to_ignore': True, 'has_to_count': False}

    if len(link) == 0:
        return {'mention': mention, 'link': link, 'has_to_ignore': True, 'has_to_count': True}
    elif len(link) == 1:
        link = link[0].upper()
    else:
        link = link[0].upper() + link[1:]

    mention = mention.replace(']]', '')

    link = link.strip()
    mention = mention.strip()

    if link.startswith('Image:') or link.startswith('File:') or link.startswith('Category:') or \
            link.startswith('Wiktionary:') or \
            link.startswith(':Image:') or link.startswith(':File:') or link.startswith(':Category:') or \
            link.startswith(':Wiktionary:'):
        return {'mention': '', 'link': '', 'has_to_ignore': True, 'has_to_count': False}

    if '#' in link:
        link = link.split('#')[0]
    # begin DELETE THE QUOTATION
    mention = re.sub(r'(?i)\'+(.*?)\'+', lambda m: m.group(1), mention)
    # end DELETE THE QUOTATION

    return {'mention': mention, 'link': link, 'has_to_ignore': False, 'has_to_count': True}


def parse_mentions_from_source(source, source_title,
                               compiled_mention_finder,
                               compiled_country_in_link, get_span_pos):
    tot_detected_mentions = 0
    tot_link_errors = 0
    to_ret_mention_links = list()
    for curr_found_mention in compiled_mention_finder.finditer(source):
        found_mention_group = curr_found_mention.group()
        tot_detected_mentions += 1

        returned = get_mention(source, found_mention_group, source_title,
                               compiled_country_in_link)

        mention = returned['mention']
        link = returned['link']
        has_to_ignore = returned['has_to_ignore']
        has_to_count = returned['has_to_count']
        if not has_to_count:
            tot_detected_mentions -= 1
        if has_to_ignore and has_to_count:
            tot_link_errors += 1
        if has_to_ignore:
            continue

        if len(link) == 0:
            logger.debug('something wrong, EMPTY LINK!!!!: "%s" all this inside "%s" "%s" qid: "%s"' %
                         (curr_found_mention, source_title, source, None))
            tot_link_errors += 1
            continue
        link = link.replace(' ', '_')
        if get_span_pos:
            to_ret_mention_links.append({
                'anchor_mention_text': mention,
                'target_wikipedia_title_orig': link,
                'span': curr_found_mention.span()
            })
        else:
            to_ret_mention_links.append({
                'anchor_mention_text': mention,
                'target_wikipedia_title_orig': link
            })

    return to_ret_mention_links, tot_detected_mentions, tot_link_errors


def parse_mentions_from_source_ret_text(source, source_title,
                                        compiled_mention_finder,
                                        compiled_country_in_link):
    tot_detected_mentions = 0
    tot_link_errors = 0
    cleaned_source = ''
    to_ret_mention_links = list()

    last_char_pos = 0
    for curr_found_mention in compiled_mention_finder.finditer(source):
        # strips the [[ and ]]
        found_mention_group = curr_found_mention.group()
        tot_detected_mentions += 1

        returned = get_mention(source, found_mention_group, source_title,
                               compiled_country_in_link)

        mention = returned['mention']
        link = returned['link']
        has_to_ignore = returned['has_to_ignore']
        has_to_count = returned['has_to_count']
        if not has_to_count:
            tot_detected_mentions -= 1
        if has_to_ignore and has_to_count:
            tot_link_errors += 1
        if has_to_ignore:
            continue
        link = link.replace(' ', '_')
        source_in_between = source[last_char_pos:curr_found_mention.span()[0]]
        start_char = len(f'{cleaned_source}{source_in_between}')
        assert found_mention_group.startswith('[[')
        if len(link) > 0:
            to_ret_mention_links.append({
                'mention_text': mention,
                'target_entity': link,
                'start_char': start_char,  # +2 because of the two [[, see also assert above
                'end_char': start_char + len(mention)  # +2 because of the two [[, see also assert above
            })
        last_char_pos = curr_found_mention.span()[1]
        cleaned_source = f'{cleaned_source}{source_in_between}{mention}'
        #
    source_in_between = source[last_char_pos:]
    cleaned_source = f'{cleaned_source}{source_in_between}'

    return cleaned_source, to_ret_mention_links, tot_detected_mentions, tot_link_errors


def get_mentions_and_links(source, source_length, source_title,
                           compiled_mention_finder, compiled_country_in_link,
                           get_span_pos=False):
    mention_links, tot_detected_mentions, tot_links_errors = (
        parse_mentions_from_source(source, source_title,
                                   compiled_mention_finder,
                                   compiled_country_in_link,
                                   get_span_pos=get_span_pos))
    anchor_content_length = source_length
    for curr_mention_link in mention_links:
        curr_mention_link['target_wikipedia_page_id'] = None
        curr_mention_link['target_wikipedia_title'] = None
        curr_mention_link['target_qid'] = None
        curr_mention_link['anchor_content_length'] = anchor_content_length
        curr_mention_link['anchor_wikipedia_title'] = source_title

    return mention_links, tot_detected_mentions, tot_links_errors


def extract_mentions_with_positions(wikitext):
    # Parse the wikitext
    parsed = mwparserfromhell.parse(wikitext)

    # Initialize variables
    plain_text = ""
    mentions = []
    current_position = 0

    # Iterate through the nodes
    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.text.Text):
            # Append plain text and update position
            plain_text += node.value
            current_position += len(node.value)
        elif isinstance(node, mwparserfromhell.nodes.wikilink.Wikilink):
            # Add the link mention, its position, and the entity it links to
            if node.text is None:
                mention_text = str(node.title)
            else:
                mention_text = str(node.text)

            link = str(node.title)
            if len(link) == 1:
                link = link[0].upper()
            elif len(link) > 1:
                link = link[0].upper() + link[1:]
            link_target = link.replace(' ', '_')  # The entity the link points to

            mentions.append(
                {
                    'mention_text': mention_text,
                    'start_char': current_position,
                    'end_char': current_position + len(mention_text),
                    'target_entity': link_target
                }
            )
            plain_text += mention_text  # Add the link text to the plain text
            current_position += len(mention_text)
        elif isinstance(node, mwparserfromhell.nodes.template.Template):
            # Add the template mention and its position (templates don't have links)
            mention_text = str(node)
            plain_text += mention_text  # Add the template text to the plain text
            current_position += len(mention_text)
        else:
            # For other node types, convert to string and append
            node_text = str(node)
            plain_text += node_text
            current_position += len(node_text)

    # Output the plain text and mentions with positions and links
    return plain_text, mentions


def extract_mentions_with_positions_v2(source, source_title,
                                       # anchor_wikidata_qid, filtered_date,
                                       compiled_mention_finder, compiled_country_in_link):
    """
    The _v2 is the version without usiong mwparserfromhell, which misses many mentions
    :param source:
    :param source_length:
    :param source_title:
    :param compiled_mention_finder:
    :param compiled_country_in_link:
    :param get_span_pos:
    :return:
    """

    cleaned_source, mention_links, tot_detected_mentions, tot_links_errors = (
        parse_mentions_from_source_ret_text(source, source_title,
                                            compiled_mention_finder,
                                            compiled_country_in_link))

    return cleaned_source, mention_links


def obtain_context(text, mention_link, nr_tokens_around_mention,
                   obtain_paragraphs: bool = False):
    logger.debug(f'the following is the span: {mention_link["span"]}')
    text_to_left: str
    text_to_right: str
    text_to_left = text[: mention_link['span'][0]]  # .strip()
    text_to_right = text[mention_link['span'][1]:]  # .strip()
    trimmed_left = ''

    to_ret_text = ''
    if not obtain_paragraphs:
        if len(text_to_left) > 0:
            splitted_left = text_to_left.split(' ')
            trimmed_left = splitted_left[-nr_tokens_around_mention:]
            trimmed_left = ' '.join(trimmed_left)

        trimmed_right = ''
        if len(text_to_right) > 0:
            splitted_right = text_to_right.split(' ')
            trimmed_right = splitted_right[:nr_tokens_around_mention]
            trimmed_right = ' '.join(trimmed_right)

        to_ret_text = trimmed_left + ' ' + mention_link['anchor_mention_text'] + ' ' + trimmed_right
    else:
        # obtains complete paragraphs
        index_start_paragraph = text_to_left.rfind('\n')
        index_end_paragraph = text_to_right.find('\n')
        if index_start_paragraph == -1:
            index_start_paragraph = 0
        if index_end_paragraph == -1:
            index_end_paragraph = len(text_to_right)
        paragraph_to_return = text[index_start_paragraph:
                                   index_end_paragraph + mention_link['span'][1]]
        to_ret_text = paragraph_to_return
    return to_ret_text
