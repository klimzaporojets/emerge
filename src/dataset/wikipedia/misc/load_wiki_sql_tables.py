import csv
import gzip
import logging
import os
import pickle
import re
import traceback

logger = logging.getLogger(__name__)


def load_wdata_qid_to_page_ids(path_cache_qids_to_page_ids,
                               qids_to_page_ids_path):
    nr_rows_processed = 0
    wdata_qid_to_page_ids = dict()
    if os.path.exists(path_cache_qids_to_page_ids):
        logger.info('starting loading from pickle %s' % path_cache_qids_to_page_ids)
        wdata_qid_to_page_ids = pickle.load(open(path_cache_qids_to_page_ids, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_qids_to_page_ids)
    else:
        logger.info('starting processing the following file in load_wdata_qid_to_page_ids: %s' %
                    qids_to_page_ids_path)
        with open(qids_to_page_ids_path, 'rt') as infile:
            csv_reader = csv.reader(infile, delimiter='\t')
            for curr_row in csv_reader:
                curr_qid = curr_row[0]
                curr_page_id = int(curr_row[2])
                if curr_qid not in wdata_qid_to_page_ids:
                    wdata_qid_to_page_ids[curr_qid] = list()
                wdata_qid_to_page_ids[curr_qid].append(curr_page_id)
        logger.info('finished loading the data in load_wdata_qid_to_page_ids')
        pickle.dump(wdata_qid_to_page_ids, open(path_cache_qids_to_page_ids, 'wb'))
        logger.info('saved pickle cache to %s' % path_cache_qids_to_page_ids)
    return wdata_qid_to_page_ids


def load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                      path_wikipedia_wikidata_map):
    nr_pages_processed = 0
    wikipedia_page_id_to_wikidata_qid = dict()
    if os.path.exists(path_cache_wikipedia_page_id_to_wikidata_qid):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_id_to_wikidata_qid)
        wikipedia_page_id_to_wikidata_qid = pickle.load(open(path_cache_wikipedia_page_id_to_wikidata_qid, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_wikidata_qid)
    else:
        logger.info('starting processing the following file in load_wiki_page_id_to_wikidata_qid: %s' %
                    path_wikipedia_wikidata_map)
        with gzip.open(path_wikipedia_wikidata_map, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8', errors='replace')
                insert_start_with = 'INSERT INTO `page_props` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ',\''
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            inter_p = '\',\''
                            index_f = rest.index(inter_p)
                            field_property = rest[:index_f].strip()
                            rest = rest[index_f + len(inter_p):]
                            index_f = rest.index('\',')
                            field_value = rest[:index_f].strip()
                            if field_property == 'wikibase_item':
                                assert field_page_id_from not in wikipedia_page_id_to_wikidata_qid
                                wikipedia_page_id_to_wikidata_qid[field_page_id_from] = field_value
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('nr of processed pages: %s' % nr_pages_processed)
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_wikidata_qid some sort of error with %s ' %
                                         curr_insert_tuple)
                            logger.error(err)
                        finally:
                            pass
        pickle.dump(wikipedia_page_id_to_wikidata_qid, open(path_cache_wikipedia_page_id_to_wikidata_qid, 'wb'))

    return wikipedia_page_id_to_wikidata_qid


def load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                            wikipedia_page_title_to_wikipedia_page_id,
                                            path_wikipedia_page_redirects):
    nr_pages_processed = 0
    wikipedia_page_id_to_redirected_page_id = dict()
    if os.path.exists(path_cache_wikipedia_page_id_to_redirected_page_id):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_id_to_redirected_page_id)
        wikipedia_page_id_to_redirected_page_id = \
            pickle.load(open(path_cache_wikipedia_page_id_to_redirected_page_id, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_redirected_page_id)
    else:
        logger.info('starting procesisng the following file in load_wiki_page_id_to_redirected_page_id: %s' %
                    path_wikipedia_page_redirects)

        # just for logs, for now None
        file_out = None
        # file_out = open(os.path.join(output_dir, 'output_page_id_to_redirected_page_id.log'), 'wt', encoding='utf-8')
        with gzip.open(path_wikipedia_page_redirects, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8', errors='replace')
                insert_start_with = 'INSERT INTO `redirect` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ','
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            # -----
                            inter_p = ',\''
                            index_f = rest.index(inter_p)
                            field_namespace = int(rest[:index_f].strip())
                            rest = rest[index_f + len(inter_p):]
                            # -----
                            inter_p = '\',\''
                            index_f = rest.index(inter_p)
                            field_title_to = rest[:index_f].strip()
                            if '\'' in field_title_to:
                                field_title_to = field_title_to.replace('\\\'', "'")

                            if field_namespace == 0:
                                page_id_to = wikipedia_page_title_to_wikipedia_page_id[field_title_to]
                                if file_out is not None:
                                    file_out.write(str(field_page_id_from) + ' --- ' + str(field_title_to) + '\n')
                                    file_out.flush()
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('have processed to get page id to title redirections: %s'
                                                % nr_pages_processed)
                                assert field_page_id_from not in wikipedia_page_id_to_redirected_page_id
                                wikipedia_page_id_to_redirected_page_id[field_page_id_from] = page_id_to
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_redirected_page_id some sort of error when '
                                         'processing redirects with %s' % curr_insert_tuple)
                            logger.error(traceback.format_exc())
                        finally:
                            pass
        pickle.dump(wikipedia_page_id_to_redirected_page_id,
                    open(path_cache_wikipedia_page_id_to_redirected_page_id, 'wb'))

    return wikipedia_page_id_to_redirected_page_id


def load_wiki_page_title_to_wiki_page_id_improved_chatgpt(
    path_cache_wikipedia_page_title_to_wikipedia_page_id,
    path_cache_wikipedia_page_id_to_wikipedia_page_title,
    path_wikipedia_page_info
):
    """
    Improved version by chatgpt, use on your own risk.

    :param path_cache_wikipedia_page_title_to_wikipedia_page_id:
    :param path_cache_wikipedia_page_id_to_wikipedia_page_title:
    :param path_wikipedia_page_info:
    :return:
    """
    import os
    import gzip
    import pickle
    import re
    import traceback

    nr_pages_processed = 0

    # matches a single SQL-quoted string, respecting escaped quotes
    title_regex = re.compile(r"'((?:\\'|[^'])*)'")

    wikipedia_page_title_to_wikipedia_page_id = {}
    wikipedia_page_id_to_wikipedia_page_title = {}

    if os.path.exists(path_cache_wikipedia_page_title_to_wikipedia_page_id):
        logger.info('starting loading from pickle %s',
                    path_cache_wikipedia_page_title_to_wikipedia_page_id)

        wikipedia_page_title_to_wikipedia_page_id = pickle.load(
            open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'rb')
        )
        wikipedia_page_id_to_wikipedia_page_title = pickle.load(
            open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'rb')
        )

        logger.info('loaded from pickle %s',
                    path_cache_wikipedia_page_title_to_wikipedia_page_id)
        logger.info('loaded from pickle %s',
                    path_cache_wikipedia_page_id_to_wikipedia_page_title)

        return (
            wikipedia_page_title_to_wikipedia_page_id,
            wikipedia_page_id_to_wikipedia_page_title
        )

    logger.info(
        'starting loading in load_wiki_page_title_to_wiki_page_id from %s',
        path_wikipedia_page_info
    )

    insert_prefix = 'INSERT INTO `page` VALUES'

    with gzip.open(path_wikipedia_page_info, 'rb') as f:
        for idx_line, raw_line in enumerate(f):
            try:
                line = raw_line.decode('utf-8')
            except UnicodeDecodeError:
                continue

            if not line.startswith(insert_prefix):
                continue

            str_inserts = line[len(insert_prefix):].strip()

            if not str_inserts:
                continue

            # remove leading '(' and trailing ');'
            if str_inserts[0] == '(':
                str_inserts = str_inserts[1:]
            if str_inserts.endswith(');'):
                str_inserts = str_inserts[:-2]

            insert_tuples = str_inserts.split('),(')

            for curr_insert_tuple in insert_tuples:
                try:
                    parts = curr_insert_tuple.split(',', 2)
                    if len(parts) < 3:
                        continue

                    page_id = int(parts[0].strip())
                    namespace = int(parts[1].strip())
                    rest = parts[2]

                    if namespace != 0:
                        continue

                    match = title_regex.search(rest)
                    if not match:
                        continue

                    title = match.group(1)
                    title = title.replace("\\'", "'")

                    if title in wikipedia_page_title_to_wikipedia_page_id:
                        logger.warning(
                            'Duplicate title encountered: %s (page_id=%s)',
                            title, page_id
                        )
                        continue

                    if page_id in wikipedia_page_id_to_wikipedia_page_title:
                        logger.warning(
                            'Duplicate page_id encountered: %s (title=%s)',
                            page_id, title
                        )
                        continue

                    wikipedia_page_title_to_wikipedia_page_id[title] = page_id
                    wikipedia_page_id_to_wikipedia_page_title[page_id] = title

                    nr_pages_processed += 1
                    if nr_pages_processed % 1_000_000 == 0:
                        logger.info(
                            'Processed %s pages for page id ↔ title mapping',
                            nr_pages_processed
                        )

                except Exception:
                    logger.error(
                        'Error parsing page.sql tuple: %s',
                        curr_insert_tuple
                    )
                    logger.error(traceback.format_exc())

    pickle.dump(
        wikipedia_page_title_to_wikipedia_page_id,
        open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'wb')
    )
    pickle.dump(
        wikipedia_page_id_to_wikipedia_page_title,
        open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'wb')
    )

    return (
        wikipedia_page_title_to_wikipedia_page_id,
        wikipedia_page_id_to_wikipedia_page_title
    )


def load_wiki_page_title_to_wiki_page_id(path_cache_wikipedia_page_title_to_wikipedia_page_id,
                                         path_cache_wikipedia_page_id_to_wikipedia_page_title,
                                         path_wikipedia_page_info):
    nr_pages_processed = 0
    regex = r"'.*?((?<!\\)')"

    wikipedia_page_title_to_wikipedia_page_id = dict()
    wikipedia_page_id_to_wikipedia_page_title = dict()
    if os.path.exists(path_cache_wikipedia_page_title_to_wikipedia_page_id):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_title_to_wikipedia_page_id)
        wikipedia_page_title_to_wikipedia_page_id = \
            pickle.load(open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'rb'))
        wikipedia_page_id_to_wikipedia_page_title = \
            pickle.load(open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_title_to_wikipedia_page_id)
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_wikipedia_page_title)
    else:
        logger.info('starting loading in load_wiki_page_title_to_wiki_page_id from %s' % path_wikipedia_page_info)

        # just for logs, for now None
        file_out = None
        with gzip.open(path_wikipedia_page_info, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8', errors='replace')
                insert_start_with = 'INSERT INTO `page` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ','
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            # -----
                            # inter_p = ',\''
                            inter_p = ','
                            index_f = rest.index(inter_p)
                            field_namespace = int(rest[:index_f].strip())
                            rest = rest[index_f + len(inter_p):]
                            # -----
                            # inter_p = '\',\''
                            inter_p = '\','
                            # found = re.findall(r"'.*?\(\(?<!\\\)'\)", rest)
                            matches = re.finditer(regex, rest, re.MULTILINE)
                            nxt = next(matches)
                            field_title_to = nxt.group()
                            field_title_to = field_title_to[1:-1]
                            # index_f = rest.index(inter_p)
                            # field_title_to = rest[:index_f].strip()

                            # field_title_to = found

                            if field_namespace == 0:
                                if file_out is not None:
                                    file_out.write(str(field_page_id_from) + ' --- ' + str(field_title_to) + '\n')
                                    file_out.flush()
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('have processed to get page id to title mapping: %s' %
                                                nr_pages_processed)

                                if '\'' in field_title_to:
                                    field_title_to = field_title_to.replace('\\\'', "'")

                                if field_title_to in wikipedia_page_title_to_wikipedia_page_id:
                                    logger.warning(
                                        f'field_title_to: {field_title_to} already in wikipedia_page_title_to_wikipedia_page_id'
                                        f'in the following line: {curr_insert_tuple}')

                                wikipedia_page_title_to_wikipedia_page_id[field_title_to] = field_page_id_from
                                assert field_page_id_from not in wikipedia_page_id_to_wikipedia_page_title
                                wikipedia_page_id_to_wikipedia_page_title[field_page_id_from] = field_title_to

                        except Exception as err:
                            logger.error('!!!load_wiki_page_title_to_wiki_page_id - some sort of error with %s' %
                                         curr_insert_tuple)
                            logger.error(traceback.format_exc())
                        finally:
                            pass
        if file_out is not None:
            file_out.flush()
            file_out.close()
        pickle.dump(wikipedia_page_title_to_wikipedia_page_id,
                    open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'wb'))
        pickle.dump(wikipedia_page_id_to_wikipedia_page_title,
                    open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'wb'))

    return wikipedia_page_title_to_wikipedia_page_id, wikipedia_page_id_to_wikipedia_page_title
