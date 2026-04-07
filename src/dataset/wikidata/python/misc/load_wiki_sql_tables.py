import gzip

import logging
import os
import pickle
import traceback
from typing import Dict

logger = logging.getLogger(__name__)

def load_wikidata_redirect(path_cache_redirect,
                           wikidata_page_id_to_qid: Dict[int, str],
                           path_redirects):
    nr_pages_processed = 0
    wikidata_qid_to_redirected_qid = dict()
    if os.path.exists(path_cache_redirect):
        logger.info('starting loading from pickle %s' % path_cache_redirect)
        wikidata_qid_to_redirected_qid = \
            pickle.load(open(path_cache_redirect, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_redirect)
    else:
        logger.info('starting processing the following file in '
                    'load_wiki_page_id_to_redirected_page_id: %s' % path_redirects)

        # just for logs, for now None
        file_out = None
        field_titles = set()
        nr_page_ids_found = 0
        nr_page_ids_not_found = 0
        with (gzip.open(path_redirects, 'r') as f):
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8')
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
                                field_titles.add(field_title_to)
                                # page_id_to = wikidata_page_id_to_qid[field_title_to]
                                if field_page_id_from not in wikidata_page_id_to_qid:
                                    # 2024.07.31 - for now commenting this log, added a log with % of nr_page_ids_not_found
                                    # logger.error(f'field_page_id_from {field_page_id_from} '
                                    #              f'not in wikidata_page_id_to_qid')
                                    nr_page_ids_not_found += 1
                                    continue
                                else:
                                    nr_page_ids_found += 1
                                qid_from = wikidata_page_id_to_qid[field_page_id_from]
                                if file_out is not None:
                                    file_out.write(str(field_page_id_from) + ' --- ' + str(field_title_to) + '\n')
                                    file_out.flush()
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    perc_not_found_page_ids = 0.0
                                    if (nr_page_ids_found + nr_page_ids_not_found) > 0:
                                        perc_not_found_page_ids = \
                                            (nr_page_ids_not_found / (nr_page_ids_found + nr_page_ids_not_found)) * 100
                                    logger.info(f'have processed to get page id to title redirections: '
                                                f'{nr_pages_processed}, % of not found page ids: % '
                                                f'{perc_not_found_page_ids:.5f}')
                                # assert field_page_id_from not in wikipedia_page_id_to_redirected_page_id
                                assert qid_from not in wikidata_qid_to_redirected_qid
                                assert field_title_to.startswith('Q')
                                # we are here debugging
                                # assert qid_from.startswith('Q')
                                # qid_from = int(qid_from[1:])
                                qid_to = int(field_title_to[1:])
                                wikidata_qid_to_redirected_qid[qid_from] = qid_to
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_redirected_page_id some sort of error when '
                                         'processing redirects with %s' % curr_insert_tuple)
                            logger.error(traceback.format_exc())
                        finally:
                            pass
        pickle.dump(wikidata_qid_to_redirected_qid,
                    open(path_cache_redirect, 'wb'))

    # print(f'nr of entities with redirect to: {len(field_titles)}')
    return wikidata_qid_to_redirected_qid

def load_wikidata_page_id_to_qid(path_cache_wikidata_page_id_qid,
                                 path_wikipedia_wikidata_map,
                                 max_nr_rows=-1):
    nr_pages_processed = 0
    wikidata_page_id_to_qid = dict()
    if os.path.exists(path_cache_wikidata_page_id_qid):
        logger.info('starting loading from pickle %s' % path_cache_wikidata_page_id_qid)
        wikidata_page_id_to_qid = pickle.load(open(path_cache_wikidata_page_id_qid, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikidata_page_id_qid)
    else:
        logger.info('starting processing the following file in load_wiki_page_id_to_wikidata_qid: %s' %
                    path_wikipedia_wikidata_map)
        with gzip.open(path_wikipedia_wikidata_map, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('ISO-8859-1')
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
                            field_page_id = curr_insert_tuple[:index_f].strip()
                            field_page_id = int(field_page_id)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            inter_p = ',\''
                            index_f = rest.index(inter_p)
                            field_namespace = rest[:index_f].strip()
                            field_namespace = int(field_namespace)
                            if field_namespace != 0:
                                continue

                            rest = rest[index_f + len(inter_p):]
                            index_f = rest.index('\',')
                            if rest.startswith("Q"):
                                field_qid = rest[:index_f].strip()
                                field_qid = int(field_qid[1:])
                            else:
                                logger.warning('weird continue with the rest of '
                                               f'{rest}')
                                continue
                            wikidata_page_id_to_qid[field_page_id] = field_qid

                            nr_pages_processed += 1
                            if nr_pages_processed % 1000000 == 0:
                                logger.info('nr of processed pages: %s' % nr_pages_processed)

                            if -1 < max_nr_rows < nr_pages_processed:
                                pickle.dump(wikidata_page_id_to_qid, open(path_cache_wikidata_page_id_qid, 'wb'))
                                return wikidata_page_id_to_qid
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_wikidata_qid some sort of error with %s ' %
                                         curr_insert_tuple)
                            logger.error(err)
                        finally:
                            pass
        pickle.dump(wikidata_page_id_to_qid, open(path_cache_wikidata_page_id_qid, 'wb'))

    return wikidata_page_id_to_qid


