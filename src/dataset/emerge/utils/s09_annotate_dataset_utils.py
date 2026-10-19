import json
import sys
import termios
from typing import List, Dict, Tuple

from sklearn.metrics import cohen_kappa_score
import pandas as pd
import logging
import os
from dataset.emerge.utils.constants import ACTION_CATEGORY_DEPRECATE, ACTION_CATEGORY_ASSERT

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

# import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import cohen_kappa_score
import krippendorff

from statsmodels.stats.inter_rater import fleiss_kappa


# def agreement_metrics(df, col1='anno', col2='anno2'):
def agreement_metrics(df, col1, col2):
    y1 = df[col1].astype(int).to_numpy()
    y2 = df[col2].astype(int).to_numpy()
    # n = len(y1)

    # Cohen's kappa
    ckappa = cohen_kappa_score(y1, y2)

    # Observed agreement
    Po = np.mean(y1 == y2)

    # PABAK (prevalence-adjusted bias-adjusted kappa)
    pabak = 2 * Po - 1

    # Scott's pi
    p_yes_1 = np.mean(y1)
    p_yes_2 = np.mean(y2)
    Pe_pi = p_yes_1 * p_yes_2 + (1 - p_yes_1) * (1 - p_yes_2)
    scotts_pi = (Po - Pe_pi) / (1 - Pe_pi) if Pe_pi != 1 else np.nan

    # Gwet's AC1
    pbar = (p_yes_1 + p_yes_2) / 2
    Pe_ac1 = 2 * pbar * (1 - pbar)
    gwet_ac1 = (Po - Pe_ac1) / (1 - Pe_ac1) if Pe_ac1 != 1 else np.nan

    # MCC
    mcc = matthews_corrcoef(y1, y2)

    # F1 (binary, positive=1)
    f1 = f1_score(y1, y2, zero_division=0)

    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y1, y2)

    # Accuracy
    acc = accuracy_score(y1, y2)

    to_ret = {
        "C Kappa": ckappa,
        "PABAK": pabak,
        "Scott's pi": scotts_pi,
        "Gwet's AC1": gwet_ac1,
        "MCC": mcc,
        "F1": f1,
        "Balanced Accuracy": bal_acc,
        "Observed Agreement": Po
        # 'Accuracy': acc
    }
    cleaned_to_ret = {k: float(round(v, 2)) for k, v in to_ret.items()}
    return cleaned_to_ret


def kappa_category(c_kappa):
    if c_kappa < 0:
        return 'Poor'
    elif 0 <= c_kappa < 0.20:
        return 'Weak'
    elif 0.20 <= c_kappa < 0.40:
        return 'Fair'
    elif 0.40 <= c_kappa < 0.60:
        return 'Moderate'
    elif 0.60 <= c_kappa < 0.80:
        return 'Strong'
    elif 0.80 <= c_kappa <= 1.00:
        return 'Almost perfect'
    else:
        return 'Invalid'


def get_cohen_kappa_agreement(df: pd.DataFrame,
                              annotator1_col_name: str,
                              annotator2_col_name: str) -> float:
    kappa = cohen_kappa_score(df[annotator1_col_name], df[annotator2_col_name])
    return kappa


def show_discrepancies_and_ask_correct(
        instance: Dict,
        config: Dict,
        df_stats: pd.DataFrame,
        idx_passage: int
        # annotated_instances: List[Dict]
) -> Tuple[Dict, int]:
    annotator_name = config['annotator_name']
    annotators_to_compare = set(config['annotators_to_compare'])
    llms_to_compare = config['llms_to_compare']
    tkgu_operations_to_check = set(config['tkgu_operations_to_check'])
    recalculate_statistics = False

    logger.info(f' =============ANNOTATION STATS=================\n'
                # f' {get_print_annotation_statistics(df_statistics=df_stats)}\n'
                f' {get_print_annotation_statistics_w_humans(df_statistics=df_stats)}\n'
                f' =============ANNOTATION STATS=================')

    for c_prompt, c_llms in llms_to_compare.items():
        llms_to_compare[c_prompt] = set(c_llms)

    for curr_tkgu_triple in instance['tkgu_triples']:
        if 'human_assessment' not in curr_tkgu_triple or \
                len(curr_tkgu_triple['human_assessment']) == 0:
            continue
        anno_assessments = list()
        for curr_human_assessment in curr_tkgu_triple['human_assessment']:
            if curr_human_assessment['annotator_name'] == annotator_name:
                anno_assessments.append(curr_human_assessment)
        for curr_anno_assessment in anno_assessments:
            anno_prompt_type = curr_anno_assessment['prompt_type']
            anno_assessment = curr_anno_assessment['assessment']
            is_there_discrepancy = False
            discrepancy_causes = list()
            if config['show_discrepancies_type'].lower().strip() in {'human', 'both'}:
                for curr_human_anno in curr_tkgu_triple['human_assessment']:
                    if curr_human_anno['prompt_type'] != anno_prompt_type:
                        continue
                    if curr_human_anno['annotator_name'] in annotators_to_compare \
                            and curr_human_anno['assessment'] != anno_assessment:
                        is_there_discrepancy = True
                        discrepancy_causes.append(f'DISCREPANCY with HUMAN {curr_human_anno}')
                    elif curr_human_anno['annotator_name'] in annotators_to_compare \
                            and curr_human_anno['assessment'] == anno_assessment:
                        discrepancy_causes.append(f'OK with HUMAN {curr_human_anno}')

            if config['show_discrepancies_type'].lower().strip() in {'llm', 'both'}:
                for curr_llm_anno in curr_tkgu_triple['llm_assessment']:
                    curr_llm_prompt_type = curr_llm_anno['llm_prompt_type']
                    curr_llm_name = curr_llm_anno['llm_name']
                    if curr_llm_prompt_type != anno_prompt_type:
                        continue
                    if curr_llm_name in llms_to_compare[curr_llm_prompt_type] \
                            and curr_llm_anno['llm_assessment'] != anno_assessment:
                        is_there_discrepancy = True
                        discrepancy_causes.append(f'DISCREPANCY with LLM {curr_llm_anno}')
                    elif curr_llm_name in llms_to_compare[curr_llm_prompt_type] \
                            and curr_llm_anno['llm_assessment'] == anno_assessment:
                        discrepancy_causes.append(f'OK with LLM {curr_llm_anno}')

            set_tkgu_ops = set(curr_tkgu_triple["tkgu_operations"])
            # if len(tkgu_operations_to_check & set_tkgu_ops)>0:
            #     pass

            if is_there_discrepancy and len(tkgu_operations_to_check & set_tkgu_ops) > 0:
                curr_disagreement_summary = \
                    '\n'.join(f'{i + 1}. {item}'
                              for i, item in enumerate(discrepancy_causes))

                clear_stdin()
                user_input = ''
                while user_input not in {'1', '2', 'y', 'n', 'c'}:
                    user_input = input(
                        f'============ {idx_passage + 1} CURRENT PASSAGE ({instance["revision_date"]})'
                        f' ===============================================\n'
                        f'{instance["passage"]} \n'
                        f'=========================================================\n'
                        f'--------------- TRIPLE: '
                        f' -- TKGU: {curr_tkgu_triple["tkgu_operations"]} -- '
                        f'prompt: {anno_prompt_type.upper()} \n'
                        # f'Does the above passage contains explicit or implicit knowledge to '
                        # f'support the triple ({curr_triple_labels})?: '
                        # f'support the triple:\n({triple_show})? WHERE: \n'
                        f'({curr_tkgu_triple["triple"]} -- {curr_tkgu_triple["triple_labels"]})?\n '
                        # f'PROMPT TYPE: {curr_action_category.upper()} ; \n'
                        f'-------------- ASSESSMENT MADE BY {annotator_name}: '
                        f'{curr_anno_assessment} \n'
                        f'-------------- DISAGREEMENT SUMMARY: \n'
                        f'{curr_disagreement_summary} \n'
                        # f'-------------- HUMAN ASSESSMENT: \n'
                        # f'{curr_human_assessment} \n'
                        f'----------- \n'
                        # f'qualifier_date ({field_qualifier_info}) \n'
                        # f'field_revision_date ({field_passage_date}) \n'
                        # f'field_triple_lifespan_date ({field_triple_lifespan_date}) \n'
                        # f'Definition of "{triple_show[1]}": "{triple_rel_definition}" \n'
                        f'enter c to leave it as it is, y to change it to YES, n '
                        f'to change it to NO, 1 to go to the next instance and 2 '
                        f'to ignore everything alltogether.')

                    user_input = user_input.lower().strip()
                if user_input == 'c':
                    continue
                elif user_input == 'y':
                    recalculate_statistics = True
                    curr_anno_assessment['assessment'] = True
                elif user_input == 'n':
                    recalculate_statistics = True
                    curr_anno_assessment['assessment'] = False
                elif user_input == '1':
                    return instance, int(user_input)
                elif user_input == '2':
                    return instance, int(user_input)
                print('DISCREPANCY, please correct')

    if not recalculate_statistics:
        return instance, 0
    else:
        return instance, 3


def merge_annotations(
        instance1: Dict,
        instance2: Dict
):
    assert len(instance1['tkgu_triples']) == len(instance2['tkgu_triples'])

    for triple_inst1, triple_inst2 in zip(
            instance1['tkgu_triples'],
            instance2['tkgu_triples']
    ):
        assert triple_inst1['triple'] == triple_inst2['triple']
        assert triple_inst1['tkgu_operations'] == triple_inst2['tkgu_operations']
        if 'human_assessment' in triple_inst2 and len(triple_inst2['human_assessment']) > 0:
            if 'human_assessment' not in triple_inst1 or \
                    len(triple_inst1['human_assessment']) == 0:
                triple_inst1['human_assessment'] = triple_inst2['human_assessment']
            else:
                hum_assessments1 = set()
                for curr_hum_assess in triple_inst1['human_assessment']:
                    hum_assessments1.add((curr_hum_assess['annotator_name'],
                                          curr_hum_assess['prompt_type']))
                for curr_hum_assess in triple_inst2['human_assessment']:
                    if (curr_hum_assess['annotator_name'],
                        curr_hum_assess['prompt_type']) not in hum_assessments1:
                        triple_inst1['human_assessment'].append(curr_hum_assess)
        if 'llm_assessment' in triple_inst2 and len(triple_inst2['llm_assessment']) > 0:
            if 'llm_assessment' not in triple_inst1 or \
                    len(triple_inst1['llm_assessment']) == 0:
                triple_inst1['llm_assessment'] = triple_inst2['llm_assessment']
            else:
                llm_assessments1 = set()
                for curr_llm_assess in triple_inst1['llm_assessment']:
                    llm_assessments1.add((curr_llm_assess['llm_name'],
                                          curr_llm_assess['llm_prompt_type']))
                for curr_llm_assess in triple_inst2['llm_assessment']:
                    if (curr_llm_assess['llm_name'],
                        curr_llm_assess['llm_prompt_type']) not in llm_assessments1:
                        triple_inst1['llm_assessment'].append(curr_llm_assess)
    return instance1


def show_discrepancies(parsed_line, config):
    llm_assessor_deprecation_triples = config['llm_assessor_deprecation_triples']
    llm_assessor_assert_triples = config['llm_assessor_assert_triples']

    for curr_triple in parsed_line['tkgu_triples']:
        if 'human_assessment' not in curr_triple:
            continue
        for curr_human_assessment in curr_triple['human_assessment']:
            for curr_llm_assessment in curr_triple['llm_assessment']:
                if curr_human_assessment['prompt_type'] == curr_llm_assessment['llm_prompt_type']:
                    if (curr_human_assessment['prompt_type'] == ACTION_CATEGORY_DEPRECATE and
                        curr_llm_assessment['llm_name'] != llm_assessor_deprecation_triples) or \
                            (curr_human_assessment['prompt_type'] == ACTION_CATEGORY_ASSERT and
                             curr_llm_assessment['llm_name'] != llm_assessor_assert_triples):
                        continue
                    if curr_human_assessment['assessment'] == curr_llm_assessment['llm_assessment']:
                        continue

                    clear_stdin()
                    user_input = ''
                    while user_input not in {'x', 'c'}:
                        user_input = input(
                            f'============CURRENT PASSAGE ({parsed_line["revision_date"]})'
                            f' ==============================\n'
                            f'{parsed_line["passage"]} \n'
                            f'=========================================================\n'
                            f'--------------- TRIPLE: '
                            f' -- TKGU: {curr_triple["tkgu_operations"]} -- '
                            f'prompt: {curr_human_assessment["prompt_type"].upper()} \n'
                            # f'Does the above passage contains explicit or implicit knowledge to '
                            # f'support the triple ({curr_triple_labels})?: '
                            # f'support the triple:\n({triple_show})? WHERE: \n'
                            f'({curr_triple["triple"]} -- {curr_triple["triple_labels"]})?\n '
                            # f'PROMPT TYPE: {curr_action_category.upper()} ; \n'
                            f'-------------- LLM ASSESSMENT: \n'
                            f'{curr_llm_assessment} \n'
                            f'-------------- HUMAN ASSESSMENT: \n'
                            f'{curr_human_assessment} \n'
                            f'----------- \n'
                            # f'qualifier_date ({field_qualifier_info}) \n'
                            # f'field_revision_date ({field_passage_date}) \n'
                            # f'field_triple_lifespan_date ({field_triple_lifespan_date}) \n'
                            # f'Definition of "{triple_show[1]}": "{triple_rel_definition}" \n'
                            f'enter X/x to exit and C/c to continue with next triple.')

                        user_input = user_input.lower().strip()
                    if user_input == 'x':
                        return False
    return True


def load_all_instances(
        lst_all_instances: List[Dict]
) -> pd.DataFrame:
    lst_instances = list()
    for curr_a_instance in lst_all_instances:
        for curr_a_triple in curr_a_instance['tkgu_triples']:
            # if 'human_assessment' not in curr_a_triple or \
            #         len(curr_a_triple['human_assessment']) == 0:
            #     continue
            if 'llm_assessment' not in curr_a_triple or \
                    len(curr_a_triple['llm_assessment']) == 0:
                logger.error('ERROR: no llm_assessment in '
                             f'{curr_a_instance["hash_id"]} '
                             f'for triple {curr_a_triple}')
                continue
            for curr_a_llm_assessment in curr_a_triple['llm_assessment']:
                for curr_a_tkgu_operation in curr_a_triple['tkgu_operations']:
                    if (
                            (curr_a_tkgu_operation == 'd-triples' and
                             curr_a_llm_assessment['llm_prompt_type'] == ACTION_CATEGORY_DEPRECATE)
                            or
                            (curr_a_tkgu_operation != 'd-triples' and
                             curr_a_llm_assessment['llm_prompt_type'] == ACTION_CATEGORY_ASSERT)
                    ):
                        to_add = {
                            # 'passage': curr_a_instance['passage'],
                            'llm_name': curr_a_llm_assessment['llm_name'],
                            'tkgu_operation': curr_a_tkgu_operation,
                            'prompt_type': curr_a_llm_assessment['llm_prompt_type'],
                            'llm_assessment': curr_a_llm_assessment['llm_assessment']
                            # 'llm_prompt': curr_a_llm_assessment['llm_prompt']
                        }
                        lst_instances.append(to_add)
    df_all_instances = pd.DataFrame(lst_instances)
    return df_all_instances


def stats_all_instances(df_all_instances: pd.DataFrame):
    # llm_annotators = df_all_instances['llm_name'].unique().tolist()
    # tkgu_operations = df_all_instances['tkgu_operation'].unique().tolist()
    grouped = (
        df_all_instances
        .groupby(['llm_name', 'tkgu_operation', 'prompt_type'])
        .agg(
            total_triples=('llm_assessment', 'count'),
            assessed_llm_true_triples=('llm_assessment', lambda x: (x == True).sum())
        )
        .reset_index()
    )
    print(f'stats_all_instances: \n {grouped} ')
    print(f'nr all triples: {grouped["total_triples"].sum()}')
    print(f'nr assessed llm true triples: {grouped["assessed_llm_true_triples"].sum()}')


def load_annotated_instances(
        annotated_instances: List[Dict]
) -> pd.DataFrame:
    lst_annotated_instances = list()
    nr_annos_per_human_per_tkgu = dict()
    for curr_a_instance in annotated_instances:
        for curr_a_triple in curr_a_instance['tkgu_triples']:
            if 'human_assessment' not in curr_a_triple or \
                    len(curr_a_triple['human_assessment']) == 0:
                continue
            if 'llm_assessment' not in curr_a_triple or \
                    len(curr_a_triple['llm_assessment']) == 0:
                logger.error('ERROR: no llm_assessment in '
                             f'{curr_a_instance["hash_id"]} '
                             f'for triple {curr_a_triple}')
                continue
            # if len(curr_a_triple['human_assessment']) != 2 and \
            #     len(curr_a_triple['human_assessment']) != 4:
            prompt_types_llms = set(ll['llm_prompt_type'] for ll in curr_a_triple['llm_assessment'])
            prompt_types_hum = set(ll['prompt_type'] for ll in curr_a_triple['human_assessment'])
            # if len(prompt_types_llms) != len(prompt_types_hum.intersection(prompt_types_llms)):
            for curr_a_human_assessment in curr_a_triple['human_assessment']:
                for curr_a_llm_assessment in curr_a_triple['llm_assessment']:
                    for curr_a_tkgu_operation in curr_a_triple['tkgu_operations']:
                        if curr_a_human_assessment['prompt_type'] == \
                                curr_a_llm_assessment['llm_prompt_type']:
                            if (
                                    (curr_a_tkgu_operation == 'd-triples' and
                                     curr_a_llm_assessment['llm_prompt_type'] == ACTION_CATEGORY_DEPRECATE)
                                    or
                                    (curr_a_tkgu_operation != 'd-triples' and
                                     curr_a_llm_assessment['llm_prompt_type'] == ACTION_CATEGORY_ASSERT)
                            ):
                                canno_name = curr_a_human_assessment['annotator_name']
                                ctkgu_name = curr_a_tkgu_operation
                                if (canno_name, ctkgu_name) not in nr_annos_per_human_per_tkgu:
                                    nr_annos_per_human_per_tkgu[(canno_name, ctkgu_name)] = 0
                                nr_annos_per_human_per_tkgu[(canno_name, ctkgu_name)] += 1
                                to_add = {
                                    'hash_id': curr_a_instance['hash_id'],
                                    'passage': curr_a_instance['passage'],
                                    'human_readable_triple': curr_a_human_assessment['human_readable_triple'],
                                    'definition_relation': curr_a_human_assessment['definition_relation'],
                                    'annotator_name': curr_a_human_assessment['annotator_name'],
                                    'llm_name': curr_a_llm_assessment['llm_name'],
                                    'tkgu_operation': curr_a_tkgu_operation,
                                    'prompt_type': curr_a_llm_assessment['llm_prompt_type'],
                                    'llm_assessment': curr_a_llm_assessment['llm_assessment'],
                                    'llm_prompt': curr_a_llm_assessment['llm_prompt'],
                                    'human_assessment': curr_a_human_assessment['assessment']
                                }
                                lst_annotated_instances.append(to_add)
    print(f'nr_annos_per_human_per_tkgu: {nr_annos_per_human_per_tkgu}')
    df_annotated_instances = pd.DataFrame(lst_annotated_instances)
    return df_annotated_instances


def get_print_annotation_statistics(
        df_statistics: pd.DataFrame
):
    if df_statistics is None:
        return ''

    human_annotators = df_statistics['annotator_name'].unique().tolist()
    llm_annotators = df_statistics['llm_name'].unique().tolist()
    tkgu_operations = df_statistics['tkgu_operation'].unique().tolist()
    to_ret = ''
    for curr_human in human_annotators:
        to_ret += f'-------------------------- \n'
        for curr_llm in llm_annotators:
            print_res = ''
            to_ret += f'----------- {curr_human} (human) vs ' \
                      f'{curr_llm} (LLM) --------\n'
            for curr_tkgu_operation in tkgu_operations:
                df_curr_stats = df_statistics[
                    (df_statistics['annotator_name'] == curr_human) &
                    (df_statistics['llm_name'] == curr_llm) &
                    (df_statistics['tkgu_operation'] == curr_tkgu_operation)
                    ]
                diff_h_values = df_curr_stats['human_assessment'].unique().tolist()
                diff_llm_values = df_curr_stats['llm_assessment'].unique().tolist()
                # Count True/False values in 'human_assessment'
                human_counts = df_curr_stats['human_assessment'].value_counts()

                # Count True/False values in 'llm_assessment'
                llm_counts = df_curr_stats['llm_assessment'].value_counts()

                human_true = human_counts.get(True, 0)
                human_false = human_counts.get(False, 0)

                llm_true = llm_counts.get(True, 0)
                llm_false = llm_counts.get(False, 0)

                if (not df_curr_stats.empty and
                        (len(diff_h_values) > 1 or len(diff_llm_values) > 1)):
                    c_kappa = get_cohen_kappa_agreement(
                        df=df_curr_stats,
                        annotator1_col_name='llm_assessment',
                        annotator2_col_name='human_assessment'
                    )
                    c_kappa_category = kappa_category(
                        c_kappa=c_kappa
                    )

                    other_stats = agreement_metrics(df=df_curr_stats,
                                                    col1='llm_assessment',
                                                    col2='human_assessment')
                    curr_print_res = (f'{curr_tkgu_operation}: {c_kappa:.2f} '
                                      f'({c_kappa_category}) -- '
                                      f'(tot triples: {human_true + human_false} - '
                                      f'hum_t: {human_true} - hum_f: {human_false} - '
                                      f'llm_t: {llm_true} - llm_f: {llm_false}) \n'
                                      f'** Other stats: {other_stats}')
                    curr_print_res = f'{curr_print_res}\n----\n'

                    print_res += curr_print_res
                else:
                    other_stats = agreement_metrics(df=df_curr_stats,
                                                    col1='llm_assessment',
                                                    col2='human_assessment')
                    curr_print_res = (f'{curr_tkgu_operation.upper()}: NaN '
                                      f'(NaN) -- '
                                      f'(tot triples: {human_true + human_false} - '
                                      f'hum_t: {human_true} - hum_f: {human_false} - '
                                      f'llm_t: {llm_true} - llm_f: {llm_false}) \n'
                                      f'** Other stats: {other_stats}')
                    curr_print_res = f'{curr_print_res}\n----\n'
                    print_res += curr_print_res
            print_res += '-----------------------------------------\n'
            to_ret += print_res
    return to_ret


def return_paper_stats(
        df_statistics: pd.DataFrame,
        human_names
):
    assert len(human_names) == 2
    # subset = df_statistics.groupby(['tkgu_operation', 'hash_id', 'human_readable_triple']).filter(
    #     lambda x: set(x['human_name']) == {'human1', 'human2'} and x['llm_assessment'].notna().all()
    # )
    subset = df_statistics.groupby(['tkgu_operation', 'hash_id', 'human_readable_triple']).filter(
        lambda x: set(x['human_name']) == set(human_names) and x['llm_assessment'].notna().all()
    )

    # 2) Pivot humans into columns
    human_wide = subset.pivot(index=['tkgu_operation', 'hash_id', 'human_readable_triple'],
                              columns='human_name',
                              values='human_assessment')

    # 3) Merge LLM assessments (drop duplicates since LLM may be repeated)
    llm_wide = subset.drop_duplicates(subset=['tkgu_operation', 'hash_id', 'human_readable_triple'])[
        ['tkgu_operation', 'hash_id', 'human_readable_triple', 'llm_assessment']
    ].set_index(['tkgu_operation', 'hash_id', 'human_readable_triple'])

    df_wide = human_wide.join(llm_wide)

    # 4) Ensure binary integer values
    df_wide = df_wide.astype(int)

    # 5) Compute agreement per TKGU operation
    results = []

    for op, group in df_wide.groupby('tkgu_operation'):
        print(f'group.shape of op {op}: {group.shape}')
        data = group.to_numpy()
        n_items = data.shape[0]

        # Fleiss' kappa
        rating_matrix = np.array([np.bincount(row, minlength=2) for row in data])
        fleiss = fleiss_kappa(rating_matrix)

        # Krippendorff's alpha
        alpha = krippendorff.alpha(reliability_data=data.T, level_of_measurement='nominal')

        # Pairwise Cohen's kappa
        h_h = cohen_kappa_score(group[human_names[0]], group[human_names[1]])
        h1_llm = cohen_kappa_score(group[human_names[0]], group['llm_assessment'])
        h2_llm = cohen_kappa_score(group[human_names[1]], group['llm_assessment'])

        results.append({
            'Operation': op,
            'H-H cohen': h_h,
            'H1-LLM cohen': h1_llm,
            'H2-LLM cohen': h2_llm,
            'H+LLM fleiss': fleiss,
            'H+LLM kripp': alpha
        })

    # 6) Overall agreement
    data_all = df_wide.to_numpy()
    rating_matrix_all = np.array([np.bincount(row, minlength=2) for row in data_all])
    fleiss_all = fleiss_kappa(rating_matrix_all)
    alpha_all = krippendorff.alpha(data_all.T, level_of_measurement='nominal')
    h_h_all = cohen_kappa_score(df_wide[human_names[0]], df_wide[human_names[1]])
    h1_llm_all = cohen_kappa_score(df_wide[human_names[0]], df_wide['llm_assessment'])
    h2_llm_all = cohen_kappa_score(df_wide[human_names[1]], df_wide['llm_assessment'])

    overall_row = {
        'Operation': 'Overall',
        'H-H cohen': h_h_all,
        'H1-LLM cohen': h1_llm_all,
        'H2-LLM cohen': h2_llm_all,
        'H+LLM fleiss': fleiss_all,
        'H+LLM kripp': alpha_all
    }

    agreement_table = pd.DataFrame(results + [overall_row])
    # print(f'agreement_table: {agreement_table}')

    return agreement_table


def get_print_annotation_statistics_w_humans(
        df_statistics: pd.DataFrame
):
    if df_statistics is None:
        return ''

    human_annotators = df_statistics['annotator_name'].unique().tolist()
    llm_annotators = df_statistics['llm_name'].unique().tolist()
    tkgu_operations = df_statistics['tkgu_operation'].unique().tolist()
    to_ret = ''
    for curr_human in human_annotators:
        to_ret += f'-------------------------- \n'
        for curr_llm in llm_annotators:
            print_res = ''
            to_ret += f'----------- {curr_human} (human) vs ' \
                      f'{curr_llm} (LLM) --------\n'
            for curr_tkgu_operation in tkgu_operations:
                df_curr_stats = df_statistics[
                    (df_statistics['annotator_name'] == curr_human) &
                    (df_statistics['llm_name'] == curr_llm) &
                    (df_statistics['tkgu_operation'] == curr_tkgu_operation)
                    ]
                diff_h_values = df_curr_stats['human_assessment'].unique().tolist()
                diff_llm_values = df_curr_stats['llm_assessment'].unique().tolist()
                # Count True/False values in 'human_assessment'
                human_counts = df_curr_stats['human_assessment'].value_counts()

                # Count True/False values in 'llm_assessment'
                llm_counts = df_curr_stats['llm_assessment'].value_counts()

                human_true = human_counts.get(True, 0)
                human_false = human_counts.get(False, 0)

                llm_true = llm_counts.get(True, 0)
                llm_false = llm_counts.get(False, 0)

                if (not df_curr_stats.empty and
                        (len(diff_h_values) > 1 or len(diff_llm_values) > 1)):
                    c_kappa = get_cohen_kappa_agreement(
                        df=df_curr_stats,
                        annotator1_col_name='llm_assessment',
                        annotator2_col_name='human_assessment'
                    )
                    c_kappa_category = kappa_category(
                        c_kappa=c_kappa
                    )

                    other_stats = agreement_metrics(df=df_curr_stats,
                                                    col1='llm_assessment',
                                                    col2='human_assessment')
                    curr_print_res = (f'{curr_tkgu_operation}: {c_kappa:.2f} '
                                      f'({c_kappa_category}) -- '
                                      f'(tot triples: {human_true + human_false} - '
                                      f'hum_t: {human_true} - hum_f: {human_false} - '
                                      f'llm_t: {llm_true} - llm_f: {llm_false}) \n'
                                      f'** Other stats: {other_stats}')
                    curr_print_res = f'{curr_print_res}\n----\n'

                    print_res += curr_print_res
                else:
                    other_stats = agreement_metrics(df=df_curr_stats,
                                                    col1='llm_assessment',
                                                    col2='human_assessment')
                    curr_print_res = (f'{curr_tkgu_operation.upper()}: NaN '
                                      f'(NaN) -- '
                                      f'(tot triples: {human_true + human_false} - '
                                      f'hum_t: {human_true} - hum_f: {human_false} - '
                                      f'llm_t: {llm_true} - llm_f: {llm_false}) \n'
                                      f'** Other stats: {other_stats}')
                    curr_print_res = f'{curr_print_res}\n----\n'
                    print_res += curr_print_res
            print_res += '-----------------------------------------\n'
            to_ret += print_res
    # humans against each other agreement
    for curr_human1 in human_annotators:
        for curr_human2 in human_annotators:
            print_res = ''

            if curr_human1 == curr_human2:
                continue
            # print_res = ''
            to_ret += f'----------- {curr_human1} (human) vs ' \
                      f'{curr_human2} (human) --------\n'
            for curr_tkgu_operation in tkgu_operations:
                df_annos_human1 = df_statistics[
                    (df_statistics['annotator_name'] == curr_human1) &
                    (df_statistics['tkgu_operation'] == curr_tkgu_operation)
                    ][['hash_id', 'annotator_name', 'tkgu_operation', 'human_assessment', 'human_readable_triple']] \
                    .drop_duplicates()
                df_annos_human2 = df_statistics[
                    (df_statistics['annotator_name'] == curr_human2) &
                    (df_statistics['tkgu_operation'] == curr_tkgu_operation)
                    ][['hash_id', 'annotator_name', 'tkgu_operation', 'human_assessment', 'human_readable_triple']] \
                    .drop_duplicates()

                df_both_humans = pd.merge(df_annos_human1,
                                          df_annos_human2,
                                          on=['hash_id', 'tkgu_operation', 'human_readable_triple'],
                                          how='inner',
                                          suffixes=('_1', '_2'))
                diff_h1_values = df_both_humans['human_assessment_1'].unique().tolist()
                diff_h2_values = df_both_humans['human_assessment_2'].unique().tolist()
                # Count True/False values in 'human_assessment'
                human1_counts = df_both_humans['human_assessment_1'].value_counts()
                human2_counts = df_both_humans['human_assessment_2'].value_counts()

                # Count True/False values in 'llm_assessment'
                human1_true = human1_counts.get(True, 0)
                human1_false = human1_counts.get(False, 0)

                human2_true = human2_counts.get(True, 0)
                human2_false = human2_counts.get(False, 0)
                other_stats = agreement_metrics(df=df_both_humans,
                                                col1='human_assessment_1',
                                                col2='human_assessment_2')
                curr_print_res = (f'{curr_tkgu_operation}: -- '
                                  f'(tot triples: {human1_true + human1_false} - '
                                  f'hum1_t: {human1_true} - hum1_f: {human1_false} - '
                                  f'hum2_t: {human2_true} - hum2_f: {human2_false}) \n'
                                  f'** Other stats: {other_stats}')
                curr_print_res = f'{curr_print_res}\n----\n'
                print_res += curr_print_res
            print_res += '-----------------------------------------\n'
            to_ret += print_res

    return to_ret


def get_annotation_statistics(
        annotated_instances: List[Dict],
        # ,
        # llm_names: List[str],
        annotator_names: List[str]
        # tkgu_operations: List[str]
):
    df_annotated_statistics = load_annotated_instances(annotated_instances=annotated_instances)
    # c_kappa = get_cohen_kappa_agreement(
    #     df=df_annotated_statistics,
    #     annotator1_col_name='llm_assessment',
    #     annotator2_col_name='human_assessment'
    # )
    # logger.info(f'general Cohen\'s Kappa so far: {c_kappa} '
    #             f'which is interpreted as {kappa_category(c_kappa=c_kappa)}')
    return df_annotated_statistics


def get_disagreements(
        annotated_instances: List[Dict],
        llm_names: List[str],
        annotator_names: List[str],
        tkgu_operations: List[str]
):
    pass


def obtain_property_ids_to_definitions(dictionary_path: str) -> Dict[str, str]:
    to_ret_dict: Dict[str, str] = dict()
    for curr_line_dict in open(dictionary_path, 'rt', encoding='utf-8'):
        curr_pars_line = json.loads(curr_line_dict)
        # property_label = curr_pars_line['text'].strip().lower()
        property_id = curr_pars_line['metadata']['property']
        # to_ret_dict[property_id] = property_label
        property_definition = curr_pars_line['metadata']['definition']
        to_ret_dict[property_id] = property_definition
    return to_ret_dict


def clear_stdin():
    """Flush the stdin buffer."""
    termios.tcflush(sys.stdin, termios.TCIFLUSH)


def human_assessment_exists(
        p_tkgu_triple,
        p_prompt_type,
        p_annotator_name
):
    for curr_h_assessment in p_tkgu_triple['human_assessment']:
        if curr_h_assessment['annotator_name'] == p_annotator_name and \
                curr_h_assessment['prompt_type'] == p_prompt_type:
            return True
    return False


def exceeds_max_per_tkgu_type(
        p_tkgu_triple,
        p_prompt_type,
        p_nr_annotated_per_tkgu_operation,
        p_max_annotations_per_tkgu_operation
):
    l_tkgu_operations = p_tkgu_triple['tkgu_operations']
    at_least_one_suitable = False
    for curr_tkgu_operation in l_tkgu_operations:
        if curr_tkgu_operation not in p_nr_annotated_per_tkgu_operation:
            p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] = 0
        if curr_tkgu_operation == 'd-triples' and p_prompt_type == ACTION_CATEGORY_DEPRECATE:
            if p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] < \
                    p_max_annotations_per_tkgu_operation[curr_tkgu_operation]:
                at_least_one_suitable = True
        elif curr_tkgu_operation != 'd-triples' and p_prompt_type == ACTION_CATEGORY_ASSERT:
            if p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] < \
                    p_max_annotations_per_tkgu_operation[curr_tkgu_operation]:
                at_least_one_suitable = True

    return not at_least_one_suitable


def update_count_annotated(
        p_instance: Dict,
        p_annotator_name: str,
        p_nr_annotated_per_tkgu_operation: Dict[str, int]
):
    for _curr_triple in p_instance['tkgu_triples']:
        _curr_triple_tkgu_ops = _curr_triple['tkgu_operations']
        if 'human_assessment' in _curr_triple:
            for _curr_human_assessment in _curr_triple['human_assessment']:
                if _curr_human_assessment['annotator_name'] == p_annotator_name:
                    _curr_prompt_type = _curr_human_assessment['prompt_type']
                    for _curr_tkgu_op in _curr_triple_tkgu_ops:
                        if _curr_tkgu_op not in p_nr_annotated_per_tkgu_operation:
                            p_nr_annotated_per_tkgu_operation[_curr_tkgu_op] = 0
                        if _curr_tkgu_op == 'd-triples' and _curr_prompt_type == \
                                ACTION_CATEGORY_DEPRECATE:
                            p_nr_annotated_per_tkgu_operation[_curr_tkgu_op] += 1
                        elif _curr_tkgu_op != 'd-triples' and _curr_prompt_type == \
                                ACTION_CATEGORY_ASSERT:
                            p_nr_annotated_per_tkgu_operation[_curr_tkgu_op] += 1

    return p_nr_annotated_per_tkgu_operation


def update_count_annotated_per_triple(
        p_tkgu_triple,
        p_prompt_type,
        p_nr_annotated_per_tkgu_operation
) -> Dict[str, int]:
    l_tkgu_operations = p_tkgu_triple['tkgu_operations']
    for curr_tkgu_operation in l_tkgu_operations:
        if curr_tkgu_operation not in p_nr_annotated_per_tkgu_operation:
            p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] = 0
        if curr_tkgu_operation == 'd-triples' and p_prompt_type == ACTION_CATEGORY_DEPRECATE:
            p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] += 1
        elif curr_tkgu_operation != 'd-triples' and p_prompt_type == ACTION_CATEGORY_ASSERT:
            p_nr_annotated_per_tkgu_operation[curr_tkgu_operation] += 1
    return p_nr_annotated_per_tkgu_operation


def get_llm_assessment(triple,
                       llm_assessor_name,
                       llm_prompt_type,
                       hash_id):
    """

    :param triple:
    :param llm_assessor_name:
    :param llm_prompt_type: 'triple_deprecation' or 'triple_assessment'
    :return:
    """
    assessment = [ct for ct in triple['llm_assessment'] if
                  ct['llm_name'] == llm_assessor_name and \
                  ct['llm_prompt_type'] == llm_prompt_type]
    if len(assessment) > 0:
        return assessment[0]['llm_assessment']
    else:
        print(f'ERROR for triple {triple} can not find '
              f'llm_assessor_name {llm_assessor_name} and '
              f'llm_prompt_type {llm_prompt_type} '
              f'hash_id {hash_id}')
        return False


def get_instance_tkgu_types_llm_assessments(instance: Dict, config: Dict):
    nr_assessed_true_d_triples = 0
    nr_assessed_false_d_triples = 0
    nr_assessed_true_e_triples = 0
    nr_assessed_false_e_triples = 0
    nr_assessed_true_x_triples = 0
    nr_assessed_false_x_triples = 0
    nr_assessed_true_ee_triples = 0
    nr_assessed_false_ee_triples = 0
    nr_assessed_true_ee_kg_triples = 0
    nr_assessed_false_ee_kg_triples = 0

    llm_assessor_deprecation_triples = config['llm_assessor_deprecation_triples']
    llm_assessor_assert_triples = config['llm_assessor_assert_triples']

    for curr_triple in instance['tkgu_triples']:
        curr_tkgu_operations = set(curr_triple['tkgu_operations'])
        if 'd-triples' in curr_tkgu_operations:
            assessment = get_llm_assessment(triple=curr_triple,
                                            llm_assessor_name=llm_assessor_deprecation_triples,
                                            llm_prompt_type=ACTION_CATEGORY_DEPRECATE,
                                            hash_id=instance['hash_id'])
            if assessment:
                nr_assessed_true_d_triples += 1
            else:
                nr_assessed_false_d_triples += 1
        if 'e-triples' in curr_tkgu_operations:
            assessment = get_llm_assessment(triple=curr_triple,
                                            llm_assessor_name=llm_assessor_assert_triples,
                                            llm_prompt_type=ACTION_CATEGORY_ASSERT,
                                            hash_id=instance['hash_id'])
            if assessment:
                nr_assessed_true_e_triples += 1
            else:
                nr_assessed_false_e_triples += 1
        if 'ee-triples' in curr_tkgu_operations:
            assessment = get_llm_assessment(triple=curr_triple,
                                            llm_assessor_name=llm_assessor_assert_triples,
                                            llm_prompt_type=ACTION_CATEGORY_ASSERT,
                                            hash_id=instance['hash_id'])
            if assessment:
                nr_assessed_true_ee_triples += 1
            else:
                nr_assessed_false_ee_triples += 1
        if 'ee-kg-triples' in curr_tkgu_operations:
            assessment = get_llm_assessment(triple=curr_triple,
                                            llm_assessor_name=llm_assessor_assert_triples,
                                            llm_prompt_type=ACTION_CATEGORY_ASSERT,
                                            hash_id=instance['hash_id'])
            if assessment:
                nr_assessed_true_ee_kg_triples += 1
            else:
                nr_assessed_false_ee_kg_triples += 1
        if 'x-triples' in curr_tkgu_operations:
            assessment = get_llm_assessment(triple=curr_triple,
                                            llm_assessor_name=llm_assessor_assert_triples,
                                            llm_prompt_type=ACTION_CATEGORY_ASSERT,
                                            hash_id=instance['hash_id'])
            if assessment:
                nr_assessed_true_x_triples += 1
            else:
                nr_assessed_false_x_triples += 1

    to_ret = {
        'nr_assessed_false_x_triples': nr_assessed_false_x_triples,
        'nr_assessed_true_x_triples': nr_assessed_true_x_triples,
        'nr_assessed_false_e_triples': nr_assessed_false_e_triples,
        'nr_assessed_true_e_triples': nr_assessed_true_e_triples,
        'nr_assessed_false_ee_triples': nr_assessed_false_ee_triples,
        'nr_assessed_true_ee_triples': nr_assessed_true_ee_triples,
        'nr_assessed_false_ee_kg_triples': nr_assessed_false_ee_kg_triples,
        'nr_assessed_true_ee_kg_triples': nr_assessed_true_ee_kg_triples,
        'nr_assessed_false_d_triples': nr_assessed_false_d_triples,
        'nr_assessed_true_d_triples': nr_assessed_true_d_triples
    }
    return to_ret


def get_nr_triples_per_tkgu(instance, nr_triples_per_tkgu_type):
    for curr_triple in instance['tkgu_triples']:
        curr_tkgu_operations = curr_triple['tkgu_operations']
        for curr_tkgu_op in curr_tkgu_operations:
            nr_triples_per_tkgu_type[curr_tkgu_op] += 1
    return nr_triples_per_tkgu_type


def is_lowest_dominant_present(all_nr_triples_per_tkgu_type,
                               curr_nr_triples_per_tkgu_type):
    less_dominant_tkgu_type = min(all_nr_triples_per_tkgu_type,
                                  key=all_nr_triples_per_tkgu_type.get)
    if curr_nr_triples_per_tkgu_type[less_dominant_tkgu_type] > 0:
        return True

    return False


def update_nr_subsampled_triples(all_nr_triples_per_tkgu_type,
                                 curr_nr_triples_per_tkgu_type):
    for k, v in curr_nr_triples_per_tkgu_type.items():
        all_nr_triples_per_tkgu_type[k] += v
    return all_nr_triples_per_tkgu_type
