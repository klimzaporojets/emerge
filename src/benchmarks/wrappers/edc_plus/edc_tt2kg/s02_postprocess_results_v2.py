from argparse import ArgumentParser
import json
import os, os.path as osp

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_file",
                        type=str,
                        default='experiments/s02_postprocess_results_v2/20250513/config.json',
                        help='A config file with the files to postprocess.')

    file_name_to_file = dict()

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    for curr_snapshot in config['snapshots_to_parse']:
        print('===========================')
        for curr_snapshot_file in curr_snapshot['files']:
            print('-----------------')
            if not os.path.exists(curr_snapshot_file['input_file']):
                print(f'WARNING, {curr_snapshot_file["input_file"]} '
                      f'does not exist, continuing')
                continue

            with open(curr_snapshot_file['input_file'], 'r') as edc_output_file:
                edc_output = json.load(edc_output_file)

            print(f" - Writing {curr_snapshot_file['output_file']} \n"
                  f"from {curr_snapshot_file['passage_file']}...")

            os.makedirs(os.path.dirname(curr_snapshot_file['output_file']), exist_ok=True)
            with open(curr_snapshot_file['passage_file'], 'r') as passage_file, \
                    open(curr_snapshot_file['output_file'], 'wt', encoding='utf-8') as postprocessed_file:

                for i, line in enumerate(passage_file):
                    if i >= len(edc_output):
                        print(f'WARNING on {i} not all predicted in '
                              f'{curr_snapshot_file["input_file"]} '
                              f'when compared with '
                              f'{curr_snapshot_file["passage_file"]}')
                        break
                    passage_data = json.loads(line)
                    edc_data = edc_output[i]

                    assert i == edc_data["index"]

                    passage_data["predictions"] = list()
                    predicted_triples = []
                    predicted_triples_oie_deprecate = []
                    predicted_triples_oie_add = []
                    predicted_triples_oie_not_in_text = []
                    for triple in edc_data["schema_canonicalization"]:
                        # In some cases EDC fails to canonicalize a triple (e.g. a relation is not in the schema)
                        if triple is None:
                            continue
                        if len(triple) < 4:
                            continue
                        if str(triple[3]).lower() not in {'add', 'deprecate'}:
                            continue
                        triple_data = {
                            "action": triple[3],
                            "extracted_relation": triple[:3],
                            "triple_qids": ["--NME--", triple[1], "--NME--"],
                            "triple_labels": ["--NME--", triple[1], "--NME--"]
                        }
                        predicted_triples.append(triple_data)
                    predicted_triples_entities_to_kg = []
                    for triple in edc_data['schema_canonicalization_not_in_text']:
                        # In some cases EDC fails to canonicalize a triple (e.g. a relation is not in the schema)
                        if triple is None:
                            continue

                        if len(triple) < 3:
                            continue

                        triple_data = {"extracted_relation": triple,
                                       "triple_qids": ["--NME--", triple[1], "--NME--"],
                                       "triple_labels": ["--NME--", triple[1], "--NME--"]}
                        predicted_triples_entities_to_kg.append(triple_data)

                    # if len(predicted_triples) > 0:
                    # predicted_triples_oie =
                    for curr_prediction_oie in edc_data['oie']:
                        if len(curr_prediction_oie) < 4:
                            continue
                        if str(curr_prediction_oie[3]).lower() not in {'add', 'deprecate'}:
                            continue
                        if curr_prediction_oie[3].lower() == 'add':
                            predicted_triples_oie_add.append(curr_prediction_oie[:3])
                        if curr_prediction_oie[3].lower() == 'deprecate':
                            predicted_triples_oie_deprecate.append(curr_prediction_oie[:3])

                    for curr_prediction_oie_not_in_text in edc_data['oie_not_in_text']:
                        if len(curr_prediction_oie_not_in_text) < 3:
                            continue
                        # if curr_prediction_oie_not_in_text[3].lower() not in {'add', 'deprecate'}:
                        #     continue
                        predicted_triples_oie_not_in_text.append(curr_prediction_oie_not_in_text[:3])

                    if len(predicted_triples_oie_deprecate +
                           predicted_triples_oie_add +
                           predicted_triples_oie_not_in_text) > 0:
                        passage_data["predictions"].append({
                            "predicted_triples": predicted_triples,
                            "predicted_triples_oie":
                                {
                                    "oie_add": predicted_triples_oie_add,
                                    "oie_deprecate": predicted_triples_oie_deprecate,
                                    "oie_not_in_text": predicted_triples_oie_not_in_text
                                },
                            "predicted_triples_entities_to_kg": predicted_triples_entities_to_kg,
                            "model": "EDC",
                            "dataset_path": curr_snapshot_file['passage_file'],
                            "predictions_path": curr_snapshot_file['input_file'],
                        })

                    postprocessed_file.write(f"{json.dumps(passage_data, ensure_ascii=False)}\n")
                    postprocessed_file.flush()
    print("Done.")
