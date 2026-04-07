import os
import re
import subprocess

import openai
import requests
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging

logger = logging.getLogger(__name__)


def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx: right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l: r + 1]
        try:
            # kzaporoj BEGIN detecting _
            bracketed_str = bracketed_str.replace("\\_", "_")
            # kzaporoj END detecting _
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_raw_triplets_emerge(raw_triplets: str):
    # Look for enclosing brackets
    # print(f'parse_raw_triplets_emerge value of raw_triplets: {raw_triplets}')
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l: r + 1]
        try:
            # kzaporoj BEGIN detecting _
            bracketed_str = bracketed_str.replace("\\_", "_")
            # kzaporoj END detecting _
            parsed_triple = ast.literal_eval(bracketed_str)
            if ((len(parsed_triple) == 3) or
                (len(parsed_triple) == 4 and parsed_triple[3] in {'ADD', 'DEPRECATE'})
            ) and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    parsed_triple[1] = parsed_triple[1].replace('_', ' ').strip()
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                parsed_triple[1] = parsed_triple[1].replace('_', ' ').strip()
                if len(parsed_triple) >= 3:
                    collected_triples.append(parsed_triple)
                else:
                    logger.error(f'ERROR, did_not_add_following_parsed_triple: '
                                 f'{parsed_triple}')
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


# find the contents inside each [...] chunk (if your input has explicit brackets)
# list_pat = re.compile(r"\[([^\[\]]+?)\]")
## BEGIN CHANGE
list_pat = re.compile(r"\[(.*?)\]", re.DOTALL)

list_pat_v3 = re.compile(r"\[([^\[\]]*)\]")

## END CHANGE
# tolerant token matcher:
# matches either a single-quoted chunk (allowing doubled single-quotes inside)
# or any run of non-comma characters (fallback)
token_pat = re.compile(r"\s*('(?:[^']|'')*'|[^,]+)\s*(?:,|$)")


def clean_token(tok: str, idx_part: int) -> str:
    tok = tok.strip()

    # NEW: strip surrounding double quotes (and common “smart quotes”) if present
    # (do it early so later whitespace/_ normalization behaves)
    tok = tok.strip('"').strip("“”")

    # remove stray [] if present from messy matches
    tok = tok.strip("[]")

    # strip any leading/trailing single quotes (one or many)
    tok = re.sub(r"^'+|'+$", "", tok)

    # collapse double single-quotes into one ('' -> ')
    tok = tok.replace("''", "'")

    # normalize whitespace inside token
    tok = re.sub(r"\s+", " ", tok)

    # if it is a relation name
    if idx_part == 1:
        tok = tok.replace("_", " ")
    else:
        tok = tok.replace("_", " ")

    return tok

def parse_raw_triplets_emerge_v3(raw_triplets: str) -> List[List[str]]:
    collected_triples: List[List[str]] = []

    for m in list_pat_v3.finditer(raw_triplets):
        raw = m.group(1)
        parts = [t.group(1) for t in token_pat.finditer(raw)]

        # Backward-compatible fallback: if token_pat finds nothing (unquoted format),
        # split by comma.
        if not parts:
            parts = [p.strip() for p in raw.split(",")]

        cleaned: List[str] = [clean_token(p, idx_part=idx_part) for idx_part, p in enumerate(parts)]
        # logger.info(f'cleaned_is: {cleaned}')
        if len(cleaned) in {3, 4}:  # keep triples and quads
            if len(cleaned) == 4 and cleaned[3] not in {"ADD", "DEPRECATE"}:
                continue
            collected_triples.append(cleaned)
    # logger.info(f'used_parse_raw_triplets_emerge_v3_to_parse "{raw_triplets}", '
    #       f'collected_triples: {collected_triples} '
    #             f'matches: {len(list_pat_v3.findall(raw_triplets))}')
    return collected_triples

def parse_raw_triplets_emerge_v2(raw_triplets: str) -> List[List[str]]:
    collected_triples:List[List[str]] = list()
    for m in list_pat.finditer(raw_triplets):
        raw = m.group(1)
        parts = [t.group(1) for t in token_pat.finditer(raw)]
        cleaned:List[str] = [clean_token(p, idx_part=idx_part) for idx_part, p in enumerate(parts)]
        if len(cleaned) in {3, 4}:  # keep triples and quads
            if len(cleaned) == 4:
                if cleaned[3] not in {'ADD', 'DEPRECATE'}:
                    continue
            collected_triples.append(cleaned)

    return collected_triples


def remove_quotes(s: str) -> str:
    # Strip leading/trailing whitespace first, then remove surrounding quotes
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()
        relation = remove_quotes(relation).strip()
        # Remove any leading numbering like "1.", "23." etc.
        relation = re.sub(r'^\d+\.\s*', '', relation).strip()
        relation = remove_quotes(relation).strip()

        relation_description = description[index_of_colon + 1:].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict

def is_script_running(script_name):
    try:
        # Use pgrep to search for the script name
        result = subprocess.run(['pgrep', '-f', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logger.info(f"Script '{script_name}' is_running_with_pid(s): {result.stdout.decode().strip()}")
            return True
    except Exception as e:
        logger.info(f"Error occurred: {e}")
    return False

def is_model_openai(model_name):
    return "gpt" in model_name


# def query_tgi(prompt, idx):
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 50,
#             "temperature": 0.7
#         }
#     }
#
#     start = time.time()
#     response = requests.post(TGI_URL, json=payload)
#     latency = time.time() - start
#
#     if response.ok:
#         output = response.json()[0]['generated_text']
#         return (idx, latency, output[:50] + "...")
#     else:
#         return (idx, latency, f"ERROR: {response.status_code}")


def generate_completion_transformers(
        input: list,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_token=256,
        answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    openai.api_key = os.environ["OPENAI_KEY"]
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            time.sleep(5)
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
