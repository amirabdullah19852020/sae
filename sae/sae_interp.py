import json
import os
import re

from collections import defaultdict
from functools import lru_cache
from typing import Callable

import datasets
from datasets import Dataset, DatasetDict, load_dataset

import numpy as np
import torch

from circuitsvis.tokens import colored_tokens
from IPython.display import display
from nnsight import LanguageModel
from transformers import AutoTokenizer
from tqdm import tqdm

from TinySQL.training_data.fragments import field_names, table_names

from .sae import Sae

def get_all_table_and_field_names():
    all_table_infos = table_names.get_TableInfo()
    all_field_infos = field_names.get_FieldInfo()

    all_table_names = []
    all_field_names = []

    for info in all_table_infos:
        all_table_names.append(info.name)
        all_table_names.extend(info.synonyms)

    for key, info in all_field_infos.items():
        all_field_names.append(info.name)
        all_field_names.extend(info.synonyms)

    all_table_names = [name.strip().lower() for name in all_table_names]
    all_field_names = [name.strip().lower() for name in all_field_names]

    return all_table_names, all_field_names

all_table_names, all_field_names = get_all_table_and_field_names()

class SaeOutput:
    def __init__(self, sae_name: str, sae: Sae, text: str,
               tokens: list[str], raw_acts: list[list[float]],
               top_acts: list[list[float]], top_indices: list[list[int]],
                skip_positions: int = 2):

        self.raw_acts = raw_acts
        self.sae_name = sae_name
        self.sae = sae
        self.text = text
        self.tokens = tokens
        self.top_acts = top_acts
        self.top_indices = top_indices
        self.focused_tokens = tokens.copy()
        self.averaged_weights_by_sae_feature = self.averaged_representation()
        self.skip_positions = skip_positions

    def zip_nested_lists(self, list1, list2):
        # Check if both lists are actually lists and have the same length
        if isinstance(list1, list) and isinstance(list2, list) and len(list1) == len(list2):
            # Recursively zip each pair of sublists
            return [self.zip_nested_lists(sub1, sub2) for sub1, sub2 in zip(list1, list2)]
        else:
            # If they're not lists, return them as a tuple (base case)
            return (list1, list2)

    def averaged_representation(self):
        weights_by_index = {}
        zipped_indices_and_acts = list(zip(self.top_indices, self.top_acts))

        for one_set in zipped_indices_and_acts:
            indices, acts = one_set

            assert len(indices) == len(acts), "Indices and acts were not aligned!"

            for i, index in enumerate(indices):
                weights = acts[i]
                simplified_name = ".".join(self.sae_name.split(".")[2:])
                index_name = f"{index}_{simplified_name}"
                curr_weight = weights_by_index.get(index_name, [])
                curr_weight.append(weights)
                weights_by_index[index_name] = curr_weight

        averaged_weights_by_sae_feature = {}
        for index_name, weights in weights_by_index.items():
            averaged_weights_by_sae_feature[index_name] = np.average(weights)

        return averaged_weights_by_sae_feature

    def __hash__(self):
        return hash(self.text)

    def get_color_coded_tokens_circuitsvis(self, feature_num):
        """
        Visualizes tokens with color coding based on weights using circuitsvis.

        Args:
            tokens (list of str): List of tokens.
            weights (list of float): Corresponding weights.
        """
        tokens = self.tokens.copy()
        tokens = [simplify_token(token) for token in tokens]
        weights = self.get_weight_by_position(feature_num=feature_num)

        for position in range(self.skip_positions):
            weights[position] = 0

        # Create a TokenVisualization
        visualization = colored_tokens(tokens, weights, positive_color="green")
        # Display in Jupyter Notebook or export
        return visualization

    @lru_cache(maxsize=2000)
    def get_weight_by_position(self, feature_num: int):
        all_weights_by_position = []
        all_indices_and_acts = self.zip_nested_lists(self.top_indices, self.top_acts)

        for position, indices_and_acts in enumerate(all_indices_and_acts):
            indices_and_acts_dict = dict(indices_and_acts)
            curr_weight = indices_and_acts_dict.get(feature_num, 0)
            if position >= self.skip_positions and (self.tokens[position] != '<|pad|>'):
                all_weights_by_position.append(curr_weight)

        return all_weights_by_position

    def restrict_to_positions(self, positions):
        output = SaeOutput(
            sae_name=self.sae_name, sae=self.sae, text=self.text, tokens=self.tokens, raw_acts=self.raw_acts,
            top_acts=self.top_acts, top_indices=self.top_indices
        )
        output.raw_acts = [self.raw_acts[position] for position in positions]
        output.top_acts = [self.top_acts[position] for position in positions]
        output.top_indices = [self.top_indices[position] for position in positions]

        output.focused_tokens = [self.tokens[position] for position in positions]

        # print(f"Truncated indices, raw_acts and top_acts to {len(output.top_indices)}, {len(output.raw_acts)}, {len(output.top_acts)} from {len(self.top_indices)}, {len(self.raw_acts)}, {len(self.top_acts)}")
        return output


    def get_max_weight_of_feature(self, feature_num, skip_positions):
        weights_by_position = self.get_weight_by_position(feature_num=feature_num).copy()

        for i, token in enumerate(self.tokens):
            if token.strip() == '<|pad|>':
                weights_by_position[i] = 0
        return max(weights_by_position[skip_positions:])

    def zero_out_except_top_n(self, scores, indices, n):
        """
        Zero out all but the top n scores in the scores vector, preserving the order.
        """
        if len(scores) != len(indices):
            raise ValueError("Scores and indices lists must have the same length.")

        if n <= 0:
            return [0] * len(scores), indices

        # Pair scores with their original indices
        paired = list(zip(scores, range(len(scores))))

        # Sort by score in descending order
        paired.sort(key=lambda x: -x[0])

        # Get the indices of the top n scores
        top_n_indices = [index for _, index in paired[:n]]

        # Create a new scores list with only the top n scores retained
        filtered_scores = [scores[i] if i in top_n_indices else 0 for i in range(len(scores))]

        return filtered_scores, indices

    def zero_out_except_top_n_for_multiple(self, scores_list, indices_list, n):
        if len(scores_list) != len(indices_list):
            raise ValueError("Scores and indices lists must have the same length.")

        filtered_scores_list = []
        filtered_indices_list = []

        for scores, indices in zip(scores_list, indices_list):
            filtered_scores, filtered_indices = self.zero_out_except_top_n(scores, indices, n)
            filtered_scores_list.append(filtered_scores)
            filtered_indices_list.append(filtered_indices)

        return filtered_scores_list, filtered_indices_list

    def reconstruction_error(self, k=32):
        decoded_activations = self.decode_to_activations(k)
        raw_acts = torch.tensor(self.raw_acts).cuda().half()

        difference = decoded_activations - raw_acts

        reconstruction_error = torch.norm(difference).half() / torch.norm(raw_acts).half()
        return reconstruction_error.item()

    def decode_to_activations(self, k=32):
        filtered_acts, top_k_indices = self.zero_out_except_top_n_for_multiple(self.top_acts.copy(),
                                                                               self.top_indices.copy(), n=k)
        return self.sae.decode(top_acts=torch.tensor(filtered_acts).cuda().half(),
                               top_indices=torch.tensor(top_k_indices).cuda().type(torch.int64))

def simplify_token(token):
    return token.strip().lower().replace("ġ", "")

def backdoors_tagger(tokens, grouped_sae_output):
    grouped_sae_output.response_position = None
    tags_by_index = {i: [] for i in range(len(tokens))}

    for i, token in enumerate(tokens):
        simple_token = simplify_token(token)
        if simple_token == "prod":
            tags_by_index[i] = ["trigger", "prod"]
        if simple_token == "dev":
            tags_by_index[i] = ["trigger", "dev"]

        if simple_token == "response":
            tags_by_index[i] = ["response"]

        if i == len(tokens) - 1:
            tags_by_index[i] = ["last"]


def sql_tagger(tokens, grouped_sae_output):
    grouped_sae_output.context_position = None
    grouped_sae_output.response_position = None
    tags_by_index = {i: [] for i in range(len(tokens))}
    table_name = ""
    for i, token in enumerate(tokens):
        simple_token = simplify_token(token)
        if simple_token == "table":
            table_name = simplify_token(tokens[i + 1])

    for i, token in enumerate(tokens):
        simple_token = simplify_token(token)

        # Check for table or field.
        if table_name and (simple_token == table_name):
            tags_by_index[i].append(("TABLE", simple_token))
        elif simple_token in all_field_names and tokens[i - 1] == ",":
            tags_by_index[i].append(("FIELD", simple_token))
        else:
            pass

        if simple_token == "context":
            grouped_sae_output.context_position = i
            tags_by_index[i].append("CONTEXT_POSITION")

        if simple_token == "response":
            grouped_sae_output.response_position = i
            tags_by_index[i].append("RESPONSE_POSITION")

        if simple_token == "from" and grouped_sae_output.response_position and (grouped_sae_output.response_position < i):
            response_table_token = tokens[i+1]
            tags_by_index[i+1].append(("RESPONSE_TABLE", response_table_token))
            tags_by_index[i].append(("RESPONSE_FROM", "from"))

        if simple_token == "select" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_SELECT", simple_token))

        if simple_token == "group" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_GROUP", simple_token))

        if simple_token == "order" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_ORDER", response_table_token))

        if simple_token == "by" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_BY", simple_token))

        if simple_token == "count" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_COUNT", simple_token))

        if simple_token == "min" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_MIN", simple_token))
            tags_by_index[i].append(("RESPONSE_AGG", simple_token))

        if simple_token == "max" and grouped_sae_output.response_position and (
                grouped_sae_output.response_position < i):
            tags_by_index[i].append(("RESPONSE_MAX", simple_token))
            tags_by_index[i].append(("RESPONSE_AGG", simple_token))

    assert grouped_sae_output.context_position and grouped_sae_output.response_position, f"Did not find both context and response! Positions were context_position:{grouped_sae_output.context_position} and response_position:{grouped_sae_output.response_position}"

    for i, token in enumerate(tokens):
        tag_by_index = tags_by_index[i]
        simple_token = simplify_token(token)
        tags = [tag[0] for tag in tag_by_index]
        table_found = {"inst": False, "cont": False, "resp": False}

        if "TABLE" in tags:
            if (i < grouped_sae_output.context_position) and not table_found["inst"]:
                tag_by_index.append(("INSTRUCTION_TABLE", simple_token))
                table_found["inst"] = True
            elif (i >= grouped_sae_output.context_position) and (i < grouped_sae_output.response_position) and (not table_found["cont"]):
                tag_by_index.append(("CONTEXT_TABLE", simple_token))
                table_found["cont"] = True
            elif (i > grouped_sae_output.response_position) and (not table_found["resp"]):
                table_found["resp"] = True
            else:
                # print(f"Found second table token {simple_token}")
                pass

        if "FIELD" in tags:
            if (i < grouped_sae_output.context_position):
                tag_by_index.append(("INSTRUCTION_FIELD", simple_token))
            elif (i >= grouped_sae_output.context_position) and (i < grouped_sae_output.response_position):
                tag_by_index.append(("CONTEXT_FIELD", simple_token))
            elif i > grouped_sae_output.response_position:
                tag_by_index.append(("RESPONSE_FIELD", simple_token))
            else:
                pass

    grouped_sae_output.tags_by_index = tags_by_index


class GroupedSaeOutput:
    """
    Class that collects and analyzes SaeOutputs over several layers.
    """

    def __init__(self, sae_outputs_by_layer, text, tokens, tags_by_index=None, function_tagger=sql_tagger):
        tags_by_index = tags_by_index or {}

        self.sae_outputs_by_layer = sae_outputs_by_layer
        self.layers = list(self.sae_outputs_by_layer.keys())
        self.text = text
        self.tokens = tokens
        if tags_by_index:
            self.tags_by_index = tags_by_index
        else:
            self.apply_tags(function_tagger)

        self.averaged_weights_by_sae_feature = self.averaged_representation()

    def averaged_representation(self):
        all_dicts = [self.sae_outputs_by_layer[layer].averaged_weights_by_sae_feature for layer in self.layers]
        final_dict = {}

        for layer_dict in all_dicts:
            final_dict = final_dict | layer_dict

        return final_dict

    def get_all_tags(self):
        all_tags = set()
        for index, tags in self.tags_by_index.items():
            all_tags.update(tags)
        return all_tags

    def get_color_coded_tokens_circuitsvis(self, layer, feature_num):
        output = self.sae_outputs_by_layer[layer]
        return output.get_color_coded_tokens_circuitsvis(feature_num=feature_num)

    def get_max_weight_of_feature(self, layer: str, feature_num: int, skip_positions=2, response_only=False):

        if response_only:
            pass
        else:
            sae_output = self.sae_outputs_by_layer[layer]
            max_weight_of_feature = sae_output.get_max_weight_of_feature(feature_num, skip_positions=skip_positions)
            return max_weight_of_feature

    def apply_tags(self, function_tagger):
        function_tagger(tokens=self.tokens, grouped_sae_output=self)

    def sae_outputs_for_positions(self, positions):
        outputs_by_layer = {}
        for layer, outputs in self.sae_outputs_by_layer.items():
            sae_output = outputs
            return_output = sae_output.restrict_to_positions(positions)
            outputs_by_layer[layer] = return_output
        return outputs_by_layer

    def sae_activations_and_indices_for_tag_by_layer(self, tag):
        positions = self.search_indices_with_tag(tag)
        return self.sae_outputs_for_positions(positions)

    def search_indices_with_tag(self, tag):
        match_indices = []
        for index, tags_list in self.tags_by_index.items():
            if tag in tags_list:
                match_indices.append(index)
        return match_indices


class LoadedSAES:
    def __init__(self, full_model_name: str, model_alias: str,
                 tokenizer: AutoTokenizer, language_model: LanguageModel, layers: list[str],
                 layer_to_directory: dict, k: str, base_path: str, layer_to_saes: dict, store_activations: bool,
                 dataset: datasets.Dataset=None, dataset_mapper: Callable=None, max_seq_len=512, dataset_name=None):

        self.dataset = dataset
        self.full_model_name = full_model_name
        self.model_alias = model_alias
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.pad_token
        self.language_model = language_model
        self.layers = layers
        self.layer_to_directory = layer_to_directory
        self.k = k
        self.base_path = base_path
        self.layer_to_saes = layer_to_saes
        self.max_seq_len = max_seq_len
        self.store_activations = store_activations

        self.dataset_name = dataset_name
        if dataset_mapper and self.dataset:
            self.mapped_dataset = self.dataset.map(dataset_mapper)
        else:
            self.mapped_dataset = self.dataset

    @staticmethod
    def get_all_subdirectories(path):
        subdirectories = [
            os.path.join(path, name) for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name)) and not name.startswith(".")
        ]
        return subdirectories

    def nnsight_eval_string_for_layer(self, layer: str):
        """
        Converts transformer.h[0].mlp into self.language_model.transformer.h[0].mlp.output.save() for nnsight
        """
        subbed_layer = re.sub(r'\.([0-9]+)\.', r'[\1].', layer)
        return f"self.language_model.{subbed_layer}.output.save()"

    def encode_to_activations_for_layer(self, text: str, layer: str):
        with self.language_model.trace() as tracer:
            with tracer.invoke(text) as invoker:
                eval_string = self.nnsight_eval_string_for_layer(layer)
                my_output = eval(eval_string)
        if len(my_output) > 1:
            return my_output[0]
        else:
            return my_output

    def tokenize_to_max_len(self, text, pad_to_max_seq_len):
        if pad_to_max_seq_len < 0:
            return text
        else:
            tokenized = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_seq_len)
            text = self.tokenizer.decode(tokenized["input_ids"][0])
            return text

    def encode_to_sae_for_layer(self, text: str, layer: str, pad_to_max_seq_len: int = -1):
        text = self.tokenize_to_max_len(text=text, pad_to_max_seq_len=pad_to_max_seq_len)
        activations = self.encode_to_activations_for_layer(text, layer).cuda()
        raw_acts = activations[0].cpu().detach().numpy().tolist() if self.store_activations else None

        relevant_sae = self.layer_to_saes[layer]
        sae_acts_and_features = relevant_sae.encode(activations)

        tokens = self.tokenizer.tokenize(text)
        top_acts = sae_acts_and_features.top_acts[0].cpu().detach().numpy().tolist()
        top_indices = sae_acts_and_features.top_indices[0].cpu().detach().numpy().tolist()

        sae_output = SaeOutput(
            sae_name=layer, text=text, tokens=tokens, top_acts=top_acts, top_indices=top_indices, raw_acts=raw_acts,
            sae=relevant_sae
        )
        return sae_output

    def encode_to_all_saes(self, text: str, averaged_representation=False) -> GroupedSaeOutput:
        sae_outputs_by_layer = {layer: self.encode_to_sae_for_layer(text=text, layer=layer) for layer in self.layers}
        tokens = self.tokenizer.tokenize(text)
        result = GroupedSaeOutput(sae_outputs_by_layer=sae_outputs_by_layer, text=text, tokens=tokens)
        if averaged_representation:
            return result.averaged_representation()
        else:
            return result

    @staticmethod
    def load_from_path_for_backdoor(
            model_alias: str, k: str, cache_dir: str, dataset_mapper: Callable = None,
            dataset_name: str = None, store_activations=True):
        k = str(k)

        base_path = f"{cache_dir}/{model_alias}/k={k}"

        print(f"Loading from path {base_path}")
        subdirectories = LoadedSAES.get_all_subdirectories(base_path)

        layer_to_directory = {
            directory.split("/")[-1]: directory for directory in subdirectories
        }

        layer_to_directory = {layer: directory for layer, directory in layer_to_directory.items()}
        layers = sorted(list(layer_to_directory.keys()))

        if not model_alias.startswith("withmartian"):
            model_alias = f"withmartian/{model_alias}"

        language_model = LanguageModel(model_alias, device_map='cuda')
        tokenizer = language_model.tokenizer
        dataset = load_dataset(dataset_name) if dataset_name else None

        layer_to_saes = {layer: Sae.load_from_disk(directory).cuda() for layer, directory in layer_to_directory.items()}

        return LoadedSAES(dataset_name=dataset_name, full_model_name=model_alias,
                          model_alias=model_alias, layers=layers, layer_to_directory=layer_to_directory,
                          tokenizer=tokenizer, k=k, base_path=base_path, dataset=dataset, store_activations=store_activations,
                          layer_to_saes=layer_to_saes, language_model=language_model, dataset_mapper=dataset_mapper)


    @staticmethod
    def load_from_path(
            model_alias: str, k: str, cache_dir: str, store_activations=True,
            dataset_mapper: Callable=None, dataset=None):
        k = str(k)

        base_path = f"{cache_dir}/{model_alias}/k={k}"

        print(f"Loading from path {base_path}")
        subdirectories = LoadedSAES.get_all_subdirectories(base_path)

        layer_to_directory = {
            directory.split("/")[-1]: directory for directory in subdirectories
        }

        layer_to_directory = {layer: directory for layer, directory in layer_to_directory.items()}
        layers = sorted(list(layer_to_directory.keys()))

        with open(f"{base_path}/model_config.json", "r") as f_in:
            model_config = json.load(f_in)

            dataset_name = model_config["dataset_name"]
            full_model_name = model_config["model_name"]
            language_model = LanguageModel(full_model_name, device_map='cuda')
            tokenizer = language_model.tokenizer

        if dataset is None:
            dataset = load_dataset(dataset_name)

        layer_to_saes = {layer: Sae.load_from_disk(directory).cuda() for layer, directory in layer_to_directory.items()}

        return LoadedSAES(dataset_name=dataset_name, dataset=dataset, full_model_name=full_model_name,
                          model_alias=model_alias, layers=layers, layer_to_directory=layer_to_directory,
                          tokenizer=tokenizer, k=k, base_path=base_path, store_activations=store_activations,
                          layer_to_saes=layer_to_saes, language_model=language_model, dataset_mapper=dataset_mapper)

class SaeCollector:
    """
    This class is responsible for collecting a large amount of text,
    assigning tags to each token for a feature name, and also SAE outputs.
    These can then be used for probes and feature analysis.
    (Still to add: ablations.)
    """

    def __init__(self, loaded_saes, seed: int = 42, sample_size=10, restricted_tags=None, averaged_representations_only=True):
        self.loaded_saes = loaded_saes
        self.restricted_tags = restricted_tags or []
        self.sample_size = sample_size
        self.mapped_dataset = loaded_saes.mapped_dataset
        self.seed=seed
        self.mapped_dataset.shuffle(seed=seed)
        self.tokenizer = self.loaded_saes.tokenizer
        self.layers = self.loaded_saes.layers
        self.averaged_repesentations_only = averaged_representations_only
        if self.averaged_repesentations_only:
            print('Only using averaged representations')
            self.encoded_set = self.get_averaged_features_only(sample_size=self.sample_size)
        else:
            print('Using full representations.')
            self.encoded_set = self.create_and_load_random_subset(sample_size=self.sample_size)

    def get_texts(self):
        return [element["prompt"] for element in self.encoded_set]

    def get_maximally_activating_datasets(self, layer: str, feature_num: int, num_elements: int = 5):
        max_feature_weights = []
        for element in tqdm(self.encoded_set):
            max_feature_weights.append(element["encoding"].get_max_weight_of_feature(layer, feature_num))

        encoding_and_weights = zip(self.encoded_set, max_feature_weights)
        encoding_and_weights = sorted(encoding_and_weights, key=lambda x: x[1], reverse=True)
        encoding_and_weights = encoding_and_weights[:num_elements]

        encodings = list(zip(*encoding_and_weights))[0]

        html_list = [element["encoding"].get_color_coded_tokens_circuitsvis(layer, feature_num) for element in encodings]

        return encoding_and_weights, html_list

    def get_tags_stats(self):
        list_of_sets = []
        for element in self.encoded_set:
            list_of_sets.append(set((element["encoding"].tags_by_index.values())))

        # Total number of sets
        total_sets = len(list_of_sets)
        value_counts = defaultdict(int)

        for s in list_of_sets:
            for value in s:
                value_counts[value] += 1

        # Calculate percentages
        value_percentages = {value: (count / total_sets) * 100 for value, count in value_counts.items()}
        return value_percentages

    def get_all_sae_outputs_for_tag(self, tag):
        sae_outputs_for_tags= []
        for element in tqdm(self.encoded_set):
            sae_outputs_for_tags.append(element["encoding"].sae_activations_and_indices_for_tag_by_layer(tag))
        return sae_outputs_for_tags

    def get_prompt_and_encoding_for_feature(self, feature):
        prompt = feature["prompt"]
        encoding = self.loaded_saes.encode_to_all_saes(prompt)
        return_dict = {
            "prompt": prompt,
            "encoding": encoding,
            "averaged_representation": encoding.averaged_representation()
        }
        # Update other keys.
        for key in feature:
            return_dict[key] = feature[key]
        return return_dict

    def get_avg_reconstruction_error_for_all_k_and_layers(self):
        all_reconstruction_errors = {layer: self.get_avg_reconstruction_error_for_all_k(layer) for layer in self.layers}
        return all_reconstruction_errors

    def get_avg_reconstruction_error_for_all_k(self, layer, min_range=0, max_range=32):
        all_reconstruction_errors = {}
        for element in tqdm(self.encoded_set):
            encoding = element["encoding"]

            # Take k = min_range to min_range + 20, and then go at intervals of 10.
            part1 = list(range(min_range, min_range+21))
            part2 = list(range(min_range+20, max_range, 10))
            result = sorted(list(set(part1 + part2)))

            for k in result:
                recon_error = encoding.sae_outputs_by_layer[layer].reconstruction_error(k)
                curr_reconstruction_error_list = all_reconstruction_errors.get(k, [])
                curr_reconstruction_error_list.append(recon_error)
                all_reconstruction_errors[k] = curr_reconstruction_error_list

        average_reconstruction_errors = {k: np.average(error_list) for k, error_list in all_reconstruction_errors.items()}
        return average_reconstruction_errors


    def get_averaged_features_only(self, sample_size: int):
        sampled_set = self.mapped_dataset['train'] if isinstance(self.mapped_dataset, DatasetDict) else self.mapped_dataset
        final_dataset = []

        print("Now getting averaged representation.")

        if sample_size < 0 or sample_size is None:
            sampled_set = sampled_set
        else:
            sampled_set = sampled_set.select(range(sample_size))

        for element in tqdm(sampled_set):
            prompt = element["prompt"]
            averaged_representation = self.loaded_saes.encode_to_all_saes(prompt, averaged_representation=True)
            return_dict = {
                "prompt": prompt,
                "averaged_representation": averaged_representation,
                "label": element["label"]
            }
            final_dataset.append(return_dict)
        return final_dataset

    def create_and_load_random_subset(self, sample_size: int):
        if isinstance(self.mapped_dataset, DatasetDict):
            sampled_set = self.mapped_dataset['train']
        else:
            sampled_set = self.mapped_dataset

        if sample_size < 0 or sample_size is None:
            sampled_set = sampled_set
        else:
            sampled_set = sampled_set.select(range(sample_size))

        encoded_set = []
        for element in tqdm(sampled_set):
            encoded_element = self.get_prompt_and_encoding_for_feature(element)
            encoded_set.append(encoded_element)
        return encoded_set