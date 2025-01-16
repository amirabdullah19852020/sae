import json
import os
import re

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
from datasets import load_dataset
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

@dataclass
class SaeOutput:
    sae_name: str
    sae: Sae
    text: str
    tokens: list[str]
    raw_acts: list[list[float]]
    top_acts: list[list[float]]
    top_indices: list[list[int]]

    def restrict_to_positions(self, positions):
        output = deepcopy(self)
        output.raw_acts = [self.raw_acts[position] for position in positions]
        output.top_acts = [self.top_acts[position] for position in positions]
        output.top_indices = [self.top_indices[position] for position in positions]

        # print(f"Truncated indices, raw_acts and top_acts to {len(output.top_indices)}, {len(output.raw_acts)}, {len(output.top_acts)} from {len(self.top_indices)}, {len(self.raw_acts)}, {len(self.top_acts)}")
        return output

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

    def reconstruction_error(self, k=128):
        decoded_activations = self.decode_to_activations(k)
        raw_acts = torch.tensor(self.raw_acts).cuda()

        difference = decoded_activations - raw_acts

        reconstruction_error = torch.norm(difference) / torch.norm(raw_acts)
        return reconstruction_error.item()

    def decode_to_activations(self, k=128):
        filtered_acts, top_k_indices = self.zero_out_except_top_n_for_multiple(self.top_acts.copy(),
                                                                               self.top_indices.copy(), n=k)
        return self.sae.decode(top_acts=torch.tensor(filtered_acts).cuda(),
                               top_indices=torch.tensor(top_k_indices).cuda())


def simplify_token(token):
    return token.strip().lower().replace("ġ", "")


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
        if table_name and (simple_token == table_name):
            tags_by_index[i].append(("TABLE", simple_token))
        elif simple_token in all_field_names and tokens[i - 1] == ",":
            tags_by_index[i].append(("FIELD", simple_token))
        else:
            tags_by_index[i].append(("NONE", simple_token))

        if simple_token == "context":
            grouped_sae_output.context_position = i
        if simple_token == "response":
            grouped_sae_output.response_position = i

    for i, token in enumerate(tokens):
        tag_by_index = tags_by_index[i]
        simple_token = simplify_token(token)
        tags = [tag[0] for tag in tag_by_index]
        if "TABLE" in tags:
            if i < grouped_sae_output.context_position:
                tag_by_index.append(("INSTRUCTION_TABLE", simple_token))
                print(f"Found instruction table {simple_token}")
            else:
                tag_by_index.append(("CONTEXT_TABLE", simple_token))

    grouped_sae_output.tags_by_index = tags_by_index


@dataclass
class GroupedSaeOutput:
    """
    Class that collects and analyzes SaeOutputs over several layers.
    """
    sae_outputs_by_layer: dict[str, SaeOutput]
    text: str
    tokens: list[str]
    tags_by_index: dict

    def __init__(self, sae_outputs_by_layer, text, tokens, function_tagger=sql_tagger):
        self.sae_outputs_by_layer = sae_outputs_by_layer
        self.layers = list(self.sae_outputs_by_layer.keys())
        self.text = text
        self.tokens = tokens
        self.apply_tags(function_tagger)

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
        positions = self.get_indices_by_tag(tag)
        return self.sae_outputs_for_positions(positions)

    def get_indices_by_tag(self, tag):
        match_indices = []
        for index, tag_tuples in self.tags_by_index.items():
            tags = [tag_tuple[0] for tag_tuple in tag_tuples]
            if tag in tags:
                match_indices.append(index)
        return match_indices


class LoadedSAES:
    def __init__(self, dataset_name: str, full_model_name: str, model_alias: str,
                 tokenizer: AutoTokenizer, language_model: LanguageModel, layers: list[str],
                 layer_to_directory: dict, k: str, base_path: str, layer_to_saes: dict, dataset_mapper: Callable):

        self.dataset_name = dataset_name
        self.full_model_name = full_model_name
        self.model_alias = model_alias
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.layers = layers
        self.layer_to_directory = layer_to_directory
        self.k = k
        self.base_path = base_path
        self.layer_to_saes = layer_to_saes

        self.dataset = self.get_dataset()
        self.mapped_dataset = self.dataset.map(dataset_mapper)

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
        if "bm1" in self.full_model_name:
            with self.language_model.trace() as tracer:
                with tracer.invoke(text) as invoker:
                    eval_string = self.nnsight_eval_string_for_layer(layer)
                    my_output = eval(eval_string)
        if len(my_output) > 1:
            return my_output[0]
        else:
            return my_output

    def encode_to_sae_for_layer(self, text: str, layer: str):
        activations = self.encode_to_activations_for_layer(text, layer).cuda()
        raw_acts = activations[0].cpu().detach().numpy().tolist()
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

    def encode_to_all_saes(self, text: str):
        sae_outputs_by_layer = {layer: self.encode_to_sae_for_layer(text=text, layer=layer) for layer in self.layers}
        tokens = self.tokenizer.tokenize(text)
        result = GroupedSaeOutput(sae_outputs_by_layer=sae_outputs_by_layer, text=text, tokens=tokens)
        return result

    @staticmethod
    def load_from_path(model_alias: str, k: str, cache_dir: str, dataset_mapper: Callable):
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
            language_model = LanguageModel(full_model_name)
            tokenizer = language_model.tokenizer

        layer_to_saes = {layer: Sae.load_from_disk(directory).cuda() for layer, directory in layer_to_directory.items()}

        return LoadedSAES(dataset_name=dataset_name, full_model_name=full_model_name,
                          model_alias=model_alias, layers=layers, layer_to_directory=layer_to_directory,
                          tokenizer=tokenizer, k=k, base_path=base_path,
                          layer_to_saes=layer_to_saes, language_model=language_model, dataset_mapper=dataset_mapper)

    def get_dataset(self):
        return load_dataset(self.dataset_name)


class SaeCollector:
    """
    This class is responsible for collecting a large amount of text,
    assigning tags to each token for a feature name, and also SAE outputs.
    These can then be used for probes and feature analysis.
    (Still to add: ablations.)
    """

    def __init__(self, loaded_saes, seed: int, sample_size=10, restricted_tags=None):
        self.loaded_saes = loaded_saes
        self.restricted_tags = restricted_tags or []
        self.sample_size = sample_size
        self.mapped_dataset = loaded_saes.mapped_dataset
        self.mapped_dataset.shuffle(seed=seed)
        self.tokenizer = self.loaded_saes.tokenizer
        self.layers = self.loaded_saes.layers
        self.encoded_set = self.create_and_load_random_subset(sample_size=self.sample_size)

    def get_texts(self):
        return [element["prompt"] for element in self.encoded_set]

    def get_all_sae_outputs_for_tag(self, tag):
        sae_outputs_for_tags = [element["encoding"].sae_activations_and_indices_for_tag_by_layer(tag) for element in self.encoded_set]
        return sae_outputs_for_tags

    def get_prompt_and_encoding_for_text(self, feature):
        prompt = feature["prompt"]
        response = feature["response"]
        encoding = self.loaded_saes.encode_to_all_saes(prompt)

        return_dict = {
            "prompt": prompt,
            "response": response,
            "encoding": encoding
        }
        return return_dict

    def get_avg_reconstruction_error_for_all_k_and_layers(self):
        all_reconstruction_errors = {layer: self.get_avg_reconstruction_error_for_all_k(layer) for layer in self.layers}
        return all_reconstruction_errors

    def get_avg_reconstruction_error_for_all_k(self, layer, min_range=0, max_range=256):
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

    def create_and_load_random_subset(self, sample_size: int):
        sampled_set = self.mapped_dataset['train'].select(range(sample_size))
        encoded_set = []
        for element in tqdm(sampled_set):
            encoded_element = self.get_prompt_and_encoding_for_text(element)
            encoded_set.append(encoded_element)
        return encoded_set