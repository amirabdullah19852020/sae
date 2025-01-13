from copy import deepcopy
from dataclasses import dataclass

import torch
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