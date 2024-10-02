import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import transformers
from ft_datasets import nt_benchmarks
from max_pool_wrapper import MaxPoolWrapper
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    AutoModelForCausalLM
)
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

transformers.utils.TRUST_REMOTE_CODE = True


def load_model_tokenizer(config):
    model_types = {
        "nt_50m": load_nt_50m,
        "nt_50m_max_pool": load_nt_50m_max_pool,
        "nt_500m": load_nt_500m,
        "nt_500m_max_pool": load_nt_500m_max_pool,
        "hyena": load_hyena,
        "hyena_max_pool": load_hyena_max_pool,
        "genalm": load_genalm,
        "genalm_max_pool": load_genalm_max_pool,
        "dnabert": load_dnabert,
        "dnabert_max_pool": load_dnabert_max_pool,
        "caduceus": load_caduceus,
        "caduceus_max_pool": load_caduceus_max_pool,
        "mistral_max_pool": load_mistral_max_pool,
    }

    load_function = model_types.get(config.model_type)
    if load_function:
        model, tokenizer = load_function(config)
    else:
        raise NotImplementedError(
            "Model type must be one of: " + ", ".join(model_types.keys())
        )

    print(f"Loaded model and tokenizer based on {config.model_type}")
    print(model)
    print(tokenizer)
    return model, tokenizer


def load_hyena(config):
    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random Hyena model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=config.num_classes, trust_remote_code=True
        )
        print("Using pre-trained Hyena model")
    return model, tokenizer


def load_caduceus(config):
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
    model_config.fused_add_norm = False
    if config.train_type == "random":
        model = AutoModelForSequenceClassification.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random Caduceus model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Initializing trained Caduceus model")
    return model, tokenizer


def load_caduceus_max_pool(config):
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
    model_config.fused_add_norm = False
    if config.train_type == "random":
        base_model = AutoModelForMaskedLM.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random Caduceus model with max pooling")
    else:
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Using pre-trained Caduceus model with max pooling")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_hyena_max_pool(config):
    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_config(model_config, trust_remote_code=True)
        print("Initializing random Hyena model")
    else:
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("Using pre-trained Hyena model")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_genalm(config):
    model_name = "AIRI-Institute/gena-lm-bert-base-t2t"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_config(model_config)
        print("Initializing random GenaLM model")
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=config.num_classes, trust_remote_code=True
        )
        print("Using pre-trained GenaLM model")
    return model, tokenizer


def load_genalm_max_pool(config):
    model_name = "AIRI-Institute/gena-lm-bert-base-t2t"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model_config.output_hidden_states = True
        base_model = AutoModel.from_config(model_config)
        print("Initializing random GenaLM model")
    else:
        base_model = AutoModel.from_pretrained(
            model_name,
            num_labels=config.num_classes,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        print("Using pre-trained GenaLM model")
    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_nt_50m(config):
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random Nucleotide Transformer 50M model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=nt_benchmarks[config.dataset_name]["num_labels"],
            trust_remote_code=True,
        )
        print("Using pre-trained Nucleotide Transformer 50M model")
    return model, tokenizer


def load_nt_50m_max_pool(config):
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        base_model = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True),
            trust_remote_code=True,
        )
        print("Initializing random Nucleotide Transformer 50M model with max pooling")
    else:
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        print("Using pre-trained Nucleotide Transformer 50M model with max pooling")

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_nt_500m(config):
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random Nucleotide Transformer 500M model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=nt_benchmarks[config.dataset_name]["num_labels"],
            trust_remote_code=True,
        )
        print("Using pre-trained Nucleotide Transformer 500M model")
    return model, tokenizer


def load_nt_500m_max_pool(config):
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        base_model = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True),
            trust_remote_code=True,
        )
        print("Initializing random Nucleotide Transformer 500M model with max pooling")
    else:
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        print("Using pre-trained Nucleotide Transformer 500M model with max pooling")

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_dnabert_max_pool(config):
    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if config.train_type == "random":
        model_config = BertConfig.from_pretrained(model_name)
        base_model = AutoModelForMaskedLM.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random DNABERT model")
    else:
        model_config = BertConfig.from_pretrained(model_name)
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Using pre-trained DNABERT model")  

    model = MaxPoolWrapper(
        base_model, config.model_type, nt_benchmarks[config.dataset_name]["num_labels"]
    )
    return model, tokenizer


def load_mistral_max_pool(config):
    raise NotImplementedError("The weights will be upon release!")


def load_dnabert(config):
    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if config.train_type == "random":
        model_config = BertConfig.from_pretrained(model_name)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_config(
            model_config, trust_remote_code=True
        )
        print("Initializing random DNABERT model")
    else:
        model_config = BertConfig.from_pretrained(model_name)
        model_config.num_labels = nt_benchmarks[config.dataset_name]["num_labels"]
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
        print("Using pre-trained DNABERT model")
    return model, tokenizer


class CharacterTokenizerLlama(PreTrainedTokenizer):
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers."""

        self._vocab_str_to_int = {
            **{ch: i for i, ch in enumerate(characters)},
            "[CLS]": 6,
            "[SEP]": 7,
            "[UNK]": 8,
            "[PAD]": 9,
            "[EOS]": 10,
            "[BOS]": 11,
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int.copy()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizerLlama":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
