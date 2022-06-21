# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# The script is largely adapted from the huggingface transformers library

import copy
import json
import os
import re
import unicodedata
from typing import Optional, Tuple, Union, List, Any

from tokenizers import AddedToken
from tokenizers import Encoding as EncodingFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.dynamic_module_utils import custom_object_save  # type:ignore
from transformers.utils import logging
from cocolm.tokenization_utils import Dictionary

logger = logging.get_logger(__name__)

VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class SentencepiecePreTokenizer(object):
    def __init__(self):
        self.transl_table = dict(
            [(ord(x), ord(y)) for x, y in zip("‘’´“”—–-", "'''\"\"---")]
        )

    def handle_single_quote(self, tokens):
        line = " ".join(tokens)
        line = re.sub(r"' ([smdSMDtT])\b", r"'\1", line)
        line = re.sub(r"' ll\b", "'ll", line)
        line = re.sub(r"' re\b", "'re", line)
        line = re.sub(r"' ve\b", "'ve", line)
        line = re.sub(r"' LL\b", "'LL ", line)
        line = re.sub(r"' RE\b", "'RE ", line)
        line = re.sub(r"' VE\b", "'VE ", line)
        return line.split()

    def split_on_cont_punc(self, tokens):
        new_tokens = []
        for token in tokens:
            if len(token) > 1:
                last_j = 0
                pre_is_punc = _is_punctuation(token[0])
                for j, ch in enumerate(token):
                    is_punc = _is_punctuation(ch)
                    if is_punc != pre_is_punc:
                        new_tokens.append(token[last_j:j])
                        last_j = j
                    pre_is_punc = is_punc
                if last_j < len(token):
                    new_tokens.append(token[last_j:])
            else:
                new_tokens.append(token)
        return new_tokens

    def split_pre_and_post_punc(self, tokens):
        def pre_punc(token):
            last_j = 0
            for j in range(1, len(token)):
                if not _is_punctuation(token[j]):
                    last_j = j
                    break
            return token[:last_j], token[last_j:]

        def post_punc(token):
            last_j = len(token)
            for j in range(len(token) - 2, -1, -1):
                is_punc = _is_punctuation(token[j])
                if not _is_punctuation(token[j]):
                    last_j = j + 1
                    break
            return token[:last_j], token[last_j:]

        new_tokens = []
        for token in tokens:
            if len(token) > 1 and _is_punctuation(token[0]):
                a, b = pre_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    if _is_punctuation(b[-1]):
                        c, d = post_punc(b)
                        if c:
                            new_tokens.append(c)
                        if d:
                            new_tokens.append(d)
                    else:
                        new_tokens.append(b)
            elif len(token) > 1 and _is_punctuation(token[-1]):
                a, b = post_punc(token)
                if a:
                    new_tokens.append(a)
                if b:
                    new_tokens.append(b)
            else:
                new_tokens.append(token)
        return new_tokens

    def tokenize(self, line):
        line = line.strip()
        line = line.replace("``", '"').replace("''", '"')
        line = line.translate(self.transl_table)
        tokens = line.split()
        tokens = self.split_pre_and_post_punc(tokens)
        tokens = self.handle_single_quote(tokens)
        return tokens


COCOLM_VOCAB_FILES_NAMES = {"vocab_file": "sp.model", "dict_file": "dict.txt"}

COCOLM_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cocolm-cased": "https://huggingface.co/microsoft/cocolm-base/resolve/main/sp.model",
    },
    "dict_file": {
        "cocolm-cased": "https://huggingface.co/microsoft/cocolm-base/resolve/main/dict.txt"
    },
}

COCOLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cocolm-cased": 512,
}


class COCOLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = COCOLM_VOCAB_FILES_NAMES
    pretrained_vocab_files_map = COCOLM_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = COCOLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, dict_file, **kwargs):
        super(COCOLMTokenizer, self).__init__(**kwargs)
        if not os.path.exists(vocab_file):
            raise EnvironmentError("file {} not found".format(vocab_file))
        try:
            import sentencepiece as spm  # type: ignore

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab_file)
            self.pre_tokenizer = SentencepiecePreTokenizer()
            self.dictionary = Dictionary.load(dict_file)
        except ImportError:
            raise ImportError(
                "Please install sentencepiece with: pip install sentencepiece"
            )
        self.dictionary.add_symbol("<mask>")

    @property
    def cls_token(self):
        return self.dictionary.alias_mapper[self.dictionary.bos_word]

    @property
    def sep_token(self):
        return self.dictionary.alias_mapper[self.dictionary.eos_word]

    @property
    def pad_token(self):
        return self.dictionary.alias_mapper[self.dictionary.pad_word]

    @property
    def unk_token(self):
        return self.dictionary.alias_mapper[self.dictionary.unk_word]

    @property
    def cls_token_id(self):
        return self.dictionary.bos_index

    @property
    def sep_token_id(self):
        return self.dictionary.eos_index

    @property
    def pad_token_id(self):
        return self.dictionary.pad_index

    @property
    def mask_token_id(self):
        return self.dictionary.index("<mask>")

    @property
    def unk_token_id(self):
        return self.dictionary.unk_index

    def encode_plus(self, text_a, text_b=None, add_special_tokens=True, max_length=512):
        tokens_a = self.tokenize(text_a)
        if text_b is not None:
            tokens_b = self.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 4)
        else:
            if len(tokens_a) > max_length - 2:
                tokens_a = tokens_a[: max_length - 2]

        if add_special_tokens:
            tokens = [self.dictionary.bos_word] + tokens_a + [self.dictionary.eos_word]
            if text_b is not None:
                tokens += (
                    [self.dictionary.eos_word] + tokens_b + [self.dictionary.eos_word]
                )
        else:
            tokens = tokens_a + tokens_b

        ids = self.convert_tokens_to_ids(tokens)
        return {"input_ids": ids}

    def encode(self, x: str, add_special_tokens=False) -> str:
        tokens = self.tokenize(x)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list) -> str:
        x = "".join([self._convert_id_to_token(token_id) for token_id in ids])
        return x.replace(" ", "").replace("\u2581", " ").strip()

    def skip_space(self, tokens):
        new_tokens = []
        for i, token in enumerate(tokens):
            skip = False
            # skip single space, to reduce total length
            if token == "\u2581":
                if i == len(tokens) - 1 or _is_punctuation(tokens[i + 1][0]):
                    skip = True
            if not skip:
                new_tokens.append(token)
        return new_tokens

    def tokenize(self, x):
        x = " ".join(self.pre_tokenizer.tokenize(x))
        tokens = self.sp.EncodeAsPieces(x)
        tokens = self.skip_space(tokens)
        return tokens

    def convert_tokens_to_ids(self, tokens: list):
        ret = []
        if isinstance(tokens, str):
            return self.dictionary.index(tokens)
        for token in tokens:
            ret.append(self.dictionary.index(token))
        return ret

    def _convert_id_to_token(self, index):
        """Converts a token (str) in an id using the vocab."""
        token = self.dictionary[index]
        return token

    def convert_tokens_to_string(self, tokens: list):
        x = " ".join(tokens)
        return x.replace(" ", "").replace("\u2581", " ").strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>", "[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save the full tokenizer state.
        This method make sure the full tokenizer can then be re-loaded using the
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..
        Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
        instance, modifying `tokenizer.do_lower_case` after creation).
        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (`bool`, *optional*):
                Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
                format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
                added_tokens files.
                If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
                "slow" tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
                loaded in the corresponding "slow" tokenizer.
                If `True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a value
                error is raised.
            filename_prefix: (`str`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.
                <Tip warning={true}>
                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.
                </Tip>
        Returns:
            A tuple of `str`: The files saved.
        """
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        special_tokens_map_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + SPECIAL_TOKENS_MAP_FILE,
        )
        tokenizer_config_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE,
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        # Sanitize AddedTokens
        def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
            if isinstance(obj, AddedToken):
                out = obj.__getstate__()
                if add_type_field:
                    out["__type"] = "AddedToken"
                return out
            elif isinstance(obj, (list, tuple)):
                return list(
                    convert_added_tokens(o, add_type_field=add_type_field) for o in obj
                )
            elif isinstance(obj, dict):
                return {
                    k: convert_added_tokens(v, add_type_field=add_type_field)
                    for k, v in obj.items()
                }
            return obj

        # add_type_field=True to allow dicts in the kwargs / differentiate from AddedToken serialization
        tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`
        if (
            tokenizer_class.endswith("Fast")
            and tokenizer_class != "PreTrainedTokenizerFast"
        ):
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=tokenizer_config)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = (
                json.dumps(
                    tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n"
            )
            f.write(out_str)
        logger.info(f"tokenizer config file saved in {tokenizer_config_file}")

        # Sanitize AddedTokens in special_tokens_map
        write_dict = convert_added_tokens(
            self.special_tokens_map_extended, add_type_field=False
        )
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            out_str = (
                json.dumps(write_dict, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            f.write(out_str)
        logger.info(f"Special tokens file saved in {special_tokens_map_file}")

        file_names = (tokenizer_config_file, special_tokens_map_file)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Tokenizer pushed to the hub in this commit: {url}")

        return save_files

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.
        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`]
        """
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE,
        )
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = (
                    json.dumps(
                        added_vocab, indent=2, sort_keys=True, ensure_ascii=False
                    )
                    + "\n"
                )
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        return file_names + (added_tokens_file,)
