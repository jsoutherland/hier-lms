# Copyright 2025 Joshua Southerland
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional
from spacy import load as spacy_load
from enum import Enum

# TODO: Use a tokenizer config file
MAX_CHARS = 20
MAX_WORD_LEN = MAX_CHARS  # includes [W_CLS], [W_SEP]


class SpecialToken(str, Enum):
    PAD = "[PAD]"
    MASK = "[MASK]"
    W_CLS = "[W_CLS]"
    W_SEP = "[W_SEP]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNK = "[UNK]"


special_map = {}
special_map[SpecialToken.PAD.value] = len(special_map)
special_map[SpecialToken.MASK.value] = len(special_map)
special_map[SpecialToken.W_CLS.value] = len(special_map)
special_map[SpecialToken.W_SEP.value] = len(special_map)
special_map[SpecialToken.CLS.value] = len(special_map)
special_map[SpecialToken.SEP.value] = len(special_map)
special_map[SpecialToken.UNK.value] = len(special_map)

NUM_SPECIALS = len(special_map)
ORD_OFFSET = NUM_SPECIALS


MAX_CHAR_VOCAB = 256+NUM_SPECIALS
MAX_ORD_VAL = MAX_CHAR_VOCAB - 1

# Maps common chars above 255 to the 0-255 range
# comments are based on a 1,200,000 random subsample
# ord, count, frequency of over 255 chars
# I don't always agree with unidecode unfortunately
ORD_TO_CHAR = {
    257: "a",       # 257 46551 0.002653589511232993 ā
    259: "a",       # 259 27736 0.0015810607437768962 ă
    263: "c",       # 263 41335 0.0023562570610043986 ć
    269: "c",       # 269 26198 0.0014933887137823453 č
    299: "i",       # 299 14168 0.0008076315480902461 ī
    305: "i",       # 305 21088 0.0012020986791450529 ı
    322: "l",       # 322 27899 0.0015903523828465398 ł
    333: "0",       # 333 40134 0.0022877953522765344 ō
    351: "s",       # 351 14624 0.0008336253359169791 ş
    353: "s",       # 353 32046 0.0018267476418760606 š
    363: "u",       # 363 17404 0.0009920962353869736 ū
    382: "z",       # 382 13834 0.0007885922385855777 ž
    945: "a",       # 945 14627 0.0008337963476789971 α
    1072: "a",      # 1072 66314 0.0037801579954867717 а
                    # 1074 32516 0.0018535394845922107 в
                    # 1076 20830 0.0011873916676115068 д
    1077: "e",      # 1077 61531 0.003507508242909439 е
                    # 1080 58133 0.0033138089204637405 и
    1082: "k",      # 1082 29808 0.0016991728674106477 к
                    # 1083 29491 0.0016811026245574145 л
                    # 1084 20137 0.0011478879505853533 м
                    # 1085 48436 0.002761041901700957 н
    1086: "o",      # 1086 78211 0.004458333639729407 о
                    # 1087 17096 0.0009745390278197944 п
                    # 1088 40159 0.0022892204502933506 р
                    # 1089 40140 0.0022881373758005703 с
                    # 1090 43023 0.002452479679099849 т
                    # 1091 18570 0.0010585628068912952 у
                    # 1103 13973 0.0007965157835590775 я
                    # 1575 21547 0.0012282634787338037 ا
                    # 1604 15697 0.0008947905427987431 ل
    8211: chr(150), # 8211 2022878 0.11531197704246901 –
    8212: chr(151), # 8212 1135872 0.0647491573822956 —
    8213: chr(151),
    8216: "'",      # 8216 307777 0.01754449569286926 ‘
    8217: "'",      # 8217 5808056 0.33108196348636665 ’
    8220: '"',      # 8220 2586827 0.1474592811018949 “
    8221: '"',      # 8221 2537909 0.14467076331042972 ”
    8224: chr(134), # 8224 22054 0.0012571644665148424 †
    8226: chr(149), # 8226 63172 0.003601051676733274 •
    8230: chr(133), # 8230 237369 0.013530963646148619 …
    8364: chr(128), # 8364 28545 0.0016271769156010782 €
    8722: "-",      # 8722 32291 0.0018407136024408623 − (dash)
                    # 65533 20816 0.0011865936127220893 � (unk)
    8943: chr(133),
}


class HlmTokenizer:
    def __init__(self):
        self.nlp = spacy_load("en_core_web_sm",
                              disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        self.nlp.enable_pipe("senter")
        self.pad_id = special_map[SpecialToken.PAD.value]
        self.vocab_size = MAX_CHAR_VOCAB
        self.mask_id = special_map[SpecialToken.MASK.value]
        self.num_specials = NUM_SPECIALS

    def build_specials_mask(self, input_ids):
        return input_ids < NUM_SPECIALS

    def tokenize_word(self, word: str) -> List[str]:
        out = [SpecialToken.W_CLS.value]
        for char in word[:MAX_WORD_LEN-2]:
            ord_char = ord(char)
            _id = ord_char+ORD_OFFSET
            if _id > MAX_ORD_VAL:
                new_char = ORD_TO_CHAR.get(ord_char)
                if new_char is not None:
                    tok = new_char
                else:
                    tok = SpecialToken.UNK.value
            else:
                tok = char
            out.append(tok)
        out.append(SpecialToken.W_SEP.value)
        return out

    def tokenize(self, input_seq: str, max_length: Optional[int] = None) -> List[str]:
        doc = self.nlp(input_seq)
        out = [[SpecialToken.CLS.value]]

        for sent in doc.sents:
            word_toks = [self.tokenize_word(word.text) for word in sent]
            out.extend(word_toks)
            if max_length and len(out) >= (max_length-1): # -1 because we plan to add SEP
                break

        out.append([SpecialToken.SEP.value])
        if max_length:
            out = out[:max_length]

        return out

    def decode_word(self, tokens: List[int]) -> str:
        out = ""
        for tok in tokens:
            if tok < ORD_OFFSET:
                continue
            out += chr(tok-ORD_OFFSET)
        return out

    def encode(self,
               input_seq: str,
               return_special_tokens_mask: bool = False,
               max_length: Optional[int] = None) -> Dict[str, List[int]]:
        all_toks = self.tokenize(input_seq, max_length)
        input_ids  = []
        for toks in all_toks:
            # sentence
            word_out = []
            for char_tok in toks:
                # word
                if char_tok in special_map:
                    _id = special_map[char_tok]
                else:
                    _id = ord(char_tok) + ORD_OFFSET
                word_out.append(_id)

            input_ids.append(word_out)

        result = {
            "input_ids": input_ids
        }


        if return_special_tokens_mask:
            special_mask = []
            for tok_seq in input_ids:
                seq_mask = []
                for tok in tok_seq:
                    if tok < NUM_SPECIALS:
                        seq_mask.append(1)
                    else:
                        seq_mask.append(0)
                special_mask.append(seq_mask)
            result["attention_mask"] = special_mask

        return result


    def batch_encode(self,
                     input_seqs: List[str],
                     return_special_tokens_mask: bool = False,
                     max_length: Optional[int] = None) -> [Dict[str, List[List[int]]]]:
        out = [
            self.encode(input_seq,
                        return_special_tokens_mask=return_special_tokens_mask,
                        max_length=max_length) for input_seq in input_seqs
        ]

        # [ {input_ids: [], specials: []}, {input_ids: [], specials: []}....]

        result = {}
        for k in out[0].keys():
            result[k] = [seq_out[k] for seq_out in out]

        return result

    def __call__(self,
                 input_seqs: List[str],
                 max_length: int = 512,
                 truncation: bool = True,
                 padding: bool = False, # TODO: implement
                 return_special_tokens_mask: bool = False
    ) -> Dict[str, List[List[int]]]:
        # TODO: implement max length, default to None
        # TODO: truncation
        # TODO: return type hint
        return self.batch_encode(input_seqs,
                                 return_special_tokens_mask=return_special_tokens_mask,
                                 max_length=max_length)

    def pad(self, *args, **kwargs):
        # TODO: implement for transformers lib
        pass

    @classmethod
    def save_pretrained(cls, output_dir):
        # TODO: save config
        pass  # noop needed for transformers.Trainer
