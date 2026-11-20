"""The programmatic setting: the 20 rewards of the P-GRPO and P-DPO bases (Table 3), their z-stats, the
32 combined rewards and the held-out rewards.

Each reward is a deterministic function of the response text. A combined reward weights z-scored basis rewards,
r_tar = Σ_r α*_r (r − μ_r) / σ_r (Eq. 2). A held-out reward is one z-scored basis reward, and its target
policy π_tar is the held-out expert itself.

    REWARD_FUNCTIONS       reward -> score_<reward>(text), in the order of Table 3
    ZSTATS                 reward -> (μ_r, σ_r)
    COMBINED_REWARDS       name -> (rewards, α*), the 32 combined rewards
    make_combined_scorer   text -> Σ_r w_r (r(text) − μ_r) / σ_r; combined_reward and heldout_reward build it
    zscores                the (N, R) held-out rewards of responses, as the held-out fits regress on
    reward_zstats          (μ_r, σ_r) over a pool of responses
    verl_compute_score     the reward of the GRPO runs that train the P-GRPO experts and targets
    ensure_nltk            fetches and checks the NLTK data of the POS-tag, stopword and sentence rewards

The module imports only the standard library, numpy and nltk (lazily) and no other poem module, so verl can load it
by file path. The NLTK-based rewards depend on the NLTK version (3.9.4) and its data (NLTK_PACKAGES).
"""
import functools
import hashlib
import re
import warnings
from pathlib import Path

import numpy as np

# The NLTK data the paper used: package -> md5 of its zip.
NLTK_PACKAGES = {
    "tokenizers/punkt_tab": "27ea9ee08fe27fe4484e4c5585cce748",
    "taggers/averaged_perceptron_tagger_eng": "729e2255f83045670374180de9bdb613",
    "corpora/stopwords": "16da2726eaacf9c9982f68b054445604",
}


@functools.cache
def ensure_nltk():
    """Downloads the NLTK_PACKAGES that NLTK cannot find, then warns for each package whose zip on NLTK's search path
    differs from the paper's. Runs once per process. The NLTK-based rewards call it on first use; call it once before
    starting parallel reward workers, which would otherwise download into one directory at the same time."""
    import nltk
    for package, md5 in NLTK_PACKAGES.items():
        try:
            nltk.data.find(package)
        except (LookupError, FileNotFoundError):
            nltk.download(package.split("/")[1], quiet=True)
        zips = [z for z in (Path(d) / (package + ".zip") for d in nltk.data.path) if z.is_file()]
        if not zips or hashlib.md5(zips[0].read_bytes()).hexdigest() != md5:
            warnings.warn(f"NLTK package {package} is not the paper's (zip md5 {md5}); "
                          "the NLTK-based rewards may differ from the paper's")


# ---------------------------------------------------------------------------------------------------------- helpers

_PUNCT = ".,!?;:'\""


def _words(text, strip=_PUNCT, lower=True):
    """The whitespace-separated words of `text`, lower-cased, stripped of the characters in `strip`."""
    return [(w.lower() if lower else w).strip(strip) for w in text.split()]


def _clean_words(text, strip, lower=True):
    """_words without the words that strip to nothing."""
    return [w for w in _words(text, strip, lower) if w]


def _share(words, keep):
    """Share of the words for which keep(word) holds, × 100; 0 below five words."""
    if len(words) < 5:
        return 0.0
    return sum(1 for w in words if keep(w)) / len(words) * 100


def _word_share(text, vocab, strip=_PUNCT):
    """Share of the words in `vocab`, × 100; 0 below five words."""
    return _share(_words(text, strip), lambda w: w in vocab)


def _pos_share(text, tag_prefix):
    """Share of the NLTK tokens whose POS tag starts with `tag_prefix`, × 100; 0 below five tokens."""
    ensure_nltk()
    from nltk import pos_tag, word_tokenize
    toks = word_tokenize(text)
    if len(toks) < 5:
        return 0.0
    return sum(1 for _, t in pos_tag(toks) if t.startswith(tag_prefix)) / len(toks) * 100


def _syllables(word, strip, empty):
    """Vowel groups (a, e, i, o, u, y) of the lower-cased word stripped of `strip`, one less for a final e, at least 1;
    `empty` for a word that strips to nothing."""
    word = word.lower().strip(strip)
    if not word:
        return empty
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ---------------------------------------------------------------------------------------------- the 20 rewards

def score_vocab_diversity(text):
    """Vocabulary diversity: type-token ratio of the words; 0 below five words."""
    words = _clean_words(text, _PUNCT)
    if len(words) < 5:
        return 0.0
    return len(set(words)) / len(words)


def score_repetition(text):
    """Repetition: share of word bigrams that are unique (higher means less repetitive); 0 below four words."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return len(set(bigrams)) / len(bigrams)


def score_stopword_density(text):
    """Stopword density: share of tokens in the NLTK English stopword list, × 100."""
    ensure_nltk()
    from nltk.corpus import stopwords
    return _word_share(text, set(stopwords.words("english")), strip="")


def score_lexical_density(text):
    """Lexical density: share of words outside a fixed list of function words, × 100."""
    function_words = {
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
        "her", "its", "our", "their", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "us", "them", "who", "which", "what", "that",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "need", "dare",
        "and", "but", "or", "nor", "so", "yet", "for", "not", "no",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "over", "out", "off", "than", "then",
        "as", "if", "when", "while", "although", "because", "since", "unless",
        "also", "just", "very", "so", "too", "more", "most", "how",
    }
    return _share(_clean_words(text, _PUNCT + "()[]"), lambda w: w not in function_words)


def score_conjunction_density(text):
    """Conjunction density: share of words in a list of 21 conjunctions, × 100."""
    return _word_share(text, {"and", "but", "or", "nor", "yet", "so", "because", "although",
                              "though", "while", "whereas", "since", "unless", "until",
                              "whether", "if", "when", "whenever", "wherever", "before", "after"})


def score_formality(text):
    """Formality: mean word length, minus 10 times the contraction rate and 5 times the rate of first-person singular
    pronouns."""
    words = text.split()
    if not words:
        return 0.0
    contractions = sum(1 for w in words if "'" in w and w.lower().endswith(
        ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")))
    contraction_rate = contractions / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    casual_pronouns = sum(1 for w in words if w.lower() in ("i", "me", "my", "mine", "myself"))
    pronoun_rate = casual_pronouns / len(words)
    return avg_word_len - 10 * contraction_rate - 5 * pronoun_rate


def score_readability(text):
    """Readability: Flesch reading ease."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    if not words or not sentences:
        return 0.0
    total_syllables = sum(_syllables(w, _PUNCT, 1) for w in words)
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    return 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word


def score_mean_sentence_length(text):
    """Mean sentence length: mean number of words per NLTK sentence."""
    ensure_nltk()
    from nltk import sent_tokenize
    sents = sent_tokenize(text)
    if not sents:
        return 0.0
    lens = [len(s.split()) for s in sents]
    lens = [l for l in lens if l > 0]
    if not lens:
        return 0.0
    return sum(lens) / len(lens)


def score_avg_word_length(text):
    """Average word length: mean number of characters per word, punctuation and brackets stripped."""
    words = _clean_words(text, _PUNCT + "()[]{}", lower=False)
    if len(words) < 5:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def score_modal_density(text):
    """Modal density: share of words that are modal verbs, × 100."""
    return _word_share(text, {"can", "could", "would", "should", "may", "might", "must", "will", "shall", "ought"},
                       _PUNCT + "()[]{}")


def score_structure(text):
    """Structure: markdown structure per line (bullets, numbered items, headers and bold text), × 10."""
    lines = text.split("\n")
    score = 0.0
    for line in lines:
        s = line.strip()
        if s.startswith(("- ", "* ", "• ")):
            score += 1.0
        if re.match(r'^\d+[\.\)]\s', s):
            score += 1.0
        if s.startswith("#"):
            score += 1.5
        if "**" in s:
            score += 0.5
    return score / max(len(lines), 1) * 10


def score_paragraph_structure(text):
    """Paragraph structure: number of paragraphs with at least ten words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    return float(sum(1 for p in paragraphs if len(p.split()) >= 10))


def score_concreteness(text):
    """Concreteness: words that start with a digit (1 each) or are capitalized and longer than one letter
    (0.5 each), per 100 words."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    concrete = 0
    for w in words:
        s = w.strip(".,!?;:'\"()")
        if re.match(r'\d', s):
            concrete += 1
        elif s and s[0].isupper() and len(s) > 1:
            concrete += 0.5
    return concrete / len(words) * 100


def score_noun_density(text):
    """Noun density: share of NLTK tokens tagged as nouns (NN, NNS, NNP, NNPS), × 100."""
    return _pos_share(text, "NN")


def score_code_content(text):
    """Code content: code markers (backticks and keywords) per 100 words."""
    code_signals = text.count("```") * 3 + text.count("`") * 0.5
    prog_keywords = {"def ", "class ", "import ", "return ", "if ", "for ", "while ",
                     "function ", "const ", "var ", "let ", "print(", "console.log",
                     "==", "!=", "=>", "->", "()"}
    for kw in prog_keywords:
        code_signals += text.count(kw)
    words = len(text.split())
    if words < 5:
        return 0.0
    return code_signals / words * 100


def score_verb_density(text):
    """Verb density: share of NLTK tokens tagged as verbs (VB, VBD, VBG, VBN, VBP, VBZ), × 100."""
    return _pos_share(text, "VB")


def score_adverb_density(text):
    """Adverb density: share of words ending in -ly, outside a list of exceptions, × 100."""
    exceptions = {"only", "family", "early", "daily", "weekly", "monthly",
                  "yearly", "holy", "ugly", "likely", "lonely", "friendly",
                  "lovely", "belly", "ally", "tally", "rally", "supply", "apply",
                  "reply", "july", "italy", "bully", "fully", "lily"}
    return _share(_words(text), lambda w: w.endswith("ly") and len(w) > 3 and w not in exceptions)


def score_personal_narrative(text):
    """Personal narrative: share of words that are first-person pronouns or end in -ed, × 100."""
    first_person = {"i", "we", "my", "our", "me", "us", "myself", "ourselves"}   # none ends in -ed
    return _share(_words(text), lambda w: w in first_person or (w.endswith("ed") and len(w) > 3))


def score_polysyllabic_word_density(text):
    """Polysyllabic word density: share of words with three or more syllables, × 100."""
    return _share(text.split(), lambda w: _syllables(w, _PUNCT + "()[]{}0123456789", 0) >= 3)


def score_second_person(text):
    """Second person: share of words that are second-person pronouns, × 100."""
    return _word_share(text, {"you", "your", "yours", "yourself", "yourselves"})


REWARD_FUNCTIONS = {
    "vocab_diversity": score_vocab_diversity,
    "repetition": score_repetition,
    "stopword_density": score_stopword_density,
    "lexical_density": score_lexical_density,
    "conjunction_density": score_conjunction_density,
    "formality": score_formality,
    "readability": score_readability,
    "mean_sentence_length": score_mean_sentence_length,
    "avg_word_length": score_avg_word_length,
    "modal_density": score_modal_density,
    "structure": score_structure,
    "paragraph_structure": score_paragraph_structure,
    "concreteness": score_concreteness,
    "noun_density": score_noun_density,
    "code_content": score_code_content,
    "verb_density": score_verb_density,
    "adverb_density": score_adverb_density,
    "personal_narrative": score_personal_narrative,
    "polysyllabic_word_density": score_polysyllabic_word_density,
    "second_person": score_second_person,
}

ZSTATS = {  # reward: (μ_r, σ_r), mean and standard deviation over the 36,000 responses of the DPO pairs
    "vocab_diversity": (0.7274252824959075, 0.07899537330534422),
    "repetition": (0.9693766811910067, 0.05195394404243148),
    "stopword_density": (34.81884107526238, 8.29546006229396),
    "lexical_density": (65.88842432789195, 8.542835218201603),
    "conjunction_density": (5.07143440341905, 1.9746466304608714),
    "formality": (5.338885983965881, 0.8963508870770182),
    "readability": (42.872464043864056, 21.40226641136815),
    "mean_sentence_length": (18.35742205463602, 11.094271206725512),
    "avg_word_length": (5.312419571831108, 0.8182623058176693),
    "modal_density": (1.0977534161240707, 1.083405963513738),
    "structure": (3.3410458542442267, 3.409097058847311),
    "paragraph_structure": (4.107888888888889, 2.012026072590445),
    "concreteness": (8.248418347647483, 4.628505339590622),
    "noun_density": (29.324178416803694, 6.741766507649543),
    "code_content": (1.826459860366179, 2.9076027897335237),
    "verb_density": (13.218601305817083, 3.5754487952379748),
    "adverb_density": (1.2989957349441918, 1.076099534799098),
    "personal_narrative": (4.294382617002356, 2.9743918586425635),
    "polysyllabic_word_density": (19.928071582620912, 8.434693496178896),
    "second_person": (0.9555107826741425, 1.6435619252840197),
}

COMBINED_REWARDS = {  # name: (rewards, α*); 6, 6, 8, 6 and 6 of the 32 mix k = 2, 4, 8, 10 and 16 rewards
    "clean_k2_conc_medium": (("readability", "vocab_diversity"), (0.65, 0.35)),
    "clean_k2_conc_peaked": (("readability", "vocab_diversity"), (0.85, 0.15)),
    "clean_k2_conc_uniform": (("readability", "vocab_diversity"), (0.5, 0.5)),
    "clean_k2_r0_code_content_concreteness": (("concreteness", "code_content"), (0.8972038510508373, 0.10279614894916263)),
    "clean_k2_r1_code_content_vocab_diversity": (("vocab_diversity", "code_content"), (0.7075857981849016, 0.2924142018150983)),
    "clean_k2_r2_concreteness_paragraph_structu": (("paragraph_structure", "concreteness"), (0.6482914726397128, 0.35170852736028724)),
    "clean_k4_conc_medium": (("code_content", "formality", "repetition", "readability"), (0.5, 0.3, 0.15, 0.05)),
    "clean_k4_conc_peaked": (("code_content", "formality", "repetition", "readability"), (0.85, 0.05, 0.05, 0.05)),
    "clean_k4_conc_uniform": (("code_content", "formality", "repetition", "readability"), (0.25, 0.25, 0.25, 0.25)),
    "clean_k4_r0_conjunction_density_formality_": (("readability", "repetition", "conjunction_density", "formality"), (0.19722438147702298, 0.15656465715224807, 0.5809225829527335, 0.06528837841799544)),
    "clean_k4_r1_code_content_formality_readabi": (("code_content", "formality", "readability", "vocab_diversity"), (0.26040783331912054, 0.19578605507147656, 0.43924085365995075, 0.1045652579494522)),
    "clean_k4_r2_code_content_paragraph_structu": (("readability", "paragraph_structure", "repetition", "code_content"), (0.4504634183965019, 0.41960414070585483, 0.07867133074718805, 0.051261110150455286)),
    "clean_k8_conc_medium": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.46875, 0.1875, 0.104167, 0.072917, 0.052083, 0.041667, 0.041667, 0.03125)),
    "clean_k8_conc_peaked": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.82, 0.025714, 0.025714, 0.025714, 0.025714, 0.025714, 0.025714, 0.025714)),
    "clean_k8_conc_uniform": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)),
    "clean_k8_r0_code_content_concreteness_conj": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.198533312402776, 0.15227033165278694, 0.25707802774741745, 0.08112939143395394, 0.06348015204193475, 0.1106716319806243, 0.035820566104270646, 0.10101658663623601)),
    "clean_k8_r1_code_content_concreteness_conj": (("code_content", "formality", "repetition", "conjunction_density", "readability", "paragraph_structure", "concreteness", "vocab_diversity"), (0.05644598698075565, 0.14499551177568878, 0.177693183541167, 0.1017727668674378, 0.023652166649193798, 0.3794082212579543, 0.09446582373014478, 0.02156633919765785)),
    "clean_k8_r2_code_content_concreteness_conj": (("repetition", "conjunction_density", "formality", "code_content", "readability", "paragraph_structure", "concreteness", "vocab_diversity"), (0.08902937186514191, 0.21683497528765178, 0.06510413857398845, 0.2783766380994757, 0.038162886403545315, 0.09680792458694357, 0.15199168776426505, 0.06369237741898806)),
    "clean_k8_r3": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.021004, 0.074393, 0.267199, 0.42005, 0.065579, 0.049226, 0.082963, 0.019587)),
    "clean_k8_r4": (("repetition", "conjunction_density", "vocab_diversity", "code_content", "readability", "paragraph_structure", "concreteness", "formality"), (0.330948, 0.037531, 0.282281, 0.0123, 0.081913, 0.069266, 0.053525, 0.132236)),
    "clean_k10_conc_medium": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.45, 0.18, 0.1, 0.07, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02)),
    "clean_k10_conc_peaked": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.82, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)),
    "clean_k10_conc_uniform": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)),
    "clean_k10_r0": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.047865, 0.071776, 0.001394, 0.00016, 0.038742, 0.114742, 0.047418, 0.05317, 0.198291, 0.426443)),
    "clean_k10_r1": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.098603, 0.028344, 0.493959, 0.033672, 0.010601, 0.165387, 0.045821, 0.050671, 0.00273, 0.070212)),
    "clean_k10_r2": (("readability", "repetition", "vocab_diversity", "conjunction_density", "formality", "concreteness", "structure", "code_content", "paragraph_structure", "adverb_density"), (0.023085, 0.038892, 0.091278, 0.125765, 0.215654, 0.238162, 0.044017, 0.031017, 0.089886, 0.102244)),
    "clean_k16_conc_medium": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.425331, 0.170132, 0.094518, 0.066163, 0.047259, 0.037807, 0.037807, 0.028355, 0.018904, 0.018904, 0.014178, 0.011342, 0.009452, 0.007561, 0.006616, 0.005671)),
    "clean_k16_conc_peaked": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.82, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012)),
    "clean_k16_conc_uniform": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625)),
    "clean_k16_r0": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.031257, 0.046872, 0.000911, 0.000104, 0.0253, 0.07493, 0.030965, 0.034722, 0.12949, 0.27848, 0.15108, 5.9e-05, 0.104312, 0.003333, 0.049161, 0.039026)),
    "clean_k16_r1": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.071749, 0.020625, 0.359435, 0.024502, 0.007714, 0.120345, 0.033342, 0.036871, 0.001987, 0.05109, 0.069638, 0.015419, 0.112149, 0.013949, 0.052574, 0.00861)),
    "clean_k16_r2": (("vocab_diversity", "repetition", "lexical_density", "conjunction_density", "formality", "readability", "avg_word_length", "modal_density", "structure", "paragraph_structure", "concreteness", "code_content", "adverb_density", "personal_narrative", "polysyllabic_word_density", "second_person"), (0.009946, 0.016756, 0.039325, 0.054184, 0.092911, 0.102608, 0.018964, 0.013363, 0.038726, 0.04405, 0.141531, 0.027311, 0.071001, 0.081826, 0.033871, 0.213627)),
}


# ------------------------------------------------------------------------------------ combined and held-out rewards

def make_combined_scorer(rewards, weights, zstats, functions=REWARD_FUNCTIONS):
    """The scorer x -> Σ_r w_r (r(x) − μ_r) / σ_r over the aligned `rewards` and `weights`, with r(x) = functions[r](x)
    (by default the programmatic reward of a text x) and (μ_r, σ_r) = zstats[r]. Each term is float(w_r) · (r − μ_r) /
    σ_r, summed left to right by Python's sum() (from Python 3.12 on, sum() compensates rounding and changes the last
    bits)."""
    specs = [(functions[r], float(zstats[r][0]), float(zstats[r][1]), float(w)) for r, w in zip(rewards, weights)]

    def score(text):
        return float(sum(w * (f(text) - m) / sd for f, m, sd, w in specs))
    return score


def reward_zstats(texts, rewards):
    """{reward: (μ_r, σ_r)} over a pool of responses: the mean and the population standard deviation plus 1e-8 (ZSTATS
    holds that epsilon)."""
    stats = {}
    for r in rewards:
        vals = np.array([REWARD_FUNCTIONS[r](s) for s in texts], dtype=float)
        stats[r] = (float(vals.mean()), float(vals.std() + 1e-8))
    return stats


def combined_reward(name, weights_dtype=np.float64):
    """The scorer of a combined reward of COMBINED_REWARDS, with α* cast to `weights_dtype`: float64 for the fit
    targets and training, float32 to score decodes, as the paper did."""
    rewards, weights = COMBINED_REWARDS[name]
    return make_combined_scorer(rewards, np.asarray(weights, dtype=weights_dtype), ZSTATS)


def heldout_reward(name):
    """The scorer of a held-out reward: the z-scored basis reward `name`."""
    return make_combined_scorer([name], [1.0], ZSTATS)


def zscores(texts, rewards):
    """(N, R) held-out rewards of the texts: column r is heldout_reward(rewards[r]), the z-scored basis reward."""
    scorers = [heldout_reward(r) for r in rewards]
    return np.array([[f(x) for f in scorers] for x in texts])


def verl_compute_score(*, solution_str, reward_name=None, combined_reward_name=None, **_):
    """verl's custom reward function (`custom_reward_function.name=verl_compute_score`, loaded from this file): the
    basis reward `reward_name` (the expert's GRPO run) or the combined reward `combined_reward_name` (π_tar's run) of
    the response, both set through `reward_kwargs`."""
    if combined_reward_name is not None:
        return combined_reward(combined_reward_name)(solution_str)
    return float(REWARD_FUNCTIONS[reward_name](solution_str))
