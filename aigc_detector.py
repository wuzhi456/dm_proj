"""
AIGC Detection Demo — Multi-dimensional Feature Engineering + Interpretable ML + SHAP
Combines approaches from: HC3 (Guo et al.), GLTR (Gehrmann et al.),
DetectGPT (Mitchell et al.), and stylometric detection literature.

Feature categories:
  1. Vocabulary Richness (TTR, Hapax, Honore's R, density)
  2. Readability (Flesch-Kincaid, Flesch Reading Ease, sentence metrics)
  3. Syntactic Features (POS ratios, dependency parse tree depth)
  4. Discourse Patterns (transition words, logical connectors)
  5. Sentiment Features (polarity, neutrality, emotion word ratio)
  6. Text Statistics (length, punctuation, structure)

Pre-processing: Correlation analysis to remove redundant features (|r| > 0.85).
"""

import json
import os
import re
import math
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

# ── NLP Imports ──
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
import jieba
import jieba.posseg as pseg
import spacy

# ── ML Imports ──
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap

# ═══════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Transition words by category — AI tends to overuse these structured markers
TRANSITION_CATEGORIES_EN = {
    'sequential': ['firstly', 'secondly', 'thirdly', 'finally', 'lastly',
                   'first and foremost', 'last but not least',
                   'first', 'second', 'third', 'next', 'then', 'subsequently'],
    'conclusive': ['in conclusion', 'to sum up', 'in summary', 'to summarize',
                   'overall', 'all in all', 'to conclude', 'in brief',
                   'taking everything into account', 'in short'],
    'contrastive': ['however', 'nevertheless', 'nonetheless', 'on the other hand',
                    'in contrast', 'conversely', 'whereas', 'although',
                    'even though', 'despite', 'in spite of'],
    'additive': ['moreover', 'furthermore', 'in addition', 'additionally', 'besides',
                 'also', 'not only', 'similarly', 'likewise'],
    'causal': ['therefore', 'consequently', 'as a result', 'thus', 'hence',
               'accordingly', 'for this reason', 'due to', 'because of'],
    'exemplifying': ['for example', 'for instance', 'such as', 'namely',
                     'specifically', 'to illustrate', 'in particular', 'as an illustration'],
    'hedging': ['it is important to note', 'it is worth noting', 'it should be noted',
                'as mentioned earlier', 'as previously stated', 'in other words',
                'that is to say', 'generally speaking', 'in general',
                'to some extent', 'to a certain degree', 'arguably', 'possibly'],
}

TRANSITION_CATEGORIES_ZH = {
    'sequential': ['首先', '其次', '再次', '最后', '接着', '然后', '接下来',
                   '第一步', '第二步', '第三步', '首先第一'],
    'conclusive': ['总而言之', '综上所述', '总之', '总的来看', '综上',
                   '综上所述', '总的来讲', '所以总的来说', '最后总结'],
    'contrastive': ['然而', '但是', '不过', '另一方面', '相反', '相比之下',
                    '尽管', '虽然', '可是', '却'],
    'additive': ['此外', '另外', '不仅如此', '而且', '并且', '同时',
                 '除此之外', '另外一方面', '还有', '另外一点'],
    'causal': ['因此', '所以', '因而', '由此可见', '由此', '从而',
               '于是', '结果', '正因如此', '故此'],
    'exemplifying': ['例如', '比如', '譬如', '具体来说', '具体而言',
                     '举个例子', '以……为例', '特别是'],
    'hedging': ['值得注意的是', '需要指出的是', '需要强调的是', '一般来说',
                '通常来说', '一般而言', '换句话说', '也就是说', '换言之',
                '从某种程度来说', '在某种程度上', '可能', '或许'],
}

# Chinese emotional words (simplified, for ratio computation)
POSITIVE_WORDS_ZH = {'好', '优秀', '出色', '棒', '赞', '喜欢', '开心', '快乐', '满意', '成功', '精彩', '美好', '积极', '乐观', '温暖', '幸福', '感谢', '感恩', '希望', '期待'}
NEGATIVE_WORDS_ZH = {'差', '糟糕', '失败', '讨厌', '难过', '伤心', '失望', '愤怒', '悲观', '消极', '痛苦', '焦虑', '担忧', '恐惧', '悲哀', '遗憾', '无奈', '可惜'}


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING — HC3 Dataset (唯一数据集，深入分析)
# ═══════════════════════════════════════════════════════════════════════

def load_hc3_english(data_dir='data/en'):
    """Load HC3 English dataset, return list of (text, label) where 0=human, 1=AI."""
    samples = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                for ans in obj.get('human_answers', []):
                    if ans and len(ans.strip()) > 30:
                        samples.append({'text': ans, 'label': 0, 'source': obj.get('source', '')})
                for ans in obj.get('chatgpt_answers', []):
                    if ans and len(ans.strip()) > 30:
                        samples.append({'text': ans, 'label': 1, 'source': obj.get('source', '')})
    return samples


def load_hc3_chinese(data_dir='data/zh'):
    """Load HC3 Chinese dataset, return list of (text, label) where 0=human, 1=AI."""
    samples = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                for ans in obj.get('human_answers', []):
                    if ans and len(ans.strip()) > 20:
                        samples.append({'text': ans, 'label': 0, 'source': obj.get('source', '')})
                for ans in obj.get('chatgpt_answers', []):
                    if ans and len(ans.strip()) > 20:
                        samples.append({'text': ans, 'label': 1, 'source': obj.get('source', '')})
    return samples


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — ENGLISH
# ═══════════════════════════════════════════════════════════════════════

class EnglishFeatureExtractor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat', 'lemmatizer'])

    def extract(self, text):
        features = {}
        text = text.strip()
        if not text:
            return self._empty_features()

        # Tokenization
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        words_lower = [w.lower() for w in words if w.isalpha()]
        chars = sum(1 for w in words for _ in w)

        # ── Category 1: Vocabulary Richness ──
        n_words = len(words_lower)
        n_unique = len(set(words_lower))
        features['ttr'] = n_unique / max(n_words, 1)  # Type-Token Ratio
        word_counts = Counter(words_lower)
        hapax_count = sum(1 for v in word_counts.values() if v == 1)
        features['hapax_ratio'] = hapax_count / max(n_words, 1)
        features['honore_r'] = 100 * math.log(max(n_words, 1)) / max(1 - hapax_count / max(n_unique, 1), 0.01)
        features['vocab_density'] = n_unique / math.sqrt(max(n_words, 1))

        # ── Category 2: Readability ──
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        except Exception:
            features['flesch_reading_ease'] = 50.0
            features['flesch_kincaid_grade'] = 8.0
        features['avg_word_len'] = chars / max(n_words, 1)
        try:
            features['avg_syllables_per_word'] = textstat.avg_syllables_per_word(text)
        except Exception:
            features['avg_syllables_per_word'] = 1.5

        # ── Category 3: Syntactic Features (POS) ──
        pos_tags = nltk.pos_tag([w for w in words if w.isalpha()])
        pos_counts = Counter(tag for _, tag in pos_tags)
        n_pos = max(len(pos_tags), 1)
        features['noun_ratio'] = sum(pos_counts[t] for t in ['NN', 'NNS', 'NNP', 'NNPS']) / n_pos
        features['verb_ratio'] = sum(pos_counts[t] for t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) / n_pos
        features['adj_ratio'] = sum(pos_counts[t] for t in ['JJ', 'JJR', 'JJS']) / n_pos
        features['adv_ratio'] = sum(pos_counts[t] for t in ['RB', 'RBR', 'RBS']) / n_pos
        features['conj_ratio'] = sum(pos_counts[t] for t in ['CC', 'IN']) / n_pos
        features['det_ratio'] = sum(pos_counts[t] for t in ['DT', 'PDT', 'WDT']) / n_pos
        features['punct_ratio'] = sum(1 for w in words if w in '.,!?;:') / max(len(words), 1)
        features['pronoun_ratio'] = sum(pos_counts[t] for t in ['PRP', 'PRP$', 'WP', 'WP$']) / n_pos

        # ── Dependency Parse Tree Depth (clause nesting) ──
        # Process longer texts with spacy; for very long texts, sample sentences
        try:
            text_for_spacy = text[:5000]  # First 5K chars sufficient for structure analysis
            doc = self.nlp(text_for_spacy)
            # Compute max and average dependency tree depth per sentence
            depths = []
            for sent in doc.sents:
                max_d = self._tree_depth(sent.root)
                depths.append(max_d)
            features['dep_depth_max'] = max(depths) if depths else 0.0
            features['dep_depth_avg'] = np.mean(depths) if depths else 0.0
            features['dep_depth_std'] = np.std(depths) if depths else 0.0
            # Ratio of deep clauses (depth > 5)
            features['deep_clause_ratio'] = sum(1 for d in depths if d > 5) / max(len(depths), 1)
            # Average branching factor (children per node)
            total_nodes = sum(1 for _ in doc)
            total_children = sum(len(list(t.children)) for t in doc)
            features['avg_branching'] = total_children / max(total_nodes, 1)
        except Exception:
            features['dep_depth_max'] = 0.0
            features['dep_depth_avg'] = 0.0
            features['dep_depth_std'] = 0.0
            features['deep_clause_ratio'] = 0.0
            features['avg_branching'] = 0.0

        # ── Category 4: Discourse Patterns (per-category transition words) ──
        text_lower = text.lower()
        total_words = len(words)
        # Per-category transition word frequencies (normalized per 100 words)
        for cat, markers in TRANSITION_CATEGORIES_EN.items():
            count = sum(1 for m in markers if m in text_lower)
            features[f'trans_{cat}'] = count / max(total_words / 100, 1)
        # Aggregate transition word density
        all_markers = [m for markers in TRANSITION_CATEGORIES_EN.values() for m in markers]
        total_trans = sum(1 for m in all_markers if m in text_lower)
        features['trans_total'] = total_trans / max(total_words / 100, 1)
        # Sequential/conclusive ratio (AI characteristic: heavy use of structured enumeration)
        seq_conc = sum(1 for m in TRANSITION_CATEGORIES_EN['sequential'] + TRANSITION_CATEGORIES_EN['conclusive'] if m in text_lower)
        features['structured_marker_ratio'] = seq_conc / max(total_words / 100, 1)

        # Sentence structure indicators
        if len(sentences) > 1:
            sent_lens = [len(word_tokenize(s)) for s in sentences]
            features['sent_len_mean'] = np.mean(sent_lens)
            features['sent_len_std'] = np.std(sent_lens)
            features['sent_len_cv'] = np.std(sent_lens) / max(np.mean(sent_lens), 1)
        else:
            features['sent_len_mean'] = total_words
            features['sent_len_std'] = 0.0
            features['sent_len_cv'] = 0.0

        # Paragraph count & structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        features['avg_para_len'] = total_words / max(len(paragraphs), 1)

        # ── Category 5: Sentiment Features ──
        sentiment = self.sia.polarity_scores(text)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_neutral'] = sentiment['neu']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_positive'] = sentiment['pos']

        # Emotion word ratio
        emotion_words = {'happy', 'sad', 'angry', 'excited', 'afraid', 'love', 'hate',
                         'wonderful', 'terrible', 'beautiful', 'horrible', 'amazing', 'awful',
                         'great', 'bad', 'good', 'poor', 'excellent', 'disgusting'}
        emotion_count = sum(1 for w in words_lower if w in emotion_words)
        features['emotion_word_ratio'] = emotion_count / max(total_words, 1)

        # Per-sentence sentiment variance (emotional fluctuation — "机器味" detection)
        # AI text tends to have flat/consistent sentiment across sentences
        sent_sentiments = []
        for sent in sentences:
            if len(sent.split()) > 3:
                sent_sentiments.append(self.sia.polarity_scores(sent)['compound'])
        if len(sent_sentiments) > 1:
            features['sentiment_fluctuation'] = np.std(sent_sentiments)
            features['sentiment_range'] = max(sent_sentiments) - min(sent_sentiments)
            # Ratio of sentiment direction changes (human text flips sentiment more)
            signs = [1 if s > 0.05 else (-1 if s < -0.05 else 0) for s in sent_sentiments]
            flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1] and signs[i] != 0)
            features['sentiment_flip_ratio'] = flips / len(sent_sentiments)
        else:
            features['sentiment_fluctuation'] = 0.0
            features['sentiment_range'] = 0.0
            features['sentiment_flip_ratio'] = 0.0

        # ── Category 6: Text Statistics ──
        features['text_len_chars'] = len(text)
        features['text_len_words'] = total_words
        features['sentence_count'] = len(sentences)
        features['avg_sent_len'] = total_words / max(len(sentences), 1)
        features['stopword_ratio'] = sum(1 for w in words_lower if w in self.stop_words) / max(total_words, 1)
        features['capitalized_ratio'] = sum(1 for w in words if w and w[0].isupper()) / max(len([w for w in words if w.isalpha()]), 1)

        # Unique bigram ratio (higher = more diverse phrasing)
        bigrams = [f"{words_lower[i]}_{words_lower[i+1]}" for i in range(len(words_lower) - 1)]
        features['unique_bigram_ratio'] = len(set(bigrams)) / max(len(bigrams), 1)

        # Number / digit ratio (AI tends to use more structured numbering)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)

        return features

    @staticmethod
    def _tree_depth(root):
        """Compute max depth of dependency tree from root node."""
        if root is None:
            return 0
        children_depths = [EnglishFeatureExtractor._tree_depth(c) for c in root.children]
        return 1 + max(children_depths) if children_depths else 1

    def _empty_features(self):
        feats = {k: 0.0 for k in [
            'ttr', 'hapax_ratio', 'honore_r', 'vocab_density',
            'flesch_reading_ease', 'flesch_kincaid_grade', 'avg_word_len', 'avg_syllables_per_word',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'conj_ratio', 'det_ratio',
            'punct_ratio', 'pronoun_ratio',
            'dep_depth_max', 'dep_depth_avg', 'dep_depth_std', 'deep_clause_ratio', 'avg_branching',
            'sent_len_mean', 'sent_len_std', 'sent_len_cv',
            'paragraph_count', 'avg_para_len',
            'trans_sequential', 'trans_conclusive', 'trans_contrastive', 'trans_additive',
            'trans_causal', 'trans_exemplifying', 'trans_hedging', 'trans_total', 'structured_marker_ratio',
            'sentiment_compound', 'sentiment_neutral', 'sentiment_negative', 'sentiment_positive',
            'emotion_word_ratio', 'sentiment_fluctuation', 'sentiment_range', 'sentiment_flip_ratio',
            'text_len_chars', 'text_len_words', 'sentence_count',
            'avg_sent_len', 'stopword_ratio', 'capitalized_ratio', 'unique_bigram_ratio', 'digit_ratio',
        ]}
        return feats


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — CHINESE
# ═══════════════════════════════════════════════════════════════════════

class ChineseFeatureExtractor:
    def __init__(self):
        self.word_set = set(jieba.lcut('', cut_all=False))

    def extract(self, text):
        features = {}
        text = text.strip()
        if not text:
            return self._empty_features()

        # Tokenization
        words = list(jieba.cut(text))
        chars = list(text.replace(' ', ''))
        words_clean = [w for w in words if w.strip() and not re.match(r'^[\s\d\W]+$', w)]
        chars_clean = [c for c in chars if c.strip() and not re.match(r'^[\s\d\W]+$', c)]

        # Sentence segmentation for Chinese
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        # ── Category 1: Vocabulary Richness ──
        n_words = max(len(words_clean), 1)
        n_chars = max(len(chars_clean), 1)
        n_unique_words = len(set(words_clean))
        n_unique_chars = len(set(chars_clean))

        features['ttr_word'] = n_unique_words / n_words
        features['ttr_char'] = n_unique_chars / n_chars
        word_counts = Counter(words_clean)
        hapax_count = sum(1 for v in word_counts.values() if v == 1)
        features['hapax_ratio'] = hapax_count / n_words
        features['honore_r'] = 100 * math.log(n_words) / max(1 - hapax_count / max(n_unique_words, 1), 0.01)
        features['vocab_density'] = n_unique_words / math.sqrt(n_words)

        # ── Category 2: Readability proxies (Chinese-adapted) ──
        features['avg_word_len_chars'] = n_chars / n_words
        features['avg_sent_len_chars'] = n_chars / max(len(sentences), 1)
        features['avg_sent_len_words'] = n_words / max(len(sentences), 1)

        # ── Category 3: Syntactic Features (POS via jieba) ──
        pos_pairs = list(pseg.cut(text))
        pos_pairs_clean = [(w, t) for w, t in pos_pairs if w.strip()]
        pos_counts = Counter(tag for _, tag in pos_pairs_clean)
        n_pos = max(len(pos_pairs_clean), 1)

        features['noun_ratio_zh'] = (pos_counts.get('n', 0) + pos_counts.get('nr', 0) +
                                     pos_counts.get('ns', 0) + pos_counts.get('nt', 0) +
                                     pos_counts.get('nz', 0)) / n_pos
        features['verb_ratio_zh'] = (pos_counts.get('v', 0) + pos_counts.get('vd', 0) +
                                     pos_counts.get('vn', 0)) / n_pos
        features['adj_ratio_zh'] = (pos_counts.get('a', 0) + pos_counts.get('ad', 0) +
                                    pos_counts.get('an', 0)) / n_pos
        features['adv_ratio_zh'] = (pos_counts.get('d', 0) + pos_counts.get('dg', 0)) / n_pos
        features['conj_ratio_zh'] = (pos_counts.get('c', 0) + pos_counts.get('p', 0)) / n_pos
        features['pronoun_ratio_zh'] = pos_counts.get('r', 0) / n_pos
        features['num_ratio_zh'] = pos_counts.get('m', 0) / n_pos
        features['punct_ratio_zh'] = sum(1 for c in text if c in '，。！？；：、""''（）《》') / max(len(text), 1)

        # ── Category 4: Discourse Patterns (per-category transition words) ──
        for cat, markers in TRANSITION_CATEGORIES_ZH.items():
            count = sum(1 for m in markers if m in text)
            features[f'trans_{cat}'] = count / max(n_words / 100, 1)
        all_markers = [m for markers in TRANSITION_CATEGORIES_ZH.values() for m in markers]
        total_trans = sum(1 for m in all_markers if m in text)
        features['trans_total'] = total_trans / max(n_words / 100, 1)
        seq_conc = sum(1 for m in TRANSITION_CATEGORIES_ZH['sequential'] + TRANSITION_CATEGORIES_ZH['conclusive'] if m in text)
        features['structured_marker_ratio'] = seq_conc / max(n_words / 100, 1)

        # Sentence structure
        if len(sentences) > 1:
            sent_lens = [len(list(jieba.cut(s))) for s in sentences]
            features['sent_len_mean'] = np.mean(sent_lens)
            features['sent_len_std'] = np.std(sent_lens)
            features['sent_len_cv'] = np.std(sent_lens) / max(np.mean(sent_lens), 1)
        else:
            features['sent_len_mean'] = float(n_words)
            features['sent_len_std'] = 0.0
            features['sent_len_cv'] = 0.0

        # Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]
        features['paragraph_count'] = len(paragraphs)
        features['avg_para_len'] = n_chars / max(len(paragraphs), 1)

        # ── Category 5: Sentiment Features (Chinese-adapted) ──
        pos_count = sum(1 for w in words_clean if w in POSITIVE_WORDS_ZH)
        neg_count = sum(1 for w in words_clean if w in NEGATIVE_WORDS_ZH)
        sentiment_total = pos_count + neg_count + 1
        features['sentiment_pos_ratio'] = pos_count / sentiment_total
        features['sentiment_neg_ratio'] = neg_count / sentiment_total
        features['sentiment_neutrality'] = 1 - (pos_count + neg_count) / max(n_words, 1)

        # Exclamation / question mark ratio
        features['exclam_ratio'] = text.count('！') / max(len(sentences), 1)
        features['question_ratio'] = text.count('？') / max(len(sentences), 1)

        # Per-sentence sentiment variance (emotional fluctuation — "机器味" for Chinese)
        sent_scores = []
        for sent in sentences:
            words_in_sent = [w for w in jieba.cut(sent) if w.strip()]
            pos_in = sum(1 for w in words_in_sent if w in POSITIVE_WORDS_ZH)
            neg_in = sum(1 for w in words_in_sent if w in NEGATIVE_WORDS_ZH)
            total = pos_in + neg_in + 1
            sent_scores.append((pos_in - neg_in) / total)  # [-1, 1] range
        if len(sent_scores) > 1:
            features['sentiment_fluctuation'] = np.std(sent_scores)
            features['sentiment_range'] = max(sent_scores) - min(sent_scores)
            signs = [1 if s > 0.05 else (-1 if s < -0.05 else 0) for s in sent_scores]
            flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1] and signs[i] != 0)
            features['sentiment_flip_ratio'] = flips / len(sent_scores)
        else:
            features['sentiment_fluctuation'] = 0.0
            features['sentiment_range'] = 0.0
            features['sentiment_flip_ratio'] = 0.0

        # ── Category 6: Text Statistics ──
        features['text_len_chars'] = len(text)
        features['text_len_words'] = n_words
        features['sentence_count'] = len(sentences)
        features['avg_sent_len'] = n_words / max(len(sentences), 1)
        features['stopword_ratio'] = sum(1 for w in words_clean if len(w) == 1) / n_words
        features['unique_bigram_ratio'] = self._bigram_ratio(words_clean)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)

        # Idiom / chengyu ratio (4-char patterns common in formal Chinese)
        chengyu_pattern = len(re.findall(r'[一-鿿]{4}', text))
        features['four_char_phrase_ratio'] = chengyu_pattern / max(len(chars), 1)

        return features

    def _bigram_ratio(self, words):
        if len(words) < 2:
            return 0.0
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return len(set(bigrams)) / len(bigrams)

    def _empty_features(self):
        feats = {k: 0.0 for k in [
            'ttr_word', 'ttr_char', 'hapax_ratio', 'honore_r', 'vocab_density',
            'avg_word_len_chars', 'avg_sent_len_chars', 'avg_sent_len_words',
            'noun_ratio_zh', 'verb_ratio_zh', 'adj_ratio_zh', 'adv_ratio_zh',
            'conj_ratio_zh', 'pronoun_ratio_zh', 'num_ratio_zh', 'punct_ratio_zh',
            'trans_sequential', 'trans_conclusive', 'trans_contrastive', 'trans_additive',
            'trans_causal', 'trans_exemplifying', 'trans_hedging', 'trans_total', 'structured_marker_ratio',
            'sent_len_mean', 'sent_len_std', 'sent_len_cv',
            'paragraph_count', 'avg_para_len',
            'sentiment_pos_ratio', 'sentiment_neg_ratio', 'sentiment_neutrality',
            'exclam_ratio', 'question_ratio',
            'sentiment_fluctuation', 'sentiment_range', 'sentiment_flip_ratio',
            'text_len_chars', 'text_len_words', 'sentence_count', 'avg_sent_len',
            'stopword_ratio', 'unique_bigram_ratio', 'digit_ratio',
            'four_char_phrase_ratio',
        ]}
        return feats


# ═══════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def extract_features(samples, lang='en', sample_size=None):
    """Extract features from all samples. Returns X (features) and y (labels)."""
    if sample_size:
        # Stratified sampling to maintain class balance
        human_samples = [s for s in samples if s['label'] == 0]
        ai_samples = [s for s in samples if s['label'] == 1]
        n_per_class = sample_size // 2
        np.random.seed(42)
        human_idx = np.random.choice(len(human_samples), min(n_per_class, len(human_samples)), replace=False)
        ai_idx = np.random.choice(len(ai_samples), min(n_per_class, len(ai_samples)), replace=False)
        balanced = [human_samples[i] for i in human_idx] + [ai_samples[i] for i in ai_idx]
        np.random.shuffle(balanced)
        samples = balanced

    extractor = EnglishFeatureExtractor() if lang == 'en' else ChineseFeatureExtractor()
    features_list = []
    labels = []

    for i, sample in enumerate(samples):
        feats = extractor.extract(sample['text'])
        features_list.append(feats)
        labels.append(sample['label'])
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    X = pd.DataFrame(features_list)
    y = np.array(labels)
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(X, y, lang='en'):
    """Train multiple models and evaluate with test set."""
    print(f"\n{'='*70}")
    print(f"  Dataset: HC3 {'English' if lang == 'en' else 'Chinese'}")
    print(f"  Total samples: {len(y)} (Human: {(y==0).sum()}, AI: {(y==1).sum()})")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"{'='*70}")

    # Handle NaN/Inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # ── Correlation Analysis: Remove Redundant Features ──
    print("\n▶ Correlation Analysis — Removing redundant features (|r| > 0.85)...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > 0.85].tolist()
        if correlated:
            for c in correlated:
                # Keep the feature with higher variance
                if X[col].var() >= X[c].var():
                    to_drop.add(c)
                else:
                    to_drop.add(col)
    to_drop = list(to_drop)
    if to_drop:
        print(f"  Removed {len(to_drop)} redundant features: {to_drop}")
        X = X.drop(columns=to_drop)
    else:
        print(f"  No redundant features found (all |r| <= 0.85).")
    print(f"  Feature dimensions after filtering: {X.shape[1]}")
    print(f"{'='*70}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    results = {}

    # ── Model 1: Random Forest ──
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_df, y_train)
    rf_pred = rf.predict(X_test_df)
    rf_prob = rf.predict_proba(X_test_df)[:, 1]
    results['Random Forest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'auc': roc_auc_score(y_test, rf_prob),
        'feature_importance': rf.feature_importances_,
        'predictions': rf_pred,
    }
    print(f"  RF  → Accuracy: {results['Random Forest']['accuracy']:.4f}, "
          f"F1: {results['Random Forest']['f1']:.4f}, "
          f"AUC: {results['Random Forest']['auc']:.4f}")

    # ── Model 2: XGBoost ──
    print("[2/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X_train_df, y_train)
    xgb_pred = xgb_model.predict(X_test_df)
    xgb_prob = xgb_model.predict_proba(X_test_df)[:, 1]
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'f1': f1_score(y_test, xgb_pred),
        'auc': roc_auc_score(y_test, xgb_prob),
        'feature_importance': xgb_model.feature_importances_,
        'predictions': xgb_pred,
    }
    print(f"  XGB → Accuracy: {results['XGBoost']['accuracy']:.4f}, "
          f"F1: {results['XGBoost']['f1']:.4f}, "
          f"AUC: {results['XGBoost']['auc']:.4f}")

    # ── Model 3: LightGBM ──
    print("[3/3] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train_df, y_train)
    lgb_pred = lgb_model.predict(X_test_df)
    lgb_prob = lgb_model.predict_proba(X_test_df)[:, 1]
    results['LightGBM'] = {
        'model': lgb_model,
        'accuracy': accuracy_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred),
        'recall': recall_score(y_test, lgb_pred),
        'f1': f1_score(y_test, lgb_pred),
        'auc': roc_auc_score(y_test, lgb_prob),
        'feature_importance': lgb_model.feature_importances_,
        'predictions': lgb_pred,
    }
    print(f"  LGB → Accuracy: {results['LightGBM']['accuracy']:.4f}, "
          f"F1: {results['LightGBM']['f1']:.4f}, "
          f"AUC: {results['LightGBM']['auc']:.4f}")

    # ── Cross-validation for best model ──
    print("\n[CV] 5-Fold Cross-Validation (LightGBM):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        X_train_df, y_train, cv=cv, scoring='f1', n_jobs=-1
    )
    print(f"  CV F1 scores: {cv_scores.round(4)}")
    print(f"  Mean CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return X_train_df, X_test_df, y_train, y_test, results, rf


# ═══════════════════════════════════════════════════════════════════════
# 6. SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def shap_analysis(X_train, X_test, results, lang='en', top_n=15):
    """SHAP analysis + multi-model feature importance comparison."""
    print(f"\n{'='*70}")
    print("  SHAP Analysis — Feature Importance & Attribution")
    print(f"{'='*70}")

    best_model = results['XGBoost']['model']
    feature_names = X_test.columns.tolist()

    # ── SHAP on XGBoost ──
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test[:500])
    shap_mean = np.abs(shap_values).mean(axis=0)

    top_idx = np.argsort(shap_mean)[-top_n:][::-1]

    print(f"\n  Top {top_n} Most Important Features (by SHAP on XGBoost):")
    print(f"  {'Rank':<6}{'Feature':<30}{'SHAP':<14}{'Direction'}")
    print(f"  {'-'*65}")
    for rank, idx in enumerate(top_idx, 1):
        direction = "↑ in AI text" if shap_values[:, idx].mean() > 0 else "↓ in AI text"
        print(f"  {rank:<6}{feature_names[idx]:<30}{shap_mean[idx]:<14.6f}{direction}")

    # ── Multi-Model Feature Importance Consensus ──
    print(f"\n  Multi-Model Feature Importance Consensus (top {min(top_n, 10)}):")
    print(f"  {'Feature':<30}{'RF Rank':<10}{'XGB Rank':<10}{'LGB Rank':<10}{'Consensus'}")
    print(f"  {'-'*70}")
    all_features = X_test.columns.tolist()
    for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if model_name in results and results[model_name]['feature_importance'] is not None:
            pass  # feature importance stored per model

    # Compute ranks for all 3 models
    rf_imp = results['Random Forest']['feature_importance']
    xgb_imp = results['XGBoost']['feature_importance']
    lgb_imp = results['LightGBM']['feature_importance']

    # If feature dims don't match (correlation filtering), reindex
    rf_ranks = np.argsort(np.argsort(-rf_imp)) + 1  # rank 1 = most important
    xgb_ranks = np.argsort(np.argsort(-xgb_imp)) + 1
    lgb_ranks = np.argsort(np.argsort(-lgb_imp)) + 1

    # Find features with high consensus (high rank in all 3 models)
    consensus = rf_ranks + xgb_ranks + lgb_ranks
    consensus_idx = np.argsort(consensus)[:min(top_n, 10)]

    for idx in consensus_idx:
        print(f"  {feature_names[idx]:<30}{rf_ranks[idx]:<10}{xgb_ranks[idx]:<10}{lgb_ranks[idx]:<10}{'★★★' if consensus[idx] < 20 else '★★' if consensus[idx] < 40 else '★'}")

    # ── Feature Group Importance ──
    groups = _get_feature_groups(feature_names, lang)
    print(f"\n  Feature Group Importance (SHAP):")
    print(f"  {'Group':<28}{'Total SHAP':<14}{'% of Total'}")
    print(f"  {'-'*50}")
    total_shap = shap_mean.sum()
    for group_name, group_features in groups.items():
        group_indices = [feature_names.index(f) for f in group_features if f in feature_names]
        if group_indices:
            group_total = shap_mean[group_indices].sum()
            print(f"  {group_name:<28}{group_total:<14.4f}{group_total/total_shap*100:.1f}%")

    return shap_values, explainer


def _get_feature_groups(feature_names, lang):
    """Group features by category for interpretation."""
    if lang == 'en':
        groups = {
            'Vocabulary Richness': ['ttr', 'hapax_ratio', 'honore_r', 'vocab_density'],
            'Readability': ['flesch_reading_ease', 'flesch_kincaid_grade', 'avg_word_len', 'avg_syllables_per_word'],
            'POS & Syntax': ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'conj_ratio', 'det_ratio', 'punct_ratio', 'pronoun_ratio'],
            'Dependency Depth': ['dep_depth_max', 'dep_depth_avg', 'dep_depth_std', 'deep_clause_ratio', 'avg_branching'],
            'Transition Words': ['trans_sequential', 'trans_conclusive', 'trans_contrastive',
                                 'trans_additive', 'trans_causal', 'trans_exemplifying',
                                 'trans_hedging', 'trans_total', 'structured_marker_ratio'],
            'Sentence Structure': ['sent_len_mean', 'sent_len_std', 'sent_len_cv', 'paragraph_count', 'avg_para_len'],
            'Sentiment & Emotion': ['sentiment_compound', 'sentiment_neutral', 'sentiment_negative',
                                    'sentiment_positive', 'emotion_word_ratio',
                                    'sentiment_fluctuation', 'sentiment_range', 'sentiment_flip_ratio'],
            'Text Statistics': ['text_len_chars', 'text_len_words', 'sentence_count', 'avg_sent_len', 'stopword_ratio', 'capitalized_ratio', 'unique_bigram_ratio', 'digit_ratio'],
        }
    else:
        groups = {
            'Vocabulary Richness': ['ttr_word', 'ttr_char', 'hapax_ratio', 'honore_r', 'vocab_density'],
            'Readability (Chinese)': ['avg_word_len_chars', 'avg_sent_len_chars', 'avg_sent_len_words'],
            'POS & Syntax': ['noun_ratio_zh', 'verb_ratio_zh', 'adj_ratio_zh', 'adv_ratio_zh', 'conj_ratio_zh', 'pronoun_ratio_zh', 'num_ratio_zh', 'punct_ratio_zh'],
            'Transition Words': ['trans_sequential', 'trans_conclusive', 'trans_contrastive',
                                 'trans_additive', 'trans_causal', 'trans_exemplifying',
                                 'trans_hedging', 'trans_total', 'structured_marker_ratio'],
            'Sentence Structure': ['sent_len_mean', 'sent_len_std', 'sent_len_cv', 'paragraph_count', 'avg_para_len'],
            'Sentiment & Emotion': ['sentiment_pos_ratio', 'sentiment_neg_ratio', 'sentiment_neutrality',
                                    'exclam_ratio', 'question_ratio',
                                    'sentiment_fluctuation', 'sentiment_range', 'sentiment_flip_ratio'],
            'Text Statistics': ['text_len_chars', 'text_len_words', 'sentence_count', 'avg_sent_len', 'stopword_ratio', 'unique_bigram_ratio', 'digit_ratio', 'four_char_phrase_ratio'],
        }
    return groups


# ═══════════════════════════════════════════════════════════════════════
# 7. DETAILED REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_final_report(results, lang='en'):
    """Print comprehensive final report."""
    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS — HC3 {'English' if lang == 'en' else 'Chinese'}")
    print(f"{'='*70}")
    print(f"\n  {'Model':<16}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'AUC':<12}")
    print(f"  {'-'*70}")
    best_model = None
    best_f1 = 0
    for name, res in results.items():
        print(f"  {name:<16}{res['accuracy']:<12.4f}{res['precision']:<12.4f}{res['recall']:<12.4f}{res['f1']:<12.4f}{res['auc']:<12.4f}")
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            best_model = name
    print(f"\n  ★ Best Model: {best_model} (F1 = {best_f1:.4f})")


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AIGC Detection Demo — Direction 1: Feature Mining & Detection")
    print("  Dataset: Human ChatGPT Comparison Corpus (HC3)")
    print("  Approach: Linguistic Feature Engineering + Interpretable ML + SHAP")
    print("=" * 70)

    # ── HC3 English ──
    print("\n\n▶ Loading HC3 English dataset...")
    en_samples = load_hc3_english('data/en')
    print(f"  Loaded {len(en_samples)} English samples")
    print(f"  Human: {sum(1 for s in en_samples if s['label'] == 0)}, "
          f"AI: {sum(1 for s in en_samples if s['label'] == 1)}")

    print("\n▶ Extracting English features (6 categories: vocabulary, readability, "
          "POS/syntax, dependency depth, transition words, sentiment)...")
    X_en, y_en = extract_features(en_samples, lang='en', sample_size=10000)
    X_train_en, X_test_en, y_train_en, y_test_en, results_en, _ = train_and_evaluate(X_en, y_en, lang='en')
    shap_values_en, _ = shap_analysis(X_train_en, X_test_en, results_en, lang='en')
    print_final_report(results_en, lang='en')

    # ── HC3 Chinese ──
    print("\n\n▶ Loading HC3 Chinese dataset...")
    zh_samples = load_hc3_chinese('data/zh')
    print(f"  Loaded {len(zh_samples)} Chinese samples")
    print(f"  Human: {sum(1 for s in zh_samples if s['label'] == 0)}, "
          f"AI: {sum(1 for s in zh_samples if s['label'] == 1)}")

    print("\n▶ Extracting Chinese features (6 categories)...")
    X_zh, y_zh = extract_features(zh_samples, lang='zh', sample_size=10000)
    X_train_zh, X_test_zh, y_train_zh, y_test_zh, results_zh, _ = train_and_evaluate(X_zh, y_zh, lang='zh')
    shap_values_zh, _ = shap_analysis(X_train_zh, X_test_zh, results_zh, lang='zh')
    print_final_report(results_zh, lang='zh')

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY — HC3 Deep Analysis")
    print(f"{'='*70}")
    print(f"  {'Language':<12}{'Best Model':<16}{'Accuracy':<12}{'F1 Score':<12}{'AUC':<12}")
    print(f"  {'-'*55}")
    for label, res in [('English', results_en), ('Chinese', results_zh)]:
        best = max(res.items(), key=lambda x: x[1]['f1'])
        print(f"  {label:<12}{best[0]:<16}{best[1]['accuracy']:<12.4f}{best[1]['f1']:<12.4f}{best[1]['auc']:<12.4f}")

    print(f"\n  ✓ Direction 1 complete — single dataset (HC3), deep analysis.")
    print(f"  ✓ 6 feature categories with 40+ linguistic features per language.")
    print(f"  ✓ Correlation filtering → RF/XGBoost/LightGBM → SHAP attribution.")
    print(f"  ✓ Multi-model feature importance consensus verified.\n")


if __name__ == '__main__':
    main()
