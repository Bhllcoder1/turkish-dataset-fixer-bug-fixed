"""
BellaTurca veri seti filtreleme pipeline'ı.

turkish-dataset-fixer (https://github.com/serda-dev/turkish-dataset-fixer) projesindeki
filtre kurallarını uygular. Normalizasyon aşamasındaki boşluk silme bug'ı düzeltilmiştir.

Test veri seti: turkish-nlp-suite/BellaTurca (OzenliDerlem subset)
https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca

Filtreler: normalization, heuristic, language, dedup, soft-score
Atlanan: KenLM (derlenmiş binary + eğitilmiş 5-gram model gerektirir)
"""

import re
import unicodedata
import xxhash
import json
import argparse
import os
from collections import Counter
from datasets import load_dataset

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH       = 50
MAX_TEXT_LENGTH       = 500_000
MIN_WORD_COUNT        = 5
MIN_ALPHA_RATIO       = 0.40
MAX_DIGIT_RATIO       = 0.50
MAX_PUNCTUATION_RATIO = 0.40
MAX_URL_COUNT         = 20
MAX_EMAIL_COUNT       = 10
MAX_HTML_TAG_COUNT    = 10
MAX_REPEATED_LINE_RATIO = 0.70
MAX_REPEATED_WORD_FRAC  = 0.50
MIN_UNIQUE_TOKEN_RATIO  = 0.10
MAX_AVG_WORD_LENGTH   = 45.0
MIN_AVG_WORD_LENGTH   = 1.5
MAX_BOILERPLATE_SCORE = 5

LANG_MIN_CONFIDENCE   = 0.3
LANG_HIGH_CONFIDENCE  = 0.5
LANG_REJECT_CONFIDENCE= 0.7
MIN_TURKISH_CHAR_RATIO = 0.015
MIN_STOPWORD_COVERAGE  = 0.005

SOFT_SCORE_THRESHOLD  = 0.25
WEIGHT_LANG           = 0.50
WEIGHT_HEURISTIC      = 0.50

TURKISH_CHARS = set("çğıİöşüÇĞÖŞÜ")
TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "da", "de", "ile", "için", "mi", "mı", "mu", "mü",
    "ne", "ben", "sen", "o", "biz", "siz", "onlar", "var", "yok", "gibi",
    "daha", "çok", "en", "ki", "ama", "veya", "ya", "bile", "olan",
    "şu", "hem", "hiç", "kadar", "sonra", "önce", "artık", "nasıl",
}

BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"çerez\s+politika",
        r"gizlilik\s+politika",
        r"kvkk",
        r"cookie\s+policy",
        r"privacy\s+policy",
        r"all\s+rights\s+reserved",
        r"tüm\s+hakları\s+saklıdır",
        r"terms\s+of\s+service",
        r"kullanım\s+koşulları",
    ]
]

URL_RE       = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE     = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_RE  = re.compile(r"<[^>]+>")

# ─── STAGE 1: TEXT NORMALIZATION ───────────────────────────────────────────────
#
# BUG FIX: Orijinal regex boşluk karakterlerini (U+0020) de siliyordu:
#   HATALI:  re.compile(r"[^\S\r\n\t\u00A0-\uFFFF]|[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
#   Bu desen "Bu bir test" → "Bubirtest" yapıyordu.
#   Tüm kelimeler birleştiği için too_few_words, avg_word_too_long, high_repeated_word
#   filtreleri tetiklenip %100 metin reddediliyordu.
#
# DÜZELTME: Sadece gerçek kontrol karakterlerini hedefleyen regex:
_ctrl_re   = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_zw_re     = re.compile(r"[\u200B-\u200D\uFEFF\u00AD]")
_uspace_re = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")
_blanks_re = re.compile(r"\n{4,}")
_spaces_re = re.compile(r" {4,}")

def normalize_text(text: str) -> str:
    text = _ctrl_re.sub("", text)
    text = _zw_re.sub("", text)
    text = _uspace_re.sub(" ", text)
    text = _blanks_re.sub("\n\n", text)
    text = _spaces_re.sub(" ", text)
    return text.strip()

# ─── STAGE 2: HEURISTIC FEATURES & FILTERS ─────────────────────────────────────
def compute_features(text: str) -> dict:
    chars  = len(text)
    words  = text.split()
    lines  = text.splitlines()
    alpha  = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    punct  = sum(1 for c in text if unicodedata.category(c).startswith("P"))
    ws     = sum(1 for c in text if c.isspace())
    tr_chars = sum(1 for c in text if c in TURKISH_CHARS)
    urls   = len(URL_RE.findall(text))
    emails = len(EMAIL_RE.findall(text))
    html_tags = len(HTML_TAG_RE.findall(text))
    word_counts = Counter(w.lower() for w in words)
    uniq_ratio = len(word_counts) / max(len(words), 1)
    most_common_word_frac = (word_counts.most_common(1)[0][1] / max(len(words), 1)) if words else 0
    line_counts = Counter(lines)
    repeated_line_ratio = sum(v - 1 for v in line_counts.values() if v > 1) / max(len(lines), 1)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1) if words else 0
    stopword_hits = sum(1 for w in words if w.lower() in TURKISH_STOPWORDS)
    stopword_coverage = stopword_hits / max(len(words), 1)
    boilerplate_score = sum(1 for p in BOILERPLATE_PATTERNS if p.search(text))
    return dict(
        char_count=chars,
        word_count=len(words),
        line_count=len(lines),
        alpha_ratio=alpha / max(chars, 1),
        digit_ratio=digits / max(chars, 1),
        punct_ratio=punct / max(chars, 1),
        whitespace_ratio=ws / max(chars, 1),
        turkish_char_ratio=tr_chars / max(chars, 1),
        url_count=urls,
        email_count=emails,
        html_tag_count=html_tags,
        unique_token_ratio=uniq_ratio,
        repeated_line_ratio=repeated_line_ratio,
        repeated_word_frac=most_common_word_frac,
        avg_word_length=avg_word_len,
        stopword_coverage=stopword_coverage,
        boilerplate_score=boilerplate_score,
    )

def apply_heuristic_filters(f: dict) -> list[str]:
    reasons = []
    if f["char_count"] < MIN_TEXT_LENGTH:
        reasons.append("too_short")
    if f["char_count"] > MAX_TEXT_LENGTH:
        reasons.append("too_long")
    if f["word_count"] < MIN_WORD_COUNT:
        reasons.append("too_few_words")
    if f["alpha_ratio"] < MIN_ALPHA_RATIO:
        reasons.append("low_alpha_ratio")
    if f["digit_ratio"] > MAX_DIGIT_RATIO:
        reasons.append("high_digit_ratio")
    if f["punct_ratio"] > MAX_PUNCTUATION_RATIO:
        reasons.append("high_punct_ratio")
    if f["url_count"] > MAX_URL_COUNT:
        reasons.append("too_many_urls")
    if f["email_count"] > MAX_EMAIL_COUNT:
        reasons.append("too_many_emails")
    if f["html_tag_count"] > MAX_HTML_TAG_COUNT:
        reasons.append("too_many_html_tags")
    if f["repeated_line_ratio"] > MAX_REPEATED_LINE_RATIO:
        reasons.append("high_repeated_lines")
    if f["repeated_word_frac"] > MAX_REPEATED_WORD_FRAC:
        reasons.append("high_repeated_word")
    if f["unique_token_ratio"] < MIN_UNIQUE_TOKEN_RATIO:
        reasons.append("low_token_diversity")
    if f["avg_word_length"] > MAX_AVG_WORD_LENGTH:
        reasons.append("avg_word_too_long")
    if f["avg_word_length"] < MIN_AVG_WORD_LENGTH:
        reasons.append("avg_word_too_short")
    if f["boilerplate_score"] > MAX_BOILERPLATE_SCORE:
        reasons.append("boilerplate")
    return reasons

# ─── STAGE 3: LANGUAGE VALIDATION ──────────────────────────────────────────────
try:
    from ftlangdetect import detect as ft_detect
    HAS_FASTTEXT = True
except Exception:
    HAS_FASTTEXT = False

def detect_language(text: str) -> tuple[str, float]:
    sample = text[:500].replace("\n", " ")
    if HAS_FASTTEXT:
        try:
            r = ft_detect(sample, low_memory=True)
            return r["lang"], r["score"]
        except Exception:
            pass
    try:
        from langdetect import detect_langs
        results = detect_langs(sample)
        for r in results:
            if r.lang == "tr":
                return "tr", r.prob
        return results[0].lang, results[0].prob
    except Exception:
        return "unknown", 0.0

def validate_language(text: str, features: dict) -> tuple[bool, str, float]:
    lang, conf = detect_language(text)
    tr_char_ok = features["turkish_char_ratio"] >= MIN_TURKISH_CHAR_RATIO
    sw_ok      = features["stopword_coverage"] >= MIN_STOPWORD_COVERAGE
    if lang == "tr" and conf >= LANG_HIGH_CONFIDENCE:
        return True, lang, conf
    if lang == "tr" and conf >= LANG_MIN_CONFIDENCE and (tr_char_ok or sw_ok):
        return True, lang, conf
    if tr_char_ok and sw_ok:
        return True, "tr_signal", conf
    if lang != "tr" and conf >= LANG_REJECT_CONFIDENCE and not tr_char_ok and not sw_ok:
        return False, lang, conf
    if tr_char_ok or sw_ok:
        return True, "tr_borderline", conf
    return False, lang, conf

# ─── STAGE 4: EXACT DEDUPLICATION ──────────────────────────────────────────────
seen_hashes: set[int] = set()

def is_duplicate(text: str) -> bool:
    h = xxhash.xxh64(text.encode("utf-8")).intdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False

# ─── STAGE 5: SOFT QUALITY SCORE (without KenLM) ───────────────────────────────
def compute_quality_score(features: dict, lang_conf: float, lang: str) -> float:
    lang_score = lang_conf if lang.startswith("tr") else max(0.0, lang_conf - 0.3)
    lang_score = min(1.0, lang_score)

    alpha_score  = min(1.0, features["alpha_ratio"] / 1.0)
    div_score    = min(1.0, features["unique_token_ratio"] / 0.5)
    sw_score     = min(1.0, features["stopword_coverage"] / 0.05)
    tr_char_score= min(1.0, features["turkish_char_ratio"] / 0.03)
    heuristic_score = (
        0.30 * alpha_score +
        0.20 * div_score   +
        0.30 * sw_score    +
        0.20 * tr_char_score
    )
    score = WEIGHT_LANG * lang_score + WEIGHT_HEURISTIC * heuristic_score
    return score

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_pipeline(num_rows: int = 2000, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] {num_rows} satır indiriliyor (BellaTurca/OzenliDerlem)...")
    ds = load_dataset(
        "turkish-nlp-suite/BellaTurca",
        name="OzenliDerlem",
        split="train",
        streaming=True,
    )
    raw_texts = []
    for item in ds.take(num_rows):
        t = item.get("text", "")
        if t:
            raw_texts.append(t)
    print(f"    İndirilen: {len(raw_texts)} metin")

    stats = {
        "total_raw": len(raw_texts),
        "rejected_normalization": 0,
        "rejected_heuristic": Counter(),
        "rejected_language": 0,
        "rejected_duplicate": 0,
        "rejected_soft_score": 0,
        "passed": 0,
    }
    kept = []

    print("[2/5] Filtreler uygulanıyor...")
    for i, raw in enumerate(raw_texts):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(raw_texts)}...")

        text = normalize_text(raw)
        if not text:
            stats["rejected_normalization"] += 1
            continue

        features = compute_features(text)
        reject_reasons = apply_heuristic_filters(features)
        if reject_reasons:
            for r in reject_reasons:
                stats["rejected_heuristic"][r] += 1
            continue

        lang_ok, lang, conf = validate_language(text, features)
        if not lang_ok:
            stats["rejected_language"] += 1
            continue

        if is_duplicate(text):
            stats["rejected_duplicate"] += 1
            continue

        score = compute_quality_score(features, conf, lang)
        if score < SOFT_SCORE_THRESHOLD:
            stats["rejected_soft_score"] += 1
            continue

        kept.append({"text": text, "lang": lang, "lang_conf": round(conf, 4), "quality_score": round(score, 4)})
        stats["passed"] += 1

    # Kaydet
    out_path = os.path.join(output_dir, "bellaturca_filtered.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[3/5] {len(kept)} satır kaydedildi → {out_path}")

    # Rapor
    total = stats["total_raw"]
    passed = stats["passed"]
    rejected = total - passed

    print(f"\n{'='*50}")
    print(f"SONUÇ")
    print(f"{'='*50}")
    print(f"  Toplam        : {total}")
    print(f"  Geçen         : {passed} ({100*passed/total:.1f}%)")
    print(f"  Reddedilen    : {rejected} ({100*rejected/total:.1f}%)")
    print(f"    Çok kısa      : {stats['rejected_heuristic'].get('too_short', 0)}")
    print(f"    Dil doğrulama  : {stats['rejected_language']}")
    print(f"    Tam kopya      : {stats['rejected_duplicate']}")
    print(f"    Soft score     : {stats['rejected_soft_score']}")
    if stats["rejected_heuristic"]:
        other = {k: v for k, v in stats["rejected_heuristic"].items() if k != "too_short"}
        if other:
            print(f"    Diğer heuristik: {dict(other)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BellaTurca veri seti filtreleme")
    parser.add_argument("--rows", type=int, default=2000, help="İndirilecek satır sayısı")
    parser.add_argument("--output", type=str, default="output", help="Çıktı dizini")
    args = parser.parse_args()
    run_pipeline(num_rows=args.rows, output_dir=args.output)
