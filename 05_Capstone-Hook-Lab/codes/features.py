"""Feature engineering for the Longevity Hook Lab (Part A).

Used by 03_model.ipynb and by hook_rewriter.py (Part B), so both build
features the same way:

    from features import add_features, build_preprocessor
    X = add_features(shorts)          # needs: title, published_at, duration_sec
    pre = build_preprocessor()        # ColumnTransformer for a sklearn Pipeline

All rules are simple keyword / regex checks on the title, so they're easy to read
and explain. publish_hour / publish_weekday are in UTC (the API's timezone).
"""

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------

# Checked in this order; the first match wins.
HOOK_STYLE_RULES = {
    "how_to": r"\bhow to\b|\bhow i\b|\bways? to\b|\btips?\b|\bstep[s]?\b",
    "list": r"^\s*(top\s+)?\d+\b|\b\d+\s+(things|ways|foods|signs|reasons|habits|tips|rules|mistakes|steps|pillars|questions|exercises|supplements)\b",
    "question": r"\?|^\s*(why|what|how|when|which|who|does|do|is|are|can|should|will|would)\b",
    "warning": r"\b(stop|avoid|never|don'?t|warning|mistakes?|worst|danger(ous)?|toxic|lies|lying|hurts?|harm(ful)?|mislead(ing)?|risk[sy]?|killing|ruin(ing|s)?)\b",
    "story": r"\b(i|my|me|i'm|i've|we|our)\b|\bstory\b|\bwhen (he|she|i)\b",
}

# Checked in this order; the first match wins. Prefix matches (\bfast also hits fasting).
TOPIC_RULES = {
    "sleep": r"\b(sleep|insomnia|melatonin|nap|circadian|bedtime|dream|tired|fatigue)",
    "diet_fasting": r"\b(fast|diet|food|eat|ate|meal|breakfast|snack|glucose|sugar|carb|keto|protein|insulin|calorie|weight|obes|glp|ozempic|fruit|vegg?ie|meat|cacao|chocolate|coffee|alcohol)",
    "supplements": r"\b(supplement|vitamin|magnesium|creatine|omega|zinc|fish oil|nad|collagen|probiotic|d3|electrolyte|multivitamin)",
    "brain": r"\b(brain|alzheimer|dementia|memory|focus|anxiety|depress|mental|dopamine|cognit|adhd|stress|mood|neuro)",
    "exercise": r"\b(exercise|workout|muscle|strength|cardio|zone 2|vo2|train|run|walk|lift|steps|gym|fitness|sprint)",
    "aging": r"\b(aging|ageing|age|longevity|lifespan|healthspan|older|live longer|die|death|mortality|young)",
}

# Hype / strong-claim words; claim_intensity = how many times they appear.
HYPE_WORDS = [
    "cure", "cures", "reverse", "reverses", "secret", "secrets", "never", "always", "#1",
    "number one", "miracle", "best", "worst", "shocking", "hack", "hacks", "instantly",
    "guaranteed", "proven", "ultimate", "destroy", "destroys", "toxic", "life-changing",
    "game changer", "must", "nobody", "everyone", "hidden", "truth", "lies", "deadly",
]
HYPE_PATTERN = r"(?<!\w)(" + "|".join(re.escape(w) for w in HYPE_WORDS) + r")(?!\w)"

# Person names: the channel's own creators plus common "guest name" patterns.
CREATOR_NAMES = [
    "Andrew Huberman", "Huberman", "Peter Attia", "Attia", "Rhonda Patrick", "Mark Hyman",
    "Hyman", "Eric Berg", "Berg", "David Perlmutter", "Perlmutter", "Jessie Inchauspe",
    "Inchauspé", "Inchauspe", "Louisa Nicola", "Gabrielle Lyon", "Bryan Johnson",
    "Gary Brecka", "Brecka",
]
NAME_PATTERNS = [
    r"\b(?:" + "|".join(re.escape(n) for n in CREATOR_NAMES) + r")(?:'s|’s)?",
    r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",                     # Dr. Jordan Peterson
    r"\|\s*[A-Z][^|]{0,60}$",                                         # ... | Edward Chang, D.O.
    r"\b(?:with|ft\.?|feat\.?)\s+(?:Dr\.?\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+",  # with Jane Smith
]
NAME_REGEX = re.compile("|".join(NAME_PATTERNS))

EMOJI_REGEX = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿\U0001F000-\U0001F2FF]")

# ---------------------------------------------------------------------------
# Feature columns used by the model
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "char_count", "word_count", "has_question", "has_number", "has_you", "has_emoji",
    "caps_ratio", "hashtag_count", "has_person_name", "claim_intensity", "duration_sec",
]
CATEGORICAL_FEATURES = [
    "hook_style", "topic", "duration_bucket", "publish_hour", "publish_weekday",
]
TEXT_FEATURE = "title_clean"


def first_match(title, rules):
    """Return the first rule name whose regex matches the title, else 'other'."""
    text = title.lower()
    for name, pattern in rules.items():
        if re.search(pattern, text):
            return name
    return "other"


def strip_names(title):
    """Remove person names so TF-IDF learns from words, not who is in the video."""
    return re.sub(r"\s+", " ", NAME_REGEX.sub(" ", title)).strip()


def caps_ratio(title):
    letters = [c for c in title if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def add_features(df):
    """Return a copy of df with every model feature added.

    Needs columns: title, published_at, duration_sec.
    """
    out = df.copy()
    title = out["title"].fillna("")
    published = pd.to_datetime(out["published_at"], utc=True)

    # Title structure
    out["char_count"] = title.str.len()
    out["word_count"] = title.str.split().str.len()
    out["has_question"] = title.str.contains("?", regex=False).astype(int)
    out["has_number"] = title.str.contains(r"\d").astype(int)
    out["has_you"] = title.str.contains(r"\byou(?:r|'re|rs|rself)?\b", case=False).astype(int)
    out["has_emoji"] = title.str.contains(EMOJI_REGEX).astype(int)
    out["caps_ratio"] = title.apply(caps_ratio).round(3)
    out["hashtag_count"] = title.str.count(r"#[A-Za-z]")
    out["has_person_name"] = title.str.contains(NAME_REGEX).astype(int)

    # Hook style, topic, claim intensity
    out["hook_style"] = title.apply(first_match, rules=HOOK_STYLE_RULES)
    out["topic"] = title.apply(first_match, rules=TOPIC_RULES)
    out["claim_intensity"] = title.str.lower().str.count(HYPE_PATTERN)

    # Timing and format (hour/weekday as categories: the effect isn't a straight line)
    out["publish_hour"] = published.dt.hour.astype(str)
    out["publish_weekday"] = published.dt.day_name().str[:3]
    out["duration_bucket"] = pd.cut(
        out["duration_sec"], bins=[0, 30, 60, 90, 10_000],
        labels=["≤30s", "31–60s", "61–90s", "91s+"],
    ).astype(str)

    # Text for TF-IDF
    out[TEXT_FEATURE] = title.apply(strip_names)
    return out


def build_preprocessor(max_tfidf_terms=500):
    """ColumnTransformer: scale numbers, one-hot categories, TF-IDF the cleaned title."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("text", TfidfVectorizer(max_features=max_tfidf_terms, stop_words="english",
                                 ngram_range=(1, 2), min_df=3), TEXT_FEATURE),
    ])
