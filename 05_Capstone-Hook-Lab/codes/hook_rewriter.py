"""Shorts Hook Rewriter (Part B of the Longevity Hook Lab).

Takes a draft YouTube Shorts title, asks Gemini for 5 rewrites (one per hook style),
scores each rewrite and the draft with the Part A model (P(beat)), drops rewrites that
break the guardrails, and returns the best title with a short "why".

Command line:
    python hook_rewriter.py "Why you wake up at 3am" --topic sleep
    python hook_rewriter.py "Magnesium before bed" --topic supplements --duration 25

In Python / the demo notebook:
    from hook_rewriter import rewrite
    result = rewrite("Why you wake up at 3am", topic="sleep")

Only the title changes between draft and rewrites: duration and posting time are held
at the draft's values, so any change in score comes from the title alone.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE,
                      TOPIC_RULES, add_features, first_match)

ROOT = Path(__file__).resolve().parents[1]          # longevity-hook-lab/
MODEL_PATH = ROOT / "models" / "hook_model.joblib"
CACHE_PATH = ROOT / "data" / "gemini_cache.json"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODEL = "gemini-flash-lite-latest"

HOOK_STYLES = ["question", "how_to", "list", "warning", "story"]
TOPICS = list(TOPIC_RULES) + ["other"]
MAX_CHARS = 70

# Never allowed in a rewrite, even if the draft uses them (responsible-AI rule, not a model finding).
BANNED_PHRASES = [
    "cure", "cures", "cured", "miracle", "reverse aging", "reverse ageing", "reverses aging",
    "guaranteed", "instantly", "100%", "doctors hate", "big pharma", "detox", "melt fat",
    "burn fat fast", "shocking", "life-changing", "secret", "secrets", "toxic", "deadly",
]
BANNED_REGEX = re.compile(r"(?<!\w)(" + "|".join(re.escape(p) for p in BANNED_PHRASES) + r")(?!\w)",
                          re.IGNORECASE)

# Point of view. Many titles are about someone else (a guest, a patient, "she"), so a rewrite must not
# add a speaker or person the draft doesn't have ("she" -> "I", "she" -> "you", invented "her").
PRONOUNS = {
    "first person": r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|we're|we've|us|our|ours|ourselves)\b",
    "'you'": r"\b(you|you're|you've|your|yours|yourself)\b",
    "third person": r"\b(he|she|he's|she's|him|her|his|hers|himself|herself)\b",
}

# Bump when the prompt changes, so cached replies from the old prompt aren't reused.
PROMPT_VERSION = "v3"

PROMPT = """You write titles for YouTube Shorts in the health and longevity niche.

Draft title: "{draft}"
Topic: {topic}

Write exactly 5 new titles for the SAME video, one in each hook style:
- question: asks the viewer a question
- how_to: "how to ..." or a practical tip
- list: starts with or contains a number of items, but ONLY if the draft already has that number
- warning: a mistake to avoid or a thing to stop doing
- story: a short personal moment, told from the same point of view as the draft. Use "I"/"my" ONLY
  if the draft is already in the first person. If the draft is about someone else (a guest, a
  patient, "she", "he", "they"), tell their story in the third person.

Rules:
- Keep the facts and claims identical to the draft. Do not add new numbers, studies,
  percentages, time frames or health promises.
- Do not change what the video is about.
- Keep the draft's point of view and speaker. Never turn someone else's words or story into "I" or "my".
  Only use "I"/"we", "you" or "she"/"he" if the draft already uses that kind of word.
- No hype words (cure, miracle, reverse aging, secret, shocking, instantly, guaranteed, toxic).
- 70 characters or fewer. Normal sentence or title case, no ALL CAPS words, no emoji, no hashtags.
- If a hook style cannot be done without breaking a rule, write the closest faithful version.

Reply with JSON only, in this form:
{{"question": "...", "how_to": "...", "list": "...", "warning": "...", "story": "..."}}"""


# ---------------------------------------------------------------------------
# 1. Gemini call (cached, so re-running the notebook doesn't spend quota)
# ---------------------------------------------------------------------------

def load_cache():
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def generate_variants(draft, topic, use_cache=True):
    """One Gemini call -> dict {hook_style: title}."""
    key = f"{PROMPT_VERSION}|{topic}|{draft}"
    cache = load_cache() if use_cache else {}
    if key in cache:
        return cache[key]

    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Free tier is often busy (503) or rate-limited (429): wait and retry, then try the lite model.
    for attempt, model_name in enumerate([GEMINI_MODEL] * 3 + [FALLBACK_MODEL] * 3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=PROMPT.format(draft=draft, topic=topic),
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7),
            )
            break
        except Exception as e:   # API errors and dropped connections
            if attempt == 5 or (isinstance(e, genai.errors.APIError) and e.code not in (429, 500, 503)):
                raise
            time.sleep(10 * (attempt % 3 + 1))
    variants = {style: str(title).strip() for style, title in json.loads(response.text).items()
                if style in HOOK_STYLES}

    if use_cache:
        cache = load_cache()
        cache[key] = variants
        save_cache(cache)
    return variants


# ---------------------------------------------------------------------------
# 2. Scoring with the Part A model
# ---------------------------------------------------------------------------

def load_model():
    return joblib.load(MODEL_PATH)


def score_titles(titles, model, duration_sec=45, published_at=None):
    """P(beat) for each title, holding duration and posting time fixed. Returns (scores, features)."""
    published_at = published_at or pd.Timestamp.now(tz="UTC")
    rows = pd.DataFrame({"title": titles, "duration_sec": duration_sec, "published_at": published_at})
    feats = add_features(rows)
    X = feats[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
    return model.predict_proba(X)[:, 1], feats


# ---------------------------------------------------------------------------
# 3. Guardrails
# ---------------------------------------------------------------------------

def numbers_in(title):
    return set(re.findall(r"\d+(?:\.\d+)?", title))


def check_guardrails(variant, draft, variant_feats, draft_feats):
    """Return a list of broken rules (empty list = passes)."""
    problems = []
    if len(variant) > MAX_CHARS:
        problems.append(f"over {MAX_CHARS} chars")
    if BANNED_REGEX.search(variant):
        problems.append(f"banned phrase '{BANNED_REGEX.search(variant).group(0)}'")
    if variant_feats["claim_intensity"] > draft_feats["claim_intensity"]:
        problems.append("more hype words than draft")
    new_numbers = numbers_in(variant) - numbers_in(draft)
    if new_numbers:
        problems.append(f"new number(s) {sorted(new_numbers)}")
    # The forest likes capital letters; don't let a rewrite win by shouting.
    if variant_feats["caps_ratio"] > draft_feats["caps_ratio"] + 0.10:
        problems.append("more capitals than draft")
    # Same subject: if the draft has a clear topic, the rewrite must keep it.
    if draft_feats["topic"] != "other" and variant_feats["topic"] != draft_feats["topic"]:
        problems.append(f"topic changed ({draft_feats['topic']} -> {variant_feats['topic']})")
    # Same point of view: no new kind of pronoun, and keep the person the draft is about.
    for name, pattern in PRONOUNS.items():
        in_draft = re.search(pattern, draft, re.IGNORECASE)
        in_variant = re.search(pattern, variant, re.IGNORECASE)
        if in_variant and not in_draft:
            problems.append(f"point of view changed (added {name})")
        if name == "third person" and in_draft and not in_variant:
            problems.append("point of view changed (dropped third person)")
    return problems


# ---------------------------------------------------------------------------
# 4. "Why": which title features changed between draft and best rewrite
# ---------------------------------------------------------------------------

def explain(draft_feats, best_feats):
    reasons = []
    for col in ["hook_style", "topic"]:
        if best_feats[col] != draft_feats[col]:
            reasons.append(f"{col.replace('_', ' ')}: {draft_feats[col]} -> {best_feats[col]}")
    for col, label in [("has_question", "question mark"), ("has_you", "'you'"), ("has_number", "number")]:
        if best_feats[col] != draft_feats[col]:
            reasons.append(f"{'+' if best_feats[col] else '-'} {label}")
    diff = best_feats["char_count"] - draft_feats["char_count"]
    if abs(diff) >= 10:
        reasons.append(f"{'longer' if diff > 0 else 'shorter'} by {abs(diff)} chars")
    return reasons or ["wording only (TF-IDF terms)"]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def rewrite(draft, topic="other", duration_sec=45, published_at=None, model=None, use_cache=True):
    """Rewrite a draft title. Returns a dict with the best title, scores, why, and all candidates."""
    model = model or load_model()
    variants = generate_variants(draft, topic, use_cache=use_cache)

    titles = [draft] + list(variants.values())
    scores, feats = score_titles(titles, model, duration_sec, published_at)
    draft_feats = feats.iloc[0]

    candidates = []
    for i, (style, title) in enumerate(variants.items(), start=1):
        problems = check_guardrails(title, draft, feats.iloc[i], draft_feats)
        candidates.append({"style": style, "title": title, "p_beat": round(float(scores[i]), 3),
                           "passes": not problems, "problems": "; ".join(problems)})
    candidates = pd.DataFrame(candidates).sort_values("p_beat", ascending=False)

    passing = candidates[candidates["passes"]]
    draft_score = round(float(scores[0]), 3)
    if passing.empty or passing["p_beat"].iloc[0] <= draft_score:
        best_title, best_score, why = draft, draft_score, ["keep the draft: no safe rewrite scored higher"]
    else:
        best = passing.iloc[0]
        best_title, best_score = best["title"], best["p_beat"]
        why = explain(draft_feats, feats.iloc[titles.index(best_title)])

    return {"draft": draft, "draft_p_beat": draft_score, "best_title": best_title,
            "best_p_beat": best_score, "uplift": round(best_score - draft_score, 3),
            "why": why, "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description="Rewrite a Shorts title and score it with the Part A model.")
    parser.add_argument("draft", help="draft title, in quotes")
    parser.add_argument("--topic", default="other", choices=TOPICS)
    parser.add_argument("--duration", type=int, default=45, help="video length in seconds (default 45)")
    args = parser.parse_args()

    result = rewrite(args.draft, args.topic, args.duration)
    print(f"\nDraft : {result['draft']}  (P(beat) = {result['draft_p_beat']:.2f})")
    print(f"Best  : {result['best_title']}  (P(beat) = {result['best_p_beat']:.2f}, "
          f"{result['uplift']:+.2f})")
    print("Why   : " + ", ".join(result["why"]))
    print("\nAll candidates:")
    print(result["candidates"].to_string(index=False))
    print("\nNote: a higher model score is a hint, not a guarantee of more views. A/B test it.")


if __name__ == "__main__":
    main()
