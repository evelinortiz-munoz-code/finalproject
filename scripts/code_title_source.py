"""Dictionary-method coding of each document's title and publisher source in
ri_snap_food_access_documents.csv.

Keyword lists come from data/Week_6_publishertypecodebook.xlsx (one row per
category, keywords as quoted strings). The title and publisher_author fields
are joined and lowercased, and keyword hits are counted per category; the
category with the most hits is assigned. No hits -> "Uncoded"; a tie for the
most hits -> "Tie".

Usage:
    python scripts/code_title_source.py [codebook.xlsx]

The codebook path defaults to data/Week_6_publishertypecodebook.xlsx; pass a
copy if the original is locked because it is open in Excel.

Output, in data/:
    ri_snap_food_access_title_source_coded.csv
        document number, document name (title), publisher source, category
        (plus per-category counts and matched terms)
"""
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CODEBOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "Week_6_publishertypecodebook.xlsx"
DOCS = DATA / "ri_snap_food_access_documents.csv"
OUT = DATA / "ri_snap_food_access_title_source_coded.csv"


def load_dictionary(path):
    """Return {category: [keywords]} from the codebook workbook."""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(min_row=2, values_only=True))
    wb.close()
    dictionary = {}
    for category, keywords in rows:
        if not category:
            continue
        # keywords look like: "news," "current," "journalist, "interview"
        # (quotes are unbalanced in places, so drop them and split on commas)
        terms = [t.strip().lower() for t in re.sub(r'["\u201c\u201d]', "", keywords).split(",")]
        dictionary[category.strip()] = [t for t in terms if t]
    return dictionary


def count_hits(text, terms):
    """Whole-word matches (optional plural 's') for each term; returns {term: n}."""
    hits = {}
    for term in terms:
        n = len(re.findall(r"\b" + re.escape(term) + r"s?\b", text))
        if n:
            hits[term] = n
    return hits


def code_text(text, dictionary):
    text = text.lower()
    hits = {cat: count_hits(text, terms) for cat, terms in dictionary.items()}
    totals = {cat: sum(h.values()) for cat, h in hits.items()}
    top = max(totals.values())
    winners = [c for c, n in totals.items() if n == top]
    category = "Uncoded" if top == 0 else winners[0] if len(winners) == 1 else "Tie"
    return category, totals, hits


def main():
    dictionary = load_dictionary(CODEBOOK)
    docs = pd.read_csv(DOCS, encoding="utf-8-sig")
    rows = []
    for _, doc in docs.iterrows():
        title = str(doc["title"]) if pd.notna(doc["title"]) else ""
        source = str(doc["publisher_author"]) if pd.notna(doc["publisher_author"]) else ""
        category, totals, hits = code_text(f"{title} {source}", dictionary)
        row = {
            "document_number": doc["id"],
            "document_name": title,
            "publisher_source": source,
            "category": category,
        }
        row.update({f"n_{c}": n for c, n in totals.items()})
        row["matched_terms"] = "; ".join(
            f"{t} ({n})" for h in hits.values() for t, n in h.items()
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_colwidth", 55)
    print(out[["document_number", "document_name", "publisher_source", "category", "matched_terms"]].to_string(index=False))
    print("\n", out["category"].value_counts().to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
