from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, abort, render_template, request

from templates.methodology.phase3.features import normalize_url, registered_domain

app = Flask(__name__)
DATASET_PATH = (
    Path(__file__).resolve().parent
    / "static"
    / "datasets"
    / "balanced_dataset_complete.csv"
)
TRANCO_PATH = (
    Path(__file__).resolve().parent / "static" / "datasets" / "top-1m.csv"
)
TRANCO_TOP_N = 20_000
MAX_URL_LENGTH = 2048

METHODOLOGY_PHASES = (
    {
        "slug": "business-understanding",
        "name": "Business Understanding",
        "template": "methodology/business_understanding.html",
    },
    {
        "slug": "data-understanding",
        "name": "Data Understanding",
        "template": "methodology/data_understanding.html",
    },
    {
        "slug": "data-preparation",
        "name": "Data Preparation",
        "template": "methodology/phase3/data_preparation.html",
    },
    {
        "slug": "modeling",
        "name": "Modeling",
        "template": "methodology/modeling.html",
    },
    {
        "slug": "evaluation",
        "name": "Evaluation",
        "template": "methodology/evaluation.html",
    },
    {
        "slug": "deployment",
        "name": "Deployment",
        "template": "methodology/deployment.html",
    },
)


@app.context_processor
def inject_methodology_phases():
    return {"methodology_phases": METHODOLOGY_PHASES}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/pgc/", strict_slashes=False)
def pgc():
    return render_template("PGC/pgc.html")


@app.route("/application/context/")
def application_context():
    return render_template("application/context.html")


@app.route("/application/platform/", methods=["GET", "POST"])
def application_platform():
    result = None
    error = None
    submitted_url = ""

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()
        if not submitted_url:
            error = "Enter a URL to analyze."
        elif len(submitted_url) > MAX_URL_LENGTH:
            error = f"The URL must be {MAX_URL_LENGTH} characters or fewer."
        else:
            normalized_url = ""
            try:
                normalized_url = normalize_url(submitted_url)
                if not normalized_url:
                    error = "The URL could not be recognized. Check it and try again."
            except ValueError:
                error = "The URL contains an invalid host or port."

            if normalized_url and not error:
                try:
                    dataset_index, conflicting_urls = load_dataset_index()
                    popular_domains = load_tranco_index()
                except FileNotFoundError as exc:
                    app.logger.error("Unable to load URL reference data: %s", exc)
                    error = "The URL reference data is unavailable."
                else:
                    if normalized_url in conflicting_urls:
                        result = {
                            "normalized_url": normalized_url,
                            "status": "conflict",
                            "label": "Conflicting records",
                            "message": (
                                "This URL has conflicting labels in the dataset, "
                                "so it cannot be classified reliably."
                            ),
                            "level": "warning",
                        }
                    elif normalized_url in dataset_index:
                        is_phishing = dataset_index[normalized_url] == 1
                        result = {
                            "normalized_url": normalized_url,
                            "status": "registered",
                            "label": (
                                "Registered as phishing"
                                if is_phishing
                                else "Registered as legitimate"
                            ),
                            "message": (
                                "This URL is listed as a phishing indicator in the dataset."
                                if is_phishing
                                else "This URL is listed as legitimate in the dataset."
                            ),
                            "level": "danger" if is_phishing else "success",
                        }
                    else:
                        domain = registered_domain(normalized_url)
                        rank = popular_domains.get(domain)
                        if rank is not None:
                            result = {
                                "normalized_url": normalized_url,
                                "status": "popular_domain",
                                "label": "Popular domain - possibly legitimate",
                                "message": (
                                    f"The domain {domain} appears at rank {rank:,} in "
                                    f"the Tranco top {TRANCO_TOP_N:,}. This URL itself "
                                    "was not found in the dataset, and popularity does "
                                    "not guarantee that this specific link is safe."
                                ),
                                "level": "success",
                            }
                        else:
                            result = {
                                "normalized_url": normalized_url,
                                "status": "not_found",
                                "label": "Not registered",
                                "message": (
                                    "This exact URL was not found in the dataset, and its "
                                    "domain is not in the Tranco top 20,000. No phishing "
                                    "prediction was made."
                                ),
                                "level": "secondary",
                            }

    return render_template(
        "application/platform.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        reference_data_available=DATASET_PATH.is_file() and TRANCO_PATH.is_file(),
    )


@lru_cache(maxsize=1)
def load_dataset_index():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"URL reference dataset not found at {DATASET_PATH}.")

    data = pd.read_csv(DATASET_PATH, usecols=["normalized_url", "is_phishing"])
    data = data.dropna(subset=["normalized_url", "is_phishing"])
    data["normalized_url"] = data["normalized_url"].astype(str).str.strip()
    data["is_phishing"] = pd.to_numeric(data["is_phishing"], errors="raise").astype(int)
    if not set(data["is_phishing"].unique()).issubset({0, 1}):
        raise ValueError("Dataset is_phishing labels must be binary values 0 and 1.")

    label_counts = data.groupby("normalized_url")["is_phishing"].nunique()
    conflicting_urls = frozenset(label_counts[label_counts > 1].index)
    unique_labels = data.drop_duplicates(
        subset=["normalized_url", "is_phishing"]
    )
    index = {
        row.normalized_url: int(row.is_phishing)
        for row in unique_labels.itertuples(index=False)
        if row.normalized_url not in conflicting_urls
    }
    return index, conflicting_urls


@lru_cache(maxsize=1)
def load_tranco_index():
    if not TRANCO_PATH.is_file():
        raise FileNotFoundError(f"Tranco ranking not found at {TRANCO_PATH}.")

    ranking = pd.read_csv(
        TRANCO_PATH,
        header=None,
        names=["rank", "domain"],
        usecols=[0, 1],
        nrows=TRANCO_TOP_N,
    )
    ranking["domain"] = ranking["domain"].astype(str).str.strip().str.lower()
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="raise").astype(int)
    return dict(zip(ranking["domain"], ranking["rank"]))


@app.route("/methodology/<phase_slug>/")
def methodology_phase(phase_slug):
    phase_index = next(
        (
            index
            for index, phase in enumerate(METHODOLOGY_PHASES)
            if phase["slug"] == phase_slug
        ),
        None,
    )
    if phase_index is None:
        abort(404)

    return render_template(
        METHODOLOGY_PHASES[phase_index]["template"],
        phase=METHODOLOGY_PHASES[phase_index],
        phase_number=phase_index + 1,
        previous_phase=METHODOLOGY_PHASES[phase_index - 1] if phase_index else None,
        next_phase=(
            METHODOLOGY_PHASES[phase_index + 1]
            if phase_index + 1 < len(METHODOLOGY_PHASES)
            else None
        ),
        total_phases=len(METHODOLOGY_PHASES),
    )


if __name__ == "__main__":
    app.run()
