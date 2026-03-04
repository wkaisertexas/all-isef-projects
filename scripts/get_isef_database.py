import os
import html as htmllib
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm


# Gets all projects into a DataFrame using an automated blank search
SEARCH_URL = "https://abstracts.societyforscience.org/"
s = requests.Session()


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_awards_cell(cell):
    awards = []
    current_parts = []

    def flush_current():
        nonlocal current_parts
        if not current_parts:
            return
        text = htmllib.unescape(" ".join(current_parts))
        text = normalize_whitespace(text)
        if text:
            # Some rows include visual line breaks inside one award description.
            if awards and (
                text.startswith(("•", "-", "–", "—"))
                or text.startswith("(")
                or (text and text[0].islower())
                or awards[-1].endswith(":")
                or awards[-1].endswith("&")
            ):
                awards[-1] = normalize_whitespace(f"{awards[-1]} {text}")
            else:
                awards.append(text)
        current_parts = []

    for child in cell.children:
        if getattr(child, "name", None) == "br":
            flush_current()
            continue

        if hasattr(child, "get_text"):
            child_text = child.get_text(" ", strip=True)
        else:
            child_text = str(child).strip()

        child_text = normalize_whitespace(child_text)
        if child_text:
            current_parts.append(child_text)

    flush_current()
    return awards if awards else None


def parse_project_metadata_from_results(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for row in soup.select("#tblAbstractSearchResults tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        link = cells[2].find("a", href=True)
        if not link:
            continue

        query = parse_qs(urlparse(link.get("href", "")).query)
        pid = (query.get("projectId") or query.get("ProjectId") or [None])[0]
        if not pid:
            continue

        def clean_cell_text(cell):
            value = normalize_whitespace(cell.get_text(" ", strip=True))
            return value if value else None

        rows.append(
            {
                "id": str(pid),
                "Project Title": normalize_whitespace(link.get_text(" ", strip=True)),
                "Fair Country": clean_cell_text(cells[4]),
                "Fair State": clean_cell_text(cells[5]),
                "Fair Province": clean_cell_text(cells[6]),
                "Awards Won": parse_awards_cell(cells[7]),
            }
        )

    return pd.DataFrame(rows)


def fetch_all_projects_html(session):
    landing = session.get(SEARCH_URL, timeout=60)
    landing.raise_for_status()

    soup = BeautifulSoup(landing.text, "html.parser")

    token_input = soup.find("input", attrs={"name": "__RequestVerificationToken"})
    if token_input is None or not token_input.get("value"):
        raise RuntimeError("Could not find __RequestVerificationToken on landing page")

    year_values = [
        x.get("value")
        for x in soup.select("input[name='SelectedIsefYears'][type='checkbox']")
        if x.get("value")
    ]

    payload = {
        "__RequestVerificationToken": token_input["value"],
        "KeywordOrPhrase": "",
        "FinalistLastName": "",
        "IsGetAllAbstracts": "True",
    }

    if "0" in year_values:
        payload["SelectedIsefYears"] = "0"
    elif year_values:
        payload["SelectedIsefYears"] = year_values

    response = session.post(SEARCH_URL, data=payload, timeout=180)
    response.raise_for_status()
    return response.text


html = fetch_all_projects_html(s)
Path("projects.html").write_text(html, encoding="utf-8")

results = parse_project_metadata_from_results(html)
ids = results["id"].astype("string").drop_duplicates().reset_index(drop=True)

print(f"There are {len(ids)} projects")

# Gets the abstract and additional project information from each page
MAX_WORKERS = min(32, max(4, (os.cpu_count() or 1) * 5))
_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=1,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def get_paper_data(id):
    try:
        session = get_session()
        root = "https://abstracts.societyforscience.org/Home/FullAbstract?projectId="
        response = session.get(root + str(id), timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Gets the div whose class is row
        col = soup.find("div", attrs={"class": "col-sm-12"})
        elements = col.find_all()

        title = elements[0].text

        category = elements[4].text.split("Category:")[1]

        year = int(elements[7].text.split("Year:")[1])

        finalists = elements[10].text.replace("  ", "").replace("\n", "")
        schools = [x.split(")")[0] for x in finalists.split("(School: ")[1:]]
        schools = set(schools)  # Removes duplicates

        abstract = elements[14].text.replace("Abstract:", "")

        return {
            "id": id,
            "title": title,
            "category": category,
            "year": year,
            "schools": schools,
            "abstract": abstract,
        }
    except Exception as e:
        return {
            "id": id,
            "title": None,
            "category": None,
            "year": None,
            "schools": None,
            "abstract": None,
            "error": e,
        }


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    records = list(
        tqdm(
            executor.map(get_paper_data, ids.tolist()),
            total=len(ids),
            desc="Fetching abstracts",
        )
    )

database = pd.DataFrame(records)

database.dropna(subset=["title"], inplace=True)
database.head()


print(f"Length Before:{len(database.index):10d}")
results = results.drop(columns=["Project Title"], errors="ignore")
results["id"] = results["id"].astype("string")
database["id"] = database["id"].astype("string")
database = database.merge(results, on="id", how="left")
print(f"Length After: {len(database.index):10d}")
database.head()

# Renames some columns
database.rename(
    columns={
        "Fair Country": "country",
        "Fair State": "State",
        "Fair Province": "Province",
        "Awards Won": "awards",
    },
    inplace=True,
)

# Saves the file to CSV
output_path = Path("./data/isef-database.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
database.to_csv(output_path, index=False)

# Removes projects.html from the output
Path("projects.html").unlink(missing_ok=True)

print(f"Saved database to: {output_path.resolve()}")
