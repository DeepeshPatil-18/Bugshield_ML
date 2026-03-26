import json
import os
import pandas as pd

INPUT_DIR = "data/nvd_json"
OUTPUT_FILE = "data/processed/nvd_dataset.csv"

rows = []

print("Processing NVD JSON files...\n")

for file in os.listdir(INPUT_DIR):

    if not file.endswith(".json"):
        continue

    path = os.path.join(INPUT_DIR, file)

    print("Reading:", file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vulnerabilities = data["vulnerabilities"]

    for item in vulnerabilities:

        cve = item["cve"]

        cve_id = cve["id"]

        description = None
        cwe = None
        cvss_score = None
        severity = None
        attack_vector = None
        attack_complexity = None
        privileges_required = None
        user_interaction = None

        # description
        try:
            description = cve["descriptions"][0]["value"]
        except:
            pass

        # CWE
        try:
            cwe = cve["weaknesses"][0]["description"][0]["value"]
        except:
            pass

        # CVSS metrics
        try:
            metrics = cve["metrics"]

            if "cvssMetricV31" in metrics:
                metric = metrics["cvssMetricV31"][0]["cvssData"]

            elif "cvssMetricV30" in metrics:
                metric = metrics["cvssMetricV30"][0]["cvssData"]

            else:
                metric = None

            if metric:
                cvss_score = metric["baseScore"]
                severity = metric["baseSeverity"]
                attack_vector = metric["attackVector"]
                attack_complexity = metric["attackComplexity"]
                privileges_required = metric["privilegesRequired"]
                user_interaction = metric["userInteraction"]

        except:
            pass

        rows.append([
            cve_id,
            description,
            cwe,
            cvss_score,
            attack_vector,
            attack_complexity,
            privileges_required,
            user_interaction,
            severity
        ])


print("\nCreating dataset...")

df = pd.DataFrame(rows, columns=[
    "cve_id",
    "description",
    "cwe",
    "cvss_score",
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "severity"
])

print("Total vulnerabilities:", len(df))

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset saved to:", OUTPUT_FILE)