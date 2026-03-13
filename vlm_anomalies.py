import ollama
import pandas as pd
import json
import os

CSV_PATH = "detection_results/results.csv"
OUTPUT_PATH = "vlm_descriptions.json"

def describe_anomaly(image_path, defect_class):

    prompt = f"""You are an industrial quality control expert analyzing a glass bottle top-down view.
A defect of type '{defect_class}' has been detected on this bottle.
Describe in one sentence the defect you see in the image."""

    with open(image_path, "rb") as f:
        image_data = f.read()

    response = ollama.chat(
        model="llava-llama3",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_data]
        }]
    )

    parts = image_path.replace("\\", "/").split("/")
    short_path = "/".join(parts[-3:])

    return {
        "image_path": short_path, 
        "description": response["message"]["content"].strip(),
        "anomaly_score": None,
        "defect_class": defect_class
    }

def main():
    df = pd.read_csv(CSV_PATH)
    anomalies = df[df["prediction"] == 1]
    print(f"Found {len(anomalies)} predicted anomalies")

    descriptions = []
    for _, row in anomalies.iterrows():
        img_path = row["path"]

        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        print(f"Processing: {img_path}")

        try:
            result = describe_anomaly(img_path, row["class"])
            result["anomaly_score"] = row["score"]
            descriptions.append(result)
            print(f"  ✓ {result['description'][:80]}...")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(descriptions, f, indent=2)

    print(f"\n✓ Saved {len(descriptions)} descriptions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()