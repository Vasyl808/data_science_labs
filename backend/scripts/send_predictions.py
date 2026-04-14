import json
import time
import requests
import os


PREDICT_URL = "http://127.0.0.1:8000/api/v1/predict"
UPDATE_TRUE_LABEL_URL = "http://127.0.0.1:8000/api/v1/predictions/{}/true-label"


def main():
    data_file = os.path.join(os.path.dirname(__file__), "..", "test_data.jsonl")

    if not os.path.exists(data_file):
        print(f"File {data_file} not found.")
        return

    predict_success = 0
    predict_error = 0
    label_success = 0
    label_error = 0

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Starting to send {len(lines)} requests...")

        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                true_label = data.pop("true_label", None)

                response = requests.post(PREDICT_URL, json=data)

                if response.status_code == 200:
                    result = response.json()
                    prediction_id = result.get("prediction_id")
                    predict_success += 1
                    print(f"[{idx}/{len(lines)}] Predict OK: id={prediction_id}, pred={result.get('prediction')}")

                    if true_label is not None and prediction_id:
                        update_url = UPDATE_TRUE_LABEL_URL.format(prediction_id)
                        update_resp = requests.patch(update_url, json={"true_label": true_label})

                        if update_resp.status_code == 200:
                            label_success += 1
                            print(f"    -> True label updated: {true_label}")
                        else:
                            label_error += 1
                            print(f"    -> Failed to update true_label: {update_resp.status_code}")
                else:
                    predict_error += 1
                    print(f"[{idx}/{len(lines)}] Predict error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                predict_error += 1
                print(f"[{idx}/{len(lines)}] Connection error: {e}")
            except json.JSONDecodeError:
                predict_error += 1
                print(f"[{idx}/{len(lines)}] JSON format error in line")

            time.sleep(0.01)

    print(f"\nDone!")
    print(f"  Predictions: {predict_success} success, {predict_error} errors")
    print(f"  True labels: {label_success} updated, {label_error} failed")


if __name__ == "__main__":
    main()
