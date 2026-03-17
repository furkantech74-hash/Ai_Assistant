import json


#dataset cleaning to replace special characters with their textual representation

with open("data/company_qa.json", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item["answer"] = (
        item["answer"]
        .replace("→", "->")
        .replace("–", "-")
        .replace("₹", "Rs")
    )

with open("data/company_qa.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Dataset cleaned successfully")
