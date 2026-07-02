import csv
import json
from collections import Counter
from pathlib import Path


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def message_columns(row):
    columns = [name for name in row.keys() if name.startswith("message_")]
    return sorted(columns, key=lambda name: int(name.split("_", 1)[1]))


def parse_mhj_messages(row):
    messages = []
    for column in message_columns(row):
        raw_message = (row.get(column) or "").strip()
        if not raw_message:
            continue

        message = json.loads(raw_message)
        role = message.get("role")
        body = message.get("body")
        if not role or body is None:
            raise ValueError("Invalid MHJ message in column {}".format(column))

        messages.append({
            "role": role,
            "body": body,
        })

    return messages


def count_user_turns(messages):
    return sum(1 for message in messages if message["role"] == "user")


def behavior_id_for_question_id(question_id, harmbench_rows, question_id_base=2):
    index = int(question_id) - question_id_base
    if index < 0 or index >= len(harmbench_rows):
        raise ValueError(
            "question_id={} is outside HarmBench row range with base={}".format(
                question_id,
                question_id_base,
            )
        )

    behavior_id = harmbench_rows[index].get("BehaviorID")
    if not behavior_id:
        raise ValueError("HarmBench row {} does not contain BehaviorID".format(index))

    return behavior_id


def load_mhj_records(
        mhj_csv,
        harmbench_csv,
        scope="multi_turn",
        question_id_base=2,
):
    if scope not in {"multi_turn", "all"}:
        raise ValueError("scope must be one of: multi_turn, all")

    mhj_rows = read_csv_rows(mhj_csv)
    harmbench_rows = read_csv_rows(harmbench_csv)

    records = []
    for row_index, row in enumerate(mhj_rows, start=1):
        messages = parse_mhj_messages(row)
        user_turn_count = count_user_turns(messages)
        if scope == "multi_turn" and user_turn_count <= 1:
            continue

        records.append({
            "mhj_id": "MHJ_{:04d}".format(row_index),
            "row_index": row_index,
            "behavior_id": behavior_id_for_question_id(
                row["question_id"],
                harmbench_rows,
                question_id_base=question_id_base,
            ),
            "question_id": int(row["question_id"]),
            "source": row["Source"],
            "tactic": row["tactic"],
            "temperature": row["temperature"],
            "time_spent": row["time_spent"],
            "submission_message": row["submission_message"],
            "messages": messages,
            "user_turn_count": user_turn_count,
            "message_count": len(messages),
        })

    return records


def summarize_mhj_records(records):
    user_turn_counts = [record["user_turn_count"] for record in records]
    source_counts = Counter(record["source"] for record in records)
    tactic_counts = Counter(record["tactic"] for record in records)
    behavior_ids = {record["behavior_id"] for record in records}
    question_ids = {record["question_id"] for record in records}

    return {
        "conversations": len(records),
        "multi_turn_conversations": sum(1 for count in user_turn_counts if count > 1),
        "single_turn_conversations": sum(1 for count in user_turn_counts if count == 1),
        "total_user_turns": sum(user_turn_counts),
        "min_user_turns": min(user_turn_counts) if user_turn_counts else 0,
        "max_user_turns": max(user_turn_counts) if user_turn_counts else 0,
        "unique_behavior_ids": len(behavior_ids),
        "unique_question_ids": len(question_ids),
        "source_counts": dict(source_counts.most_common()),
        "tactic_counts": dict(tactic_counts.most_common()),
    }
