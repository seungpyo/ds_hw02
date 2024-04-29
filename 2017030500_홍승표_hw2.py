import sys
from math import log2
from typing import Dict, Set


def parse_file(file_name, is_training=False):
    keys = []
    data = []
    with open(file_name, "r") as f:
        for i, line in enumerate(f):
            l = line.strip().split("\t")
            if i == 0:
                keys = l
            else:
                d = {k: v for k, v in zip(keys, l)}
                data.append(d)
    return data, keys[-1] if is_training else None


def get_feature_info(data) -> Set[Dict[str, str]]:
    assert len(data) > 0
    ret = {k: set() for k in data[0].keys()}
    for d in data:
        for k, v in d.items():
            ret[k].add(v)
    return ret


def get_feature_histogram(data, feature_name, feature_info: Dict[str, Set[str]]):
    histogram = {v: 0 for v in feature_info[feature_name]}
    for d in data:
        histogram[d[feature_name]] += 1
    return histogram


def get_entropy(data, label_feature_name, feature_info):
    histogram = get_feature_histogram(data, label_feature_name, feature_info)
    historgram_probs = {
        k: 0 if len(data) == 0 else v / len(data) for k, v in histogram.items()
    }
    entropy = sum([0 if p == 0 else -p * log2(p) for p in historgram_probs.values()])
    return entropy


def split_data(data, feature_name, feature_info):
    splits = {c: [] for c in feature_info[feature_name]}
    for d in data:
        splits[d[feature_name]].append(d)
    return splits


def build(data, label_feature_name, feature_info):
    # Case 1: No data left to split on
    if len(data) == 0:
        return None
    # Case 2: No features left to split on
    classes = feature_info[label_feature_name]
    if len(classes) == 0:
        return None
    # Case 3: All data has the same class
    classes_of_data = split_data(data, label_feature_name, feature_info)
    for c, d in classes_of_data.items():
        if len(d) == len(data):
            return {
                "class": c,
            }
    entropy = get_entropy(data, label_feature_name, feature_info)
    gain_ratio_by_feature = {}
    for feature_name in feature_info.keys():
        if feature_name == label_feature_name:
            continue
        splits = split_data(data, feature_name, feature_info)
        feature_entropy = sum(
            [
                get_entropy(split, label_feature_name, feature_info)
                * len(split)
                / len(data)
                for split in splits.values()
            ]
        )
        split_info = sum(
            [
                get_entropy(split, label_feature_name, feature_info) + 1e-6
                for split in splits.values()
            ]
        )
        gain_ratio = (entropy - feature_entropy) / split_info
        gain_ratio_by_feature[feature_name] = gain_ratio
    best_feature = max(gain_ratio_by_feature, key=gain_ratio_by_feature.get)
    reamining_features = feature_info.copy()
    reamining_features.pop(best_feature)
    best_splits = split_data(data, best_feature, feature_info)
    return {
        "feature": best_feature,
        "splits": {
            c: build(split, label_feature_name, reamining_features)
            for c, split in best_splits.items()
        },
    }


def classify(tree, record):
    if tree is None:
        return None
    if "class" in tree:
        return tree["class"]
    feature = tree["feature"]
    splits = tree["splits"]
    if record[feature] not in splits:
        return None
    subtree = splits[record[feature]]
    return classify(subtree, record)


if __name__ == "__main__":
    assert len(sys.argv) == 4
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]

    training_data, class_key = parse_file(training_file, is_training=True)
    train_feature_info = get_feature_info(training_data)
    tree = build(training_data, class_key, train_feature_info)
    print(tree)
    test_data, _ = parse_file(test_file, is_training=False)
    test_classes = [classify(tree, record) for record in test_data]
    result_lines = []
    with open(test_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                result_lines.append(f"{line.strip()}\t{class_key}")
                continue
            result_lines.append(f"{line.strip()}\t{test_classes[i-1]}")
    with open(result_file, "w") as f:
        f.write("\n".join(result_lines))
