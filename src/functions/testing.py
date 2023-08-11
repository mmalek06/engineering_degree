def group_models(names: list[str]) -> list[list[str]]:
    groups = {}

    for name in names:
        group = '_'.join(name.split('_')[:-1])

        if group not in groups:
            groups[group] = []

        groups[group].append(name)

    values = list(groups.values())
    sorted_values = []

    for group in values:
        sorted_values.append(sorted(group, key=lambda x: int(x.split('_')[-1])))

    return sorted_values
