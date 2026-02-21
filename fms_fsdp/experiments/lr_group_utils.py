def parse_lr_groups(s):
    if not s:
        return []
    groups = []
    for entry in s.split(';'):
        pattern, scale = entry.split(':')
        groups.append((pattern.strip(), float(scale.strip())))
    return groups

def get_lr_scale(name, lr_groups):
    for pattern, scale in lr_groups:
        if pattern in name:
            return scale
    return 1.0
