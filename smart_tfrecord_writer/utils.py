def convert_to_human_readable(total_mb):
    if total_mb >= 1024:
        gb = total_mb / 1024
        if gb >= 1024:
            return f"{gb / 1024:.2f} TB"
        return f"{gb:.2f} GB"
    if total_mb >= 1:
        return f"{total_mb:.1f} MB"
    kb = total_mb * 1024
    return f"{kb:.1f} KB"
