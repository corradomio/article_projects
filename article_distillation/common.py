from datetime import datetime
import pandas as pd


def reshape(l: list[str], columns: list[str]):
    m = len(columns)
    M = []
    for i in range(0, len(l), m):
        M.append(l[i:i+m])
    return pd.DataFrame(data=M, columns=columns)


def delta_time(start: datetime, done: datetime):
    seconds = int((done - start).total_seconds())
    if seconds < 60:
        return f"{seconds} s"
    elif seconds < 3600:
        s = seconds % 60
        m = seconds // 60
        return f"{m:02}:{s:02} s"
    else:
        s = seconds % 60
        seconds = seconds // 60
        m = seconds % 60
        h = seconds // 60
        return f"{h:02}:{m:02}:{s:02} s"
# end

