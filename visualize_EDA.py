import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

README = Path(__file__).parent / "README.md"


def list_headings(text: str):
    return re.findall(r"===\s*(.*?)\s*===", text, flags=re.IGNORECASE)


def find_section_position(text: str, section_title: str):
    # Accept either full "=== Title ===" or bare title.
    title = re.sub(r"=+", "", section_title).strip()
    # 1) exact match ignoring case for === Title ===
    pattern = re.compile(rf"===\s*{re.escape(title)}\s*===", re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.start(), m.end()
    # 2) try headings found by list_headings (robust fallback)
    headings = list_headings(text)
    for h in headings:
        if title.lower() in h.lower() or h.lower() in title.lower():
            pattern2 = re.compile(rf"===\s*{re.escape(h)}\s*===", re.IGNORECASE)
            m2 = pattern2.search(text)
            if m2:
                return m2.start(), m2.end()
    # 3) try to find any line that contains the title text (very tolerant)
    #    (handles cases where README has small formatting differences)
    line_pattern = re.compile(rf"^.*{re.escape(title)}.*$", re.IGNORECASE | re.MULTILINE)
    m3 = line_pattern.search(text)
    if m3:
        return m3.start(), m3.end()
    # 4) nothing
    raise ValueError(f"Section '{section_title}' not found. Available: {headings}")


def parse_section(text: str, section_title: str):
    start, end = find_section_position(text, section_title)
    tail = text[end:]
    lines = tail.splitlines()
    rows = []
    for line in lines:
        # stop at next section header line (=== ... ===) or at an empty line followed by another header
        if re.match(r"===\s*", line):
            break
        # ignore table separators like ------
        if re.match(r"^\s*[-=]{3,}\s*$", line):
            continue
        # extract integers from the line (handles negative if any)
        nums = re.findall(r"-?\d+", line)
        # If the section has rows where the first column might be non-numeric (unlikely here),
        # we still require at least two integers to build (x,y) pairs.
        if len(nums) >= 2:
            rows.append((int(nums[0]), int(nums[1])))
        # stop if we reach a blank line after we've already collected rows
        elif rows and line.strip() == "":
            break
    if not rows:
        raise ValueError(f"No numeric rows found for section '{section_title}'")
    return pd.DataFrame(rows)


def plot_histogram_trips_per_taxi(df: pd.DataFrame, out_path: Path, target_bins: int = 30):
    import math
    df = df.copy()
    df.columns = ["count_trips", "num_taxis"]
    df["count_trips"] = df["count_trips"].astype(int)

    max_count = int(df["count_trips"].max())
    # choose a bin width so we get about `target_bins` bars
    bin_width = max(1, math.ceil(max_count / target_bins))
    bins = list(range(0, max_count + bin_width, bin_width))

    # create human-readable labels for each bin
    labels = [f"{b}-{min(b + bin_width - 1, max_count)}" for b in bins[:-1]]

    # assign counts to bins and aggregate number of taxis
    df["bin"] = pd.cut(df["count_trips"], bins=bins, labels=labels, right=False)
    binned = df.groupby("bin", observed=True)["num_taxis"].sum().reset_index()

    plt.figure(figsize=(14, 6))
    plt.bar(binned["bin"].astype(str), binned["num_taxis"], width=0.8)
    plt.xlabel(f"Trips per taxi (binned, bin width = {bin_width})")
    plt.ylabel("Number of taxis")
    plt.title("Histogram of Trips per Taxi (binned)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_trips_by_hour(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df.columns = ["hour_of_day", "num_trips"]
    df = df.sort_values("hour_of_day")
    plt.figure(figsize=(10, 5))
    plt.bar(df["hour_of_day"], df["num_trips"], color="#2b8cbe")
    plt.xlabel("Hour of day")
    plt.ylabel("Number of trips")
    plt.title("Trips by Hour of Day")
    plt.xticks(range(0, 24))
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    text = README.read_text(encoding="utf-8")
    # use the exact titles you have in README (bare or with ===)
    hist_section = "Histogram of Trips per Taxi"
    hour_section = "Trips by Hour of Day"

    df_hist = parse_section(text, hist_section)
    df_hour = parse_section(text, hour_section)

    out_dir = Path(__file__).parent / "figures"
    plot_histogram_trips_per_taxi(df_hist, out_dir / "histogram_trips_per_taxi.png")
    plot_trips_by_hour(df_hour, out_dir / "trips_by_hour.png")
    print("Plots saved to", out_dir)


if __name__ == "__main__":
    main()