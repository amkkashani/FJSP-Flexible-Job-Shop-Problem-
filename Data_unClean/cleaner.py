import pandas as pd
from pathlib import Path
from datetime import datetime


def load_clean(path: str) -> pd.DataFrame:
    """
    Load and clean an Excel file with specific column mappings and transformations.

    Args:
        path: Path to the Excel file to clean

    Returns:
        Cleaned DataFrame with standardized columns
    """
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    if "ElemIdent" in df.columns:
        df = df[df["ElemIdent"].astype(str).str.strip().ne("ElemIdent")].copy()

    mapping = {
        "Mat": "mat",
        "Teilbez": "code",
        "Flaenge": "length",
        "Fbreite": "width",
        "Stueck": "quantity",
        "Unnamed: 15": "ParentWareCode",
        "WA": "wa",
        "WF": "wf",
        "WD": "wd",
        "WO": "wo",
        "WG": "wg",
        "WV": "wv",
        "WX": "wx",
    }

    df = df.rename(columns=mapping)

    out_cols = [
        "ElemIdent",
        "mat",
        "code",
        "length",
        "width",
        "quantity",
        "ParentWareCode",
        "Info8",
        "مساحت",
        "wa",
        "wf",
        "wd",
        "wo",
        "wg",
        "wv",
        "wx",
    ]

    missing = [c for c in out_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input file: {missing}")

    df_out = df[out_cols].copy()
    row_ids = [f"row_{i:05d}" for i in range(1, len(df_out) + 1)]
    df_out.insert(0, "row_id", row_ids)

    # Rename مساحت → area
    df_out = df_out.rename(columns={"مساحت": "area"})

    for c in ["length", "width", "quantity", "area", "wa", "wf", "wd", "wo", "wg", "wv", "wx"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    return df_out


def process_all_xlsx_files(directory: str = "."):
    """
    Process all xlsx files in the directory that don't have 'clean' in their name.

    Args:
        directory: Directory path to search for xlsx files (default: current directory)

    Returns:
        List of dictionaries containing processing results for each file
    """
    data_dir = Path(directory)

    # Find all xlsx files that don't contain 'clean' in their name
    xlsx_files = [
        f for f in data_dir.glob("*.xlsx")
        if "clean" not in f.name.lower() and not f.name.startswith("~$")
    ]

    total_files = len(xlsx_files)
    print(f"\n{'='*60}")
    print(f"Found {total_files} xlsx files to process")
    print(f"{'='*60}\n")

    processed_count = 0
    failed_count = 0
    results = []

    for idx, file_path in enumerate(xlsx_files, 1):
        print(f"[{idx}/{total_files}] Processing: {file_path.name}")
        print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Load and clean the data
            cleaned_df = load_clean(str(file_path))

            # Create output filename
            output_filename = file_path.stem + "_cleaned.xlsx"
            output_path = data_dir / output_filename

            # Save the cleaned data
            cleaned_df.to_excel(output_path, index=False)

            print(f"  ✓ Successfully cleaned: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
            print(f"  ✓ Saved to: {output_filename}")
            processed_count += 1

            results.append({
                "file": file_path.name,
                "status": "success",
                "rows": cleaned_df.shape[0],
                "output": output_filename
            })

        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {str(e)}")
            failed_count += 1
            results.append({
                "file": file_path.name,
                "status": "failed",
                "error": str(e)
            })

        print()

    # Summary
    print(f"{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    # Process all xlsx files in the current directory
    results = process_all_xlsx_files()

    # Optional: Display detailed results
    print("Detailed Results:")
    for result in results:
        if result["status"] == "success":
            print(f"  ✓ {result['file']} → {result['output']} ({result['rows']} rows)")
        else:
            print(f"  ✗ {result['file']} → Error: {result['error']}")
