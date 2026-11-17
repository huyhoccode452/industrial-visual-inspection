import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # TODO: vì sao parents[1]?
DEFAULT_CATEGORIES = ["bottle", "cable", "metal_nut", "screw", "capsule"]


def parse_args():
    p = argparse.ArgumentParser(description="Extract selected MVTec AD categories.")
    p.add_argument("--zip", type=Path, required=True, help="Duong dan toi mvtec-ad.zip")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "datasets" / "mvtec_ad", help="Thu muc luu du lieu")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="Cac category can lay")
    return p.parse_args()


def select_members(names, category):
    result = []
    for name in names:
        parts = name.split("/")
        if  parts[-1] != "" and category in parts[:-1]:      # điều kiện 1: là file; điều kiện 2: category nằm trong thư mục
            result.append(name)
    return result


def main():
    args = parse_args()
    if not args.zip.exists():
        raise SystemExit(
            f"Khong tim thay file zip: {args.zip}\n"
            "Hay tai MVTec AD tu trang chinh thuc, roi truyen duong dan bang --zip"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Se luu du lieu vao: {args.out}")
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        for category in args.categories:
            members = select_members(names, category)
            if not members:
                raise SystemExit(f"Khong co file nao cho category '{category}'. Kiem tra lai ten.")
            print(f"Dang giai nen {category}: {len(members)} file")
            for member in members:
                zf.extract(member, args.out)
    print("Xong!")

if __name__ == "__main__":
    main()