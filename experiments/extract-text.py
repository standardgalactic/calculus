#!/usr/bin/env python3
"""
Extract plain text from PDF, EPUB, and MHTML files.

Usage:

    python extract_text.py
        Process all supported files in current directory.

    python extract_text.py -r
        Process all supported files recursively.

    python extract_text.py -m
        Process only .mht/.mhtml files.

    python extract_text.py -m -r
        Process only .mht/.mhtml files recursively.

    python extract_text.py -m -r -s
        Process only .mht/.mhtml files recursively,
        skipping files whose .txt output already exists.

Dependencies:

    pip install PyMuPDF ebooklib beautifulsoup4
"""

import os
import argparse
import fitz
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser


def normalize_quotes(text):
    """Replace smart quotes and related characters."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "—",
        "\u2013": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }

    for smart, plain in replacements.items():
        text = text.replace(smart, plain)

    return text


def is_supported_file(filename, mht_only=False):
    filename = filename.lower()

    if mht_only:
        return filename.endswith((".mht", ".mhtml"))

    return filename.endswith((".pdf", ".epub", ".mht", ".mhtml"))


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text("text") + "\n"

        doc.close()

    except Exception as e:
        print(f"[!] Error extracting text from {pdf_path}: {e}")

    return normalize_quotes(text)


def extract_text_from_epub(epub_path):
    text = ""

    try:
        book = epub.read_epub(epub_path)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text += soup.get_text() + "\n"

    except Exception as e:
        print(f"[!] Error extracting text from {epub_path}: {e}")

    return normalize_quotes(text)


def extract_text_from_mhtml(mhtml_path):
    text_parts = []

    try:
        with open(mhtml_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        def decode_part(part):
            charset = part.get_content_charset() or "utf-8"

            try:
                payload = part.get_payload(decode=True)

                if payload is None:
                    return ""

                return payload.decode(
                    charset,
                    errors="replace"
                )

            except Exception as e:
                print(f"[!] Decode error in {mhtml_path}: {e}")
                return ""

        if msg.is_multipart():

            for part in msg.walk():
                ctype = part.get_content_type()

                if ctype == "text/html":
                    html = decode_part(part)

                    if html:
                        soup = BeautifulSoup(html, "html.parser")
                        text_parts.append(
                            soup.get_text(
                                separator="\n",
                                strip=True
                            )
                        )

                elif ctype == "text/plain":
                    text_parts.append(decode_part(part))

        else:
            content = decode_part(msg)
            ctype = msg.get_content_type()

            if ctype == "text/html":
                soup = BeautifulSoup(content, "html.parser")
                text_parts.append(
                    soup.get_text(
                        separator="\n",
                        strip=True
                    )
                )
            else:
                text_parts.append(content)

    except Exception as e:
        print(f"[!] Error reading MHTML {mhtml_path}: {e}")

    return normalize_quotes("\n\n".join(text_parts))


def save_text(text, original_path):
    output_file = os.path.splitext(original_path)[0] + ".txt"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[✓] Text saved to: {output_file}")

    except Exception as e:
        print(f"[!] Error writing to {output_file}: {e}")


def process_file(file_path, skip_existing=False):
    if not os.path.isfile(file_path):
        print(f"[!] File not found: {file_path}")
        return

    output_file = os.path.splitext(file_path)[0] + ".txt"

    if skip_existing and os.path.exists(output_file):
        print(f"[↷] Skipping existing: {output_file}")
        return

    print(f"[•] Processing {file_path}...")

    lower = file_path.lower()

    if lower.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)

    elif lower.endswith(".epub"):
        text = extract_text_from_epub(file_path)

    elif lower.endswith((".mht", ".mhtml")):
        text = extract_text_from_mhtml(file_path)

    else:
        print(f"[!] Unsupported file type: {file_path}")
        return

    if text.strip():
        save_text(text, file_path)
    else:
        print(f"[!] No text extracted from: {file_path}")


def process_directory(
    directory,
    mht_only=False,
    recursive=False,
    skip_existing=False
):
    if recursive:

        for root, dirs, files in os.walk(directory):
            for filename in files:

                if is_supported_file(
                    filename,
                    mht_only
                ):
                    process_file(
                        os.path.join(root, filename),
                        skip_existing=skip_existing
                    )

    else:

        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)

            if (
                os.path.isfile(path)
                and is_supported_file(
                    filename,
                    mht_only
                )
            ):
                process_file(
                    path,
                    skip_existing=skip_existing
                )


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF, EPUB, and MHTML files."
    )

    parser.add_argument(
        "-m",
        "--mht-only",
        action="store_true",
        help="Process only .mht/.mhtml files."
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively."
    )

    parser.add_argument(
        "-s",
        "--skip-existing",
        action="store_true",
        help="Skip files whose .txt output already exists."
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to process."
    )

    args = parser.parse_args()

    if args.files:

        for file in args.files:

            if not is_supported_file(
                file,
                args.mht_only
            ):
                print(
                    f"[!] Skipping unsupported file: {file}"
                )
                continue

            process_file(
                file,
                skip_existing=args.skip_existing
            )

    else:

        if (
            args.mht_only
            and args.recursive
            and args.skip_existing
        ):
            print(
                "[i] Processing MHT/MHTML files recursively "
                "(skipping existing)..."
            )

        elif args.mht_only and args.recursive:
            print(
                "[i] Processing MHT/MHTML files recursively..."
            )

        elif args.mht_only:
            print(
                "[i] Processing MHT/MHTML files..."
            )

        elif args.recursive:
            print(
                "[i] Processing supported files recursively..."
            )

        else:
            print(
                "[i] Processing supported files in current directory..."
            )

        process_directory(
            os.getcwd(),
            mht_only=args.mht_only,
            recursive=args.recursive,
            skip_existing=args.skip_existing
        )


if __name__ == "__main__":
    main()
