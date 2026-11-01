import re
from datetime import datetime


def extract_page_info(filename):
    """
    Extract the publication date and page number
    from a newspaper filename.

    Example:
    1956-01-05_page_01.jpg
    """

    filename = str(filename).split("/")[-1]
    
    match = re.match(
        r"(\d{4}-\d{2}-\d{2})_page_(\d+)",
        filename
    )

    if not match:
        raise ValueError(
            f"Unexpected newspaper filename: {filename}"
        )

    publication_date = datetime.strptime(
        match.group(1), "%Y-%m-%d"
    ).date().isoformat()

    page_number = int(match.group(2))

    return publication_date, page_number


def create_crop_metadata(
    source_filename,
    crop_filename,
    category,
    confidence,
    box
):
    """
    Create metadata for one detected and saved crop.
    """

    publication_date, page_number = extract_page_info(
        str(source_filename)
    )

    x1, y1, x2, y2 = box

    return {
        "source_filename": source_filename,
        "publication_date": publication_date,
        "page_number": page_number,
        "crop_filename": crop_filename,
        "category": category,
        "confidence": round(float(confidence), 6),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }