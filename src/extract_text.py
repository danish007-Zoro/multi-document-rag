import pymupdf
def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path (str): Path to the PDF.

    Returns:
        list: A list where each element contains the text of one page.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_num in range(document.page_count):
        page = document[page_num]
        pages.append(page.get_text())

    document.close()

    return pages