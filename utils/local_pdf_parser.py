import os
import json
import streamlit as st

st.set_page_config(layout="wide")


def parse_for_dataset(pdf):
    data = {"Font": [], "Size": [], "Text": []}

    fonts = set()
    for page in pdf:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    data["Font"].append(span["font"])
                    data["Size"].append(span["size"])
                    data["Text"].append(span["text"])
                    fonts.add(span["size"])
    font_sizes = sorted(fonts, reverse=True)

    font_counter = {}

    for font_size in font_sizes:
        font_counter[font_size] = 0

        for page in pdf:
            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] == font_size:
                            font_counter[font_size] += 1

    largest_font = max(font_sizes)

    category = ""
    categories = {}
    product_headings = []
    product = ""
    data = ""
    products_data = []
    meta_data1 = []
    meta_data2 = []
    specs = {}
    images = []
    for page_index, page in enumerate(pdf):
        page_dict = page.get_text("dict")

        for block_index, block in enumerate(page_dict["blocks"]):
            for line_index, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    if "price:" in span["text"].lower():
                        specs["price"] = span["text"].lower().replace("price:", "")

                    if "material:" in span["text"].lower():
                        specs["material"] = (
                            span["text"].lower().replace("material:", "")
                        )

                    if "warranty:" in span["text"].lower():
                        specs["warranty"] = (
                            span["text"].lower().replace("warranty:", "")
                        )

                    if "colors:" in span["text"].lower():
                        specs["colors"] = span["text"].lower().replace("colors:", "")

                    if span["size"] == largest_font:
                        if product != "":
                            categories[category].append({product: data.strip()})
                            products_data.append(data)
                        category = span["text"]
                        categories[category] = []
                        product = ""
                        data = ""
                    elif span["size"] == 10:
                        if span["flags"] & 16:
                            if span["text"].lower() not in [
                                "specifications",
                                "ideal for",
                            ]:
                                if specs:
                                    meta_data2.append(specs)
                                if product != "":
                                    categories[category].append({product: data})

                                product = span["text"]
                                for file_name in os.listdir(f"images/{category}"):
                                    if product in file_name:
                                        images.append(f"images/{category}/{file_name}")
                                product_headings.append(span["text"])

                                meta_data1.append(
                                    {
                                        "category": category,
                                        "product": product,
                                        "page": page_index + 1,
                                    }
                                )
                                specs = {}
                                if data != "":
                                    products_data.append(data)
                                data = ""
                        if block_index == 1:
                            continue
                        data += span["text"] + "\n"

    products_data.append(data)
    if product != "":
        categories[category].append({product: data})
    meta_data2.append(specs)

    # st.write(products_data)
    col1, col2 = st.columns(2)
    meta_data = [meta_data1[i] | meta_data2[i] for i in range(len(product_headings))]

    final_document = []

    for i in range(len(product_headings)):
        final_document.append(
            {
                "id": i,
                "text": products_data[i],
                "metadata": meta_data[i],
                "image": images[i],
            }
        )

    with open("catalog dataset.json", "w") as f:
        json.dump(final_document, f, indent=4)

    return final_document
