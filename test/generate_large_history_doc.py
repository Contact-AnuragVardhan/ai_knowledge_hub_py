from docx import Document
import os
import random
import textwrap

# ----- CONFIG -----
TARGET_MB = 10            # Target file size in MB
OUTPUT_FILE = "NewYork_History_" + str(TARGET_MB) + "MB.docx"
# -------------------

print("Generating " + str(TARGET_MB) + "MB of historical text...")

# Base real history content
base_history = """
New York State and New York City have one of the richest historical narratives in the United States.
From Indigenous cultures such as the Haudenosaunee Confederacy and the Lenape nation,
to Dutch colonization under New Netherland, to British rule, the American Revolution,
the Erie Canal, Ellis Island immigration, rapid industrialization, Wall Street’s rise,
and New York City becoming a global cultural and financial capital—few regions have had as deep
an impact on American history.

Indigenous Era:
The Haudenosaunee Confederacy (Iroquois) formed one of the earliest democratic systems.
The Lenape people lived in present-day New York City, relying on sophisticated trade and diplomacy.

Dutch Settlement:
In 1609 Henry Hudson arrived. By 1624 the Dutch created New Netherland and built New Amsterdam.
The Dutch introduced multicultural trade, tolerance, early city planning, and open immigration.

British Rule:
In 1664 the English captured New Amsterdam and renamed it New York. It grew into a major port.

American Revolution:
New York was central to the war, including the Battle of Long Island and the pivotal Saratoga victory.
New York City served as the first capital of the United States in 1789.

19th Century:
The 1825 Erie Canal turned NYC into the gateway of American commerce. Upstate cities boomed.

Immigration:
Ellis Island processed more than 12 million immigrants between 1892 and 1954.

20th Century:
New York became the world’s finance capital (Wall Street), culture capital (Broadway),
and international diplomacy center (United Nations).

21st Century:
New York has faced 9/11, financial crises, and COVID-19 yet remains a global powerhouse.
"""

# Create filler generator (unique English paragraphs)
WORDS = [
    "development", "expansion", "migration", "commerce", "culture",
    "architecture", "innovation", "infrastructure", "industry", "heritage",
    "economic", "political", "transportation", "community", "urbanization"
]

def generate_filler_paragraph():
    paragraph = " ".join(random.choice(WORDS) for _ in range(250))
    return textwrap.fill(paragraph, width=90)

# Build large text in memory (FAST)
target_bytes = TARGET_MB * 1024 * 1024
text_chunks = []

current_size = 0
base_bytes = len(base_history.encode("utf-8"))

print("Building text blocks...")

while current_size < target_bytes:
    text_chunks.append(base_history)
    current_size += base_bytes

    # add 5 unique filler paragraphs each cycle
    for _ in range(5):
        p = generate_filler_paragraph()
        text_chunks.append(p)
        current_size += len(p.encode("utf-8"))

# Join into one giant string (VERY FAST, avoids docx slow loops)
final_text = "\n\n".join(text_chunks)

print(f"Final text size = {len(final_text)/1024/1024:.2f} MB")
print("Writing DOCX (single write operation)...")

# Create DOCX with one huge paragraph (SUPER FAST)
doc = Document()
doc.add_paragraph(final_text)
doc.save(OUTPUT_FILE)

print("DONE!")
print(f"Saved file: {os.path.abspath(OUTPUT_FILE)}")
