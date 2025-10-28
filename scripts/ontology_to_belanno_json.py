import gzip
import re
import pandas as pd
from rdflib import Graph, RDFS, Namespace, URIRef, Literal
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import requests
import tempfile
import os
import json
import zipfile
import obonet
import xml.etree.ElementTree as ET


def detect_format(file_path: Path):
    """
    Detect the RDF format based on file extension and content.
    """
    extension = file_path.suffix.lower()

    # Format mapping for common ontology file extensions
    format_map = {
        ".owl": "xml",
        ".rdf": "xml",
        ".xml": "xml",
        ".ttl": "turtle",
        ".turtle": "turtle",
        ".n3": "n3",
        ".nt": "nt",
        ".ntriples": "nt",
        ".jsonld": "json-ld",
        ".json": "json-ld",
        ".obo": "obo",  # Note: rdflib doesn't natively support OBO format
        ".trig": "trig",
        ".nq": "nquads",
        ".nquads": "nquads",
        ".gz": "gz",
        ".txt": "txt",
    }

    return format_map.get(extension)  # Default to XML if unknown


def sanitize_xml_content(content: bytes) -> bytes:
    """
    Remove invalid xml:lang attributes and any illegal characters in XML content.
    """
    text = content.decode("utf-8", errors="replace")
    # Remove xml:lang attributes with invalid values (anything not matching RFC 5646)
    text = re.sub(r'xml:lang="[^"]*"', "", text)
    # Remove control characters that break XML parsing
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return text.encode("utf-8")


def download_drugbank(url: str, username: str, password: str):
    """
    Download and parse the DrugBank RDF (all-full-database) ZIP file using basic authentication.
    """

    print(f"🔑 Downloading DrugBank release .")

    # Send HTTP GET request with Basic Auth
    response = requests.get(url, auth=(username, password), stream=True)
    if response.status_code == 401:
        raise PermissionError(
            "❌ Authentication failed. Check your DrugBank credentials."
        )
    response.raise_for_status()

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_path = Path(tmp_file.name)

    print(f"✅ Downloaded DrugBank data to: {tmp_path}")

    # Optionally extract the RDF/Turtle/XML file inside the ZIP
    with zipfile.ZipFile(tmp_path, "r") as zip_ref:
        extract_dir = Path(tempfile.mkdtemp())
        zip_ref.extractall(extract_dir)

    print(f"📂 Extracted contents to: {extract_dir}")

    # Find RDF or TTL file inside
    rdf_files = (
        list(extract_dir.glob("*.ttl"))
        + list(extract_dir.glob("*.xml"))
        + list(extract_dir.glob("*.rdf"))
    )
    if not rdf_files:
        raise FileNotFoundError(
            "❌ No RDF/Turtle/XML file found in the DrugBank ZIP archive."
        )

    xml_path = rdf_files[0]
    print(f"🧬 Found ontology file: {xml_path.name}")

    print(f"🔍 Parsing DrugBank XML: {xml_path}")

    # Parse XML incrementally (efficient for large files)
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)  # get root element

    terms = []

    for event, elem in tqdm(context, desc="Processing drugs", unit="drug"):
        if event == "end" and elem.tag == "{http://www.drugbank.ca}drug":
            # Extract DrugBank IDs (primary + secondary)
            ids = [
                id_elem.text
                for id_elem in elem.findall("{http://www.drugbank.ca}drugbank-id")
            ]
            if not ids:
                continue
            primary_id = ids[0]

            # Extract the drug's name
            name_elem = elem.find("{http://www.drugbank.ca}name")
            if name_elem is not None and name_elem.text:
                terms.append((name_elem.text.strip(), primary_id))

            # Clear element to free memory
            elem.clear()
            root.clear()

    print(f"✅ Extracted {len(terms)} drugs.")
    # Clean up temporary files
    tmp_path.unlink(missing_ok=True)
    for item in extract_dir.iterdir():
        item.unlink(missing_ok=True)
    extract_dir.rmdir()

    return terms


def parse_ontology(file_url: str, value_column="", code_column="", header=None):
    """
    Parse ontology file from a URL with automatic format detection and cleanup.
    Downloads the file to a temporary location, parses it, then deletes it.
    """
    print(f"\nDownloading ontology from: {file_url}")

    response = requests.get(file_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    # Create a temporary file
    suffix = os.path.splitext(file_url)[-1]
    if len(suffix) > 5 or not suffix:
        suffix = ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = Path(tmp_file.name)

    if not tmp_path.exists():
        raise FileNotFoundError(f"Temporary ontology file not found: {tmp_path}")

    g = Graph()
    detected_format = detect_format(tmp_path)
    print(f"Detected format: {detected_format}")
    try:
        if detected_format is None:
            if "text/plain" in content_type or tmp_path.suffix.lower() == ".list":
                detected_format = "txt"
            else:
                detected_format = "xml"
        print(f"Using format: {detected_format}")
        if detected_format == "gz" or detected_format == "txt":

            if detected_format == "gz":
                with gzip.open(tmp_path, "rt", encoding="utf-8") as f:
                    df = pd.read_csv(f, sep=",", dtype=str, header=header)

                    if header is None:
                        value_column = df.columns[1]
                        code_column = df.columns[0]
                    else:
                        if (
                            value_column not in df.columns
                            or code_column not in df.columns
                        ):
                            raise ValueError(
                                f"Columns '{value_column}' or '{code_column}' not found in the file."
                            )

            elif detected_format == "txt":
                # find the first tab-delimited header line
                with open(tmp_path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if line.startswith("#") or not line.strip():
                            continue
                        if "\t" in line:  # found header line
                            skip_rows = i
                            break
                df = pd.read_csv(
                    tmp_path,
                    sep="\t",
                    header=0,
                    dtype=str,
                    comment="#",
                    on_bad_lines="skip",
                    skiprows=skip_rows,
                )
                if not value_column or not code_column:
                    raise ValueError(
                        "Either 'value_column' or 'code_column' or both not specified for gz or txt files."
                    )
            g = df[[value_column, code_column]].drop_duplicates()
        elif detected_format == "xml":
            print("Trying to sanitize malformed RDF/XML content...")
            cleaned_content = sanitize_xml_content(response.content)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as clean_file:
                clean_file.write(cleaned_content)
                clean_path = Path(clean_file.name)
            try:
                g.parse(clean_path, format="xml")
                print("Successfully parsed after XML sanitization.")
            except Exception as e2:
                print(f"Failed after sanitization: {e2}")
                raise e2
            finally:
                clean_path.unlink(missing_ok=True)
        elif detected_format == "obo":
            print("🧬 OBO format detected — parsing via obonet...")
            try:
                # ensure it's installed: pip install obonet

                obo_graph = obonet.read_obo(tmp_path)
                print(f"✅ Loaded {len(obo_graph.nodes)} OBO terms.")
                # Convert OBO to RDF Graph for consistency
                from rdflib import URIRef, RDFS, Literal

                g = Graph()
                for node, data in obo_graph.nodes(data=True):
                    label = data.get("name")
                    if label:
                        g.add(
                            (
                                URIRef(f"http://purl.obolibrary.org/obo/{node}"),
                                RDFS.label,
                                Literal(label),
                            )
                        )
                print(f"✅ Converted {len(g)} triples from OBO graph.")
            except Exception as e:
                raise Exception(f"❌ Failed to parse OBO file via obonet: {e}")
        else:
            try:
                g.parse(tmp_path, format=detected_format)
                print(f"Successfully parsed using {detected_format} format.")
            except Exception as e1:
                print(f"Failed with detected format: {e1}")
                fallback_formats = ["xml", "turtle", "n3", "nt", "json-ld"]
                for fmt in fallback_formats:
                    if fmt != detected_format:
                        try:
                            print(f"Trying fallback format: {fmt}")
                            g.parse(tmp_path, format=fmt)
                            print(f"Successfully parsed using fallback format: {fmt}")
                            break
                        except Exception as e2:
                            print(f"Failed with {fmt}: {e2}")
                else:
                    raise Exception(
                        f"Could not parse file with any format. Last error: {e1}"
                    )
        return g
    finally:
        # Always delete temporary file
        tmp_path.unlink(missing_ok=True)


# ---------------------- MAIN SCRIPT ----------------------
def main(input_dictionary, author, contact_info, output_dir):

    # Get input parameters

    ontology_url = input_dictionary.get("URL")
    if not ontology_url:
        raise ValueError("The ontology_url is missing in the provided JSON file.")

    Annotation = input_dictionary.get("Annotation")
    if not Annotation:
        raise ValueError("The Annotation is missing in the provided JSON file.")

    print(f"\nWorking on Annotation:", Annotation)

    description = input_dictionary.get("DescriptionString")
    if not description:
        description = "No description provided."

    NameString = input_dictionary.get("NameString")
    if not NameString:
        NameString = Annotation

    UsageString = input_dictionary.get("UsageString")
    if not UsageString:
        UsageString = "other"

    VersionString = input_dictionary.get("VersionString")
    if not VersionString:
        VersionString = f"{datetime.now().strftime('%Y%m%d')}"

    value_column = input_dictionary.get("value_column")
    code_column = input_dictionary.get("code_column")
    header = input_dictionary.get("header", 0)
    if not header:
        header = None

    subcategories = input_dictionary.get("subcategories")

    # Parse the ontology
    if "drugbank.com" in ontology_url:
        username = input("Enter DrugBank username: ").strip()
        password = input("Enter DrugBank password: ").strip()
        terms = download_drugbank(ontology_url, username, password)
    else:
        try:
            g = parse_ontology(ontology_url, value_column, code_column, header)
            print("Ontology loaded successfully.")
            print(f"Ontology contains {len(g)} triples.")
        except Exception as e:
            print(f"Error loading ontology: {e}")
            exit(1)

        print("=" * 50)
        print("Extracting terms...")

        # Special handling for DrugBank with authentication

        # Handle DataFrame case (for gz or txt files)
        if isinstance(g, pd.DataFrame):
            terms = []
            for _, row in g.iterrows():
                label = str(row.iloc[0])
                code_value = str(row.iloc[1])
                terms.append((label, code_value))

        # Extract labels and URIs for relevant terms
        elif subcategories and Annotation.startswith("GO"):
            terms = []
            for s, p, o in tqdm(
                g.triples((None, RDFS.label, None)), desc="Processing triples"
            ):
                label = str(o)
                terms.append((label, str(s)))
            print(f"✅ Extracted {len(terms)} MeSH terms.")
            # Generate belanno file
            generate_file(
                input_dictionary,
                terms,
                Annotation,
                NameString,
                UsageString,
                VersionString,
                description,
                author,
                contact_info,
                output_dir,
            )

            for sub, data in subcategories.items():

                prefix, desc = data[0], data[1]
                print(f"\nProcessing subcategory: {desc}")
                sub_dict = input_dictionary.copy()
                Annotation = f"{sub_dict['Annotation']}{sub}"
                print(f"\nWorking on Annotation:", Annotation)
                NameString = f"{sub_dict['NameString']} ({sub})"
                description = f"{sub_dict.get('DescriptionString', '')} - {desc}"
                BP_ROOT = URIRef(f"http://purl.obolibrary.org/obo/{prefix}")
                descendants = set()
                queue = [BP_ROOT]
                while queue:
                    current = queue.pop()
                    for subclass in g.subjects(RDFS.subClassOf, current):
                        if subclass not in descendants:
                            descendants.add(subclass)
                            queue.append(subclass)

                # Extract labels
                terms = []
                for uri in descendants:
                    for _, _, label in g.triples((uri, RDFS.label, None)):
                        if isinstance(label, Literal):
                            terms.append((str(label), uri.split("/")[-1]))

                print(f"✅ Extracted {len(terms)} {desc} terms.")
                generate_file(
                    sub_dict,
                    terms,
                    Annotation,
                    NameString,
                    UsageString,
                    VersionString,
                    description,
                    author,
                    contact_info,
                    output_dir,
                )
            return

        elif subcategories and Annotation.startswith("MESH"):
            terms = []
            for s, p, o in tqdm(
                g.triples((None, RDFS.label, None)), desc="Processing triples"
            ):
                label = str(o)
                terms.append((label, str(s)))
            print(f"✅ Extracted {len(terms)} MeSH terms.")

            for sub, data in subcategories.items():

                print(f"\nProcessing subcategory: {sub}")

                sub_dict = input_dictionary.copy()
                Annotation = f"{sub_dict['Annotation']}{sub}"
                print(f"Working on Annotation: {Annotation}")

                NameString = f"{sub_dict['NameString']} ({sub})"
                description = f"{sub_dict.get('DescriptionString', '')} - {sub}"

                # Reset terms list for each subcategory

                MESHV = Namespace("http://id.nlm.nih.gov/mesh/vocab#")

                roots = set()
                TREE_NUMBER_PRED = MESHV.treeNumber
                TREE_LABEL = RDFS.label

                terms = []

                for prefix in data:
                    print(
                        f"🔍 Finding root descriptors for tree number prefix: {prefix}"
                    )
                    # Step 1: For each descriptor linked to a tree number node
                    for descriptor, _, tree_node in g.triples(
                        (None, TREE_NUMBER_PRED, None)
                    ):
                        # Step 2: Get the label of the tree node
                        for _, _, label in g.triples((tree_node, TREE_LABEL, None)):
                            tree_num = str(label)
                            if tree_num.startswith(prefix):
                                roots.add(descriptor)

                    print(f"🌳 Found {len(roots)} root descriptors for prefix {prefix}")
                    visited = set()

                    queue = list(roots)

                    while queue:
                        node = queue.pop()
                        if node in visited:
                            continue
                        visited.add(node)

                        label = g.value(subject=node, predicate=RDFS.label)
                        if isinstance(label, Literal):
                            mesh_id = node.split("/")[-1]
                            terms.append((str(label), mesh_id))

                        # 🔁 Traverse deeper through narrowerDescriptor relations
                        for narrower in g.objects(node, MESHV.narrowerDescriptor):
                            if narrower not in visited:
                                queue.append(narrower)

                    print(
                        f"✅ Extracted {len(terms)} MeSH terms for category '{sub}' (prefix {prefix})."
                    )

                generate_file(
                    sub_dict,
                    terms,
                    Annotation,
                    NameString,
                    UsageString,
                    VersionString,
                    description,
                    author,
                    contact_info,
                    output_dir,
                )
            return

        else:
            terms = []
            for s, p, o in tqdm(
                g.triples((None, RDFS.label, None)), desc="Processing triples"
            ):
                label = str(o)
                terms.append((label, str(s)))

        # Try alternative label properties if no RDFS labels found
        if not terms:
            print("No RDFS labels found. Trying alternative label properties...")
            label_properties = [
                URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
                URIRef("http://purl.obolibrary.org/obo/IAO_0000111"),
                URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
                URIRef("http://xmlns.com/foaf/0.1/name"),
            ]
            for label_prop in label_properties:
                for s, p, o in g.triples((None, label_prop, None)):
                    label = str(o)
                    terms.append((label, str(s)))
                if terms:
                    print(f"Found {len(terms)} terms using {label_prop}")
                    break

        # Fallback: extract all subjects as labels
        if not terms:
            print(
                "Warning: No labeled terms found. Extracting all subjects as terms..."
            )
            subjects = {
                s for s, p, o in g.triples((None, None, None)) if isinstance(s, URIRef)
            }
            for s in subjects:
                label = str(s).split("/")[-1].split("#")[-1]
                terms.append((label, str(s)))

        print("Extraction complete.")
        print(f"Found {len(terms)} terms.")

    # Generate belanno file
    generate_file(
        input_dictionary,
        terms,
        Annotation,
        NameString,
        UsageString,
        VersionString,
        description,
        author,
        contact_info,
        output_dir,
    )


def generate_file(
    input_dictionary,
    terms,
    Annotation,
    NameString,
    UsageString,
    VersionString,
    description,
    author,
    contact_info,
    output_dir,
):
    print("=" * 50)
    print("Generating belanno file...")

    # Create belanno content

    # Check for illegal delimiter in labels
    DelimiterString = "|"
    if any(DelimiterString in label for label, _ in terms):
        DelimiterString = "§"
        print(
            f"Delimiter '|' found in labels. Switching to delimiter: {DelimiterString}"
        )
    else:
        print(f"Using delimiter: {DelimiterString}")

    # Construct metadata section
    metadata = f"""
[AnnotationDefinition]
Keyword={Annotation}
TypeString=list
DescriptionString={description}
UsageString={UsageString}
VersionString={VersionString}
CreatedDateTime={datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}


[Author]
NameString={author}
CopyrightString=Creative Commons by 4.0
ContactInfoString={contact_info}

[Citation]
NameString={NameString}

[Processing]
CaseSensitiveFlag=no
DelimiterString={DelimiterString}
CacheableFlag=yes

[Values]
"""

    # Construct values section
    code = input_dictionary.get("code")
    if code:
        values = "\n".join(
            f"{label}{DelimiterString}{code}"
            for label, uri in dict(sorted(terms)).items()
        )
    else:
        values_list = []
        for label, uri in dict(sorted(terms)).items():
            code = uri.split("/")[-1]
            if len(code) > 8:
                code = "OGRPBCAM"
            values_list.append(f"{label}{DelimiterString}{code}")
        values = "\n".join(values_list)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save belanno file
    output_file = output_path / f"{Annotation}_{VersionString}.belanno"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(metadata + values)

    print("belanno file created successfully.")
    print(f"Output file: {output_file}")
    print(f"Total terms exported: {len(terms)}")


if __name__ == "__main__":
    json_file = input("Enter the path to the input JSON file : ").strip()
    author = input("Enter the author name: ").strip()
    contact_info = input("Enter the contact information: ").strip()
    output_dir = (
        input("Enter the output directory (default is current directory): ").strip()
        or "."
    )
    print("Universal Ontology to belanno Converter")
    print("Supports: OWL, RDF, TTL, NT, N3, JSON-LD, OBO, and more")
    print("=" * 60)
    dictionary_list = json.load(open(json_file))
    for dictionary in tqdm(dictionary_list):
        main(dictionary, author, contact_info, output_dir)
