# generate_ml_data.py
import os
import sys
import django
import csv
import random
import json

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define path to the configuration file
DATA_JSON_PATH = os.path.join(script_dir, 'data.json')

# Initialize variables for paths that will be loaded from configuration
nesis_project_full_path = None
csv_directory = None
csv_file_name = None

# Try to load configuration from data.json
try:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)
        # Extract paths from configuration
        nesis_project_full_path = data.get('nesis_project_full_path')
        csv_directory = data.get('csv_directory')
        csv_file_name = data.get('csv_file_name')

    # Verify required configuration exists
    if not nesis_project_full_path:
        raise ValueError("'nesis_project_full_path' not found in data.json")
    print(f"Loaded nesis_project_full_path: {nesis_project_full_path}")
except FileNotFoundError:
    print(f"Error: data.json not found at {DATA_JSON_PATH}. Please create it with 'nesis_project_full_path'.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: data.json at {DATA_JSON_PATH} is not a valid JSON file.")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Add the Django project path to Python's import path
if nesis_project_full_path not in sys.path:
    sys.path.insert(0, nesis_project_full_path)
# --- END IMPORTANT ADDITIONS ---


# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nesis_project.settings')
try:
    django.setup()
except Exception as e:
    print(f"Error setting up Django: {e}")
    print(
        f"Please ensure '{nesis_project_full_path}' is the correct path to your Django project root and 'nesis_project.settings' is correct.")
    sys.exit(1)

# Import Django models after Django is set up
from nesis_app.models import Application, Project, Product, Document_Media, Mission, Theme, Geography
from django.db.models import Q

# --- Configuration ---
OUTPUT_CSV_FILE = os.path.join(csv_directory, csv_file_name)
MAX_EXAMPLES_PER_ENTITY = 100
MIN_ITEMS_PER_TYPE = 50

# Mapping from model classes to their result type string
MODEL_TO_RESULT_TYPE = {
    Application: "application",
    Project: "project",
    Product: "product",
    Document_Media: "document",
    Mission: "mission",
}

# --- Configuration for Additional Theme-focused Queries (from generate_more_to_help_theme.py) ---
NUM_ADDITIONAL_QUERIES_TO_GENERATE = 500  # Adjusted for clarity

themes_for_additional_data = [  # Renamed for clarity from 'themes' to avoid conflict
    "Climate", "Carbon Management", "Disasters", "Agriculture",
    "Biodiversity and Ecological Conservation", "Health and Air Quality",
    "Water Resources", "Coasts and Oceans", "Cryosphere",
    "Weather and Atmosphere", "Energy Resources", "Population Dynamics and Settlements",
    "Ecological Conservation", "Earth Surface and Interior",
    "Sun Earth Interactions", "Wildland Fires"
]

result_types_for_additional_data = ["application", "project", "product", "document"]  # Renamed for clarity
result_type_phrases_for_additional_data = [  # Renamed for clarity
    lambda rt: f"show me {rt}",
    lambda rt: f"find {rt}",
    lambda rt: f"list {rt}",
    lambda rt: f"I need {rt}",
    lambda rt: f"query for {rt}",
    lambda rt: f"what {rt} are there",
    lambda rt: f"tell me about {rt}",
    lambda rt: f"get {rt}"
]

theme_preps_for_additional_data = [  # Renamed for clarity
    "in", "on", "about the theme of", "addressing", "related to the theme of",
    "regarding the theme of", "within the scope of", "focusing on the area of",
    "categorized under", "under the theme of", "concerned with",
    "that are about the theme of", "pertaining to the theme of",
    "aligned with", "belonging to the domain of"
]

search_term_preps_for_additional_data = [  # Renamed for clarity
    "about", "for", "related to", "concerning", "dealing with", "pertaining to",
    "looking for", "searching for", "details on", "keywords like",
    "information on", "data regarding", "specifics on", "analysis of"
]

generic_search_terms_for_additional_data = [  # Renamed for clarity
    "satellite imagery", "remote sensing data", "modeling tools", "prediction models",
    "sensor networks", "data visualization", "environmental impacts", "policy analysis",
    "risk assessment", "early warning systems", "community resilience", "sustainable practices",
    "resource management", "urban planning", "atmospheric chemistry", "ocean currents",
    "ice melt", "crop yield", "disease outbreaks", "energy efficiency", "population growth",
    "land use change", "solar flares", "wildfire prevention", "flood prediction",
    "drought monitoring", "biodiversity loss", "pollution detection", "air quality forecasting",
    "coastal erosion", "tsunami warnings", "ecosystem services", "habitat mapping",
    "resource conservation", "energy consumption", "extreme temperatures", "natural hazards",
    "carbon sequestration methods", "urban heat islands"
]

specific_search_terms_by_theme_for_additional_data = {  # Renamed for clarity
    "Climate": ["global warming", "sea level rise", "carbon emissions", "deforestation rates",
                "renewable energy technologies", "climate models"],
    "Carbon Management": ["carbon sequestration techniques", "emission reduction strategies",
                          "carbon footprint analysis"],
    "Disasters": ["earthquake preparedness", "flood forecasting", "tsunami warnings", "volcanic eruption impacts",
                  "drought relief efforts", "hurricane recovery plans"],
    "Agriculture": ["crop rotation methods", "soil health indicators", "food security analysis",
                    "precision farming techniques"],
    "Biodiversity and Ecological Conservation": ["species monitoring programs", "habitat restoration projects",
                                                 "deforestation impacts on ecosystems"],
    "Health and Air Quality": ["air pollution levels", "respiratory diseases prevalence",
                               "environmental health risks assessments"],
    "Water Resources": ["water scarcity solutions", "groundwater depletion rates", "irrigation techniques efficiency",
                        "water quality monitoring"],
    "Coasts and Oceans": ["coral bleaching causes", "sea level rise impact on coastal cities",
                          "ocean acidification effects"],
    "Cryosphere": ["glacier retreat observations", "arctic ice melt patterns", "permafrost thawing impacts"],
    "Weather and Atmosphere": ["storm prediction models", "atmospheric pressure changes", "weather patterns analysis",
                               "extreme weather event mitigation"],
    "Energy Resources": ["fossil fuel alternatives", "solar energy production efficiency",
                         "wind power development challenges"],
    "Population Dynamics and Settlements": ["urban expansion trends", "migration patterns analysis",
                                            "rural development initiatives"],
    "Ecological Conservation": ["wildlife protection laws", "ecosystem services valuation methods",
                                "deforestation monitoring tools"],
    "Earth Surface and Interior": ["geological mapping techniques", "seismic activity prediction",
                                   "volcanic monitoring systems"],
    "Sun Earth Interactions": ["space weather forecasting", "solar storm impacts", "geomagnetic activity studies"],
    "Wildland Fires": ["fire behavior models", "smoke dispersion patterns", "fuel mapping for prevention",
                       "post-fire recovery strategies"]
}


# --- Data Generation Logic (Original generate_ml_data.py content) ---

def generate_annotated_queries_from_db():
    """
    Generate training data by querying the database and creating labeled examples.
    This function creates a diverse set of natural language queries based on actual database content.
    It generates queries about items' names, descriptions, themes, locations, and missions.

    Returns:
        list: List of dictionaries, where each dictionary contains:
            - Query: The generated natural language question
            - Intent: Always 'search_query' for this implementation
            - RESULT_TYPE: The type of item being queried (application, project, etc.)
            - THEME: Theme name if applicable
            - MISSION_NAME: Mission name if applicable
            - COUNTRY_NAME: Country name if applicable
            - STATE_NAME: State name if applicable
            - SEARCH_TERM: Specific term being searched for
    """

    # List to store all generated queries and their annotations
    annotated_data = []

    # Process each model type (Application, Project, Product, etc.)
    for model_class, result_type_str in MODEL_TO_RESULT_TYPE.items():
        print(f"Generating queries for {result_type_str} from DB...")
        # Get random items from database
        items = list(model_class.objects.all().order_by('?')[:MAX_EXAMPLES_PER_ENTITY])

        # Warning if not enough items found
        if len(items) < MIN_ITEMS_PER_TYPE and model_class != Mission:
            print(
                f"  Warning: Only {len(items)} {result_type_str}s found. Consider increasing MIN_ITEMS_PER_TYPE or adding more data.")

        # Process each database item
        for item in items:
            base_annotation = {
                'Query': '', 'Intent': 'search_query', 'RESULT_TYPE': result_type_str,
                'THEME': '', 'MISSION_NAME': '', 'COUNTRY_NAME': '', 'STATE_NAME': '', 'SEARCH_TERM': '',
            }

            # Try to get the item's name (different models might use different field names)
            name_field = getattr(item, 'project_name', getattr(item, 'name', None))
            if not name_field:
                name_field = getattr(item, 'title', None)

            # Generate queries using the item's name
            if name_field:
                # Create different variations of queries using the name
                queries = [
                    f"Find the {result_type_str} called \"{name_field}\".",
                    f"Show me {result_type_str} \"{name_field}\".",
                    f"Search for {name_field} {result_type_str}.",
                ]
                # Add each query variation with its annotations
                for q in queries:
                    row = base_annotation.copy()
                    row['Query'] = q
                    row['SEARCH_TERM'] = name_field
                    annotated_data.append(row)

            # Generate queries using the item's description
            description_field = getattr(item, 'description', None)
            if description_field and len(description_field) > 20:
                # Take first sentence of description as search term
                search_term_from_desc = description_field.split('.')[0].strip()
                # Only use descriptions of reasonable length
                if len(search_term_from_desc) > 5 and len(search_term_from_desc) < 100:
                    queries = [
                        f"Find {result_type_str}s about {search_term_from_desc}.",
                        f"Show {result_type_str}s related to {search_term_from_desc}.",
                    ]
                    for q in queries:
                        row = base_annotation.copy()
                        row['Query'] = q
                        row['SEARCH_TERM'] = search_term_from_desc
                        annotated_data.append(row)

            # Handle themes (accommodating different theme field names and relationships)
            themes_for_item = []
            # Try different ways themes might be stored in the model
            if hasattr(item, 'themes') and hasattr(item.themes, 'all') and callable(getattr(item.themes, 'all')):
                themes_for_item = list(item.themes.all())
            elif hasattr(item, 'app_themes') and hasattr(item.app_themes, 'all') and callable(
                    getattr(item.app_themes, 'all')):
                themes_for_item = list(item.app_themes.all())
            elif hasattr(item, 'themes') and item.themes is not None and isinstance(item.themes, Theme):
                themes_for_item = [item.themes]

            # Generate queries for each theme
            for theme in themes_for_item:
                queries = [
                    f"Find {result_type_str}s in {theme.name}.",
                    f"Show {result_type_str}s for {theme.name}.",
                    f"{result_type_str}s related to {theme.name}.",
                ]
                for q in queries:
                    row = base_annotation.copy()
                    row['Query'] = q
                    row['THEME'] = theme.name
                    annotated_data.append(row)

                if name_field:
                    queries = [
                        f"Find {name_field} {result_type_str} in the {theme.name} theme.",
                    ]
                    for q in queries:
                        row = base_annotation.copy()
                        row['Query'] = q
                        row['SEARCH_TERM'] = name_field
                        row['THEME'] = theme.name
                        annotated_data.append(row)

            geographies_for_item = []
            if hasattr(item, 'geographies') and hasattr(item.geographies, 'all') and callable(
                    getattr(item.geographies, 'all')):
                geographies_for_item = list(item.geographies.all())
            elif hasattr(item, 'geography') and hasattr(item.geography, 'all') and callable(
                    getattr(item.geography, 'all')):
                geographies_for_item = list(item.geography.all())
            elif hasattr(item, 'geography') and item.geography is not None and isinstance(item.geography,
                                                                                          Geography):
                geographies_for_item = [item.geography]

            for geo in geographies_for_item:
                if geo.country and geo.country != 'N/A':
                    queries = [
                        f"Find {result_type_str}s in {geo.country}.",
                        f"Show {result_type_str}s from {geo.country}.",
                    ]
                    for q in queries:
                        row = base_annotation.copy()
                        row['Query'] = q
                        row['COUNTRY_NAME'] = geo.country
                        annotated_data.append(row)
                    if name_field:
                        queries = [
                            f"Find {name_field} {result_type_str} in {geo.country}.",
                        ]
                        for q in queries:
                            row = base_annotation.copy()
                            row['Query'] = q
                            row['SEARCH_TERM'] = name_field
                            row['COUNTRY_NAME'] = geo.country
                            annotated_data.append(row)

                if geo.state and geo.state != 'N/A':
                    queries = [
                        f"Find {result_type_str}s in {geo.state}.",
                        f"Show {result_type_str}s from {geo.state}.",
                    ]
                    for q in queries:
                        row = base_annotation.copy()
                        row['Query'] = q
                        row['STATE_NAME'] = geo.state
                        annotated_data.append(row)
                    if name_field:
                        queries = [
                            f"Find {name_field} {result_type_str} in {geo.state}.",
                        ]
                        for q in queries:
                            row = base_annotation.copy()
                            row['Query'] = q
                            row['SEARCH_TERM'] = name_field
                            row['STATE_NAME'] = geo.state
                            annotated_data.append(row)

            missions = []
            if hasattr(item, 'mission') and item.mission:
                missions = [item.mission]
            elif hasattr(item, 'missions') and item.missions.exists():
                missions = list(item.missions.all())

            for mission_obj in missions:
                if mission_obj.name:
                    queries = [
                        f"Find {result_type_str}s for the {mission_obj.name} mission.",
                        f"Show {result_type_str}s related to {mission_obj.name} mission.",
                    ]
                    for q in queries:
                        row = base_annotation.copy()
                        row['Query'] = q
                        row['MISSION_NAME'] = mission_obj.name
                        annotated_data.append(row)
                    if name_field:
                        queries = [
                            f"Find {name_field} {result_type_str} for the {mission_obj.name} mission.",
                        ]
                        for q in queries:
                            row = base_annotation.copy()
                            row['Query'] = q
                            row['SEARCH_TERM'] = name_field
                            row['MISSION_NAME'] = mission_obj.name
                            annotated_data.append(row)

    # Add some generic queries that don't depend on database content
    generic_queries = [
        "What's new in **climate change**?",
        "Tell me about **AI research**.",
        "Show me **clean energy** solutions.",
        "I need information on **space exploration**.",
        "Where are the **health initiatives**?",
        "Search for **education technology**.",
        "Discover **sustainability** efforts in **Texas**.",
        "Find anything about **urban development** in **Canada**.",
    ]
    for g_q in generic_queries:
        row = {
            'Query': g_q, 'Intent': 'search_query', 'RESULT_TYPE': '', 'THEME': '',
            'MISSION_NAME': '', 'COUNTRY_NAME': '', 'STATE_NAME': '', 'SEARCH_TERM': '',
        }
        if "**" in g_q:
            start = g_q.find('**') + 2
            end = g_q.rfind('**')
            if start != -1 and end != -1 and start < end:
                row['SEARCH_TERM'] = g_q[start:end]
        annotated_data.append(row)

    print(f"Generated {len(annotated_data)} DB-based and generic queries.")
    return annotated_data


# --- Additional Data Generation Logic (from generate_more_to_help_theme.py) ---

def generate_single_additional_query_data():
    """Generates a single query string and its corresponding entity labels using predefined lists."""

    row = {
        "Query": "",
        "Intent": "search_query",
        "RESULT_TYPE": "",
        "THEME": "",
        "COUNTRY_NAME": "",
        "STATE_NAME": "",
        "SEARCH_TERM": ""
    }

    result_type_val = random.choice(result_types_for_additional_data)
    result_type_phrase_func = random.choice(result_type_phrases_for_additional_data)
    query_start = result_type_phrase_func(result_type_val)
    row["RESULT_TYPE"] = result_type_val

    entity_target_type = random.choice(["THEME", "SEARCH_TERM"])

    if entity_target_type == "THEME":
        selected_theme = random.choice(themes_for_additional_data)
        selected_prep = random.choice(theme_preps_for_additional_data)

        row["Query"] = f"{query_start} {selected_prep} {selected_theme}."
        row["THEME"] = selected_theme

    elif entity_target_type == "SEARCH_TERM":
        selected_search_term = ""
        if random.random() < 0.5:
            random_theme_for_search_term = random.choice(themes_for_additional_data)
            if random_theme_for_search_term in specific_search_terms_by_theme_for_additional_data and \
                    specific_search_terms_by_theme_for_additional_data[random_theme_for_search_term]:
                selected_search_term = random.choice(
                    specific_search_terms_by_theme_for_additional_data[random_theme_for_search_term])

        if not selected_search_term:
            selected_search_term = random.choice(generic_search_terms_for_additional_data)

        selected_prep = random.choice(search_term_preps_for_additional_data)

        row["Query"] = f"{query_start} {selected_prep} {selected_search_term}."
        row["SEARCH_TERM"] = selected_search_term

    # Clean up markdown if any
    row['Query'] = row['Query'].replace('**', '')

    return row


def generate_additional_theme_queries_bulk(num_queries=NUM_ADDITIONAL_QUERIES_TO_GENERATE):
    print(f"Generating {num_queries} additional theme-focused queries...")
    additional_data = []
    for _ in range(num_queries):
        additional_data.append(generate_single_additional_query_data())
    print(f"Generated {len(additional_data)} additional queries.")
    return additional_data


def write_to_csv(data, filename):
    fieldnames = [
        'Query', 'Intent', 'RESULT_TYPE', 'THEME',
        'MISSION_NAME', 'COUNTRY_NAME', 'STATE_NAME', 'SEARCH_TERM'
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Combined annotated data written to {filename}")


if __name__ == "__main__":
    print("Starting combined data generation...")

    # 1. Generate data from the database and existing generic queries
    db_and_generic_data = generate_annotated_queries_from_db()

    # 2. Generate additional theme-focused queries
    additional_theme_data = generate_additional_theme_queries_bulk()

    # 3. Combine both sets of data
    combined_data = db_and_generic_data + additional_theme_data

    # Shuffle the combined data to mix things up before writing
    random.shuffle(combined_data)

    # 4. Write the combined data to a single CSV file
    write_to_csv(combined_data, OUTPUT_CSV_FILE)

    print("\n--- Sample of Combined Generated Queries (First 10) ---")
    with open(OUTPUT_CSV_FILE, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip header
        for i, row in enumerate(csv_reader):
            if i >= 10:  # Print only first 10 data rows
                break
            query = row[0]
            theme_val = row[3]
            search_term_val = row[6]
            print(f"Query: \"{query}\"")
            print(f"  -> THEME: \"{theme_val}\"")
            print(f"  -> SEARCH_TERM: \"{search_term_val}\"\n")