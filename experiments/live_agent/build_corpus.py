"""Build the self-contained document corpus + task cards for the demo.

Everything here is original factual prose authored for this harness (no scraped
or licensed text), so the demo is fully self-contained and the FREE fallback
(local keyword retrieval + extractive snippet) always has a real corpus to draw
from. Task value is graded 0-1 by a DETERMINISTIC keyword-rubric scorer (see
agent_loop.score_answer) -- no LLM judging in the loop.

Run:  python3 demo/build_corpus.py
Writes: demo/corpus_docs/*.txt  and  demo/tasks/task_XXX.json
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "corpus_docs"
TASKS = HERE / "tasks"

# --------------------------------------------------------------------------- #
# Corpus documents: {filename: prose}. Kept short and factual.
# --------------------------------------------------------------------------- #
CORPUS = {
"solar_system.txt": """The Solar System formed about 4.6 billion years ago from a collapsing cloud of gas and dust.
The Sun contains more than 99 percent of the system's total mass.
There are eight planets. The four inner planets -- Mercury, Venus, Earth, and Mars -- are small and rocky.
The four outer planets -- Jupiter, Saturn, Uranus, and Neptune -- are gas and ice giants.
Jupiter is the largest planet and has a Great Red Spot, a storm larger than Earth.
Saturn is famous for its bright rings made of ice and rock.
Mars is called the Red Planet because iron oxide on its surface gives it a reddish color.
A belt of asteroids lies between Mars and Jupiter.
Pluto was reclassified as a dwarf planet in 2006.
""",
"photosynthesis.txt": """Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy.
It takes place mainly in the chloroplasts, which contain the green pigment chlorophyll.
The overall reaction uses carbon dioxide and water and, powered by sunlight, produces glucose and oxygen.
The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
The Calvin cycle, also called the light-independent reactions, occurs in the stroma and fixes carbon dioxide into sugar.
Oxygen is released as a byproduct, which is essential for most life on Earth.
Photosynthesis is the foundation of nearly every food chain.
""",
"python_language.txt": """Python is a high-level, general-purpose programming language created by Guido van Rossum and first released in 1991.
It emphasizes code readability and uses significant indentation to define blocks.
Python is dynamically typed and garbage-collected.
It supports multiple paradigms, including procedural, object-oriented, and functional programming.
The reference implementation is called CPython.
Python's package installer is pip, and packages are commonly published on the Python Package Index, known as PyPI.
Popular libraries include NumPy for numerical computing and pandas for data analysis.
The language is named after the comedy group Monty Python, not the snake.
""",
"water_cycle.txt": """The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.
Evaporation turns liquid water from oceans and lakes into water vapor using energy from the Sun.
Transpiration is the release of water vapor from plants.
Condensation is when water vapor cools and forms clouds.
Precipitation falls as rain, snow, sleet, or hail when droplets grow heavy enough.
Runoff and infiltration return water to rivers, oceans, and underground aquifers.
The total amount of water on Earth stays roughly constant; it is recycled over and over.
""",
"roman_empire.txt": """The Roman Empire was one of the largest empires in ancient history.
It began in 27 BC when Augustus became the first Roman emperor.
Before the empire, Rome had been a republic governed by elected officials and the Senate.
At its height under Trajan around 117 AD, the empire stretched from Britain to the Persian Gulf.
The Romans built an extensive network of roads, aqueducts, and public baths.
Latin was the official language and is the ancestor of the Romance languages.
The Western Roman Empire fell in 476 AD when the emperor Romulus Augustulus was deposed.
The Eastern half continued as the Byzantine Empire for nearly another thousand years.
""",
"dna_genetics.txt": """DNA, or deoxyribonucleic acid, carries the genetic instructions for living organisms.
Its structure is a double helix, discovered by James Watson and Francis Crick in 1953, building on the X-ray work of Rosalind Franklin.
DNA is made of four bases: adenine, thymine, guanine, and cytosine.
Adenine pairs with thymine, and guanine pairs with cytosine.
A gene is a segment of DNA that codes for a protein.
The complete set of an organism's DNA is called its genome.
During cell division, DNA is copied so each new cell receives an identical set of instructions.
Mutations are changes in the DNA sequence and are a source of genetic variation.
""",
"electricity_basics.txt": """Electricity is the flow of electric charge, usually carried by electrons moving through a conductor.
Voltage is the electrical potential difference that pushes charge through a circuit, measured in volts.
Current is the rate of flow of charge, measured in amperes.
Resistance opposes the flow of current and is measured in ohms.
Ohm's law states that voltage equals current times resistance.
Conductors like copper allow charge to flow easily, while insulators like rubber resist it.
Direct current flows in one direction, while alternating current periodically reverses direction.
Power, measured in watts, equals voltage times current.
""",
"plate_tectonics.txt": """Plate tectonics is the theory that Earth's outer shell is divided into large plates that move over the mantle.
There are seven major plates and many smaller ones.
At divergent boundaries, plates move apart and new crust forms, such as at mid-ocean ridges.
At convergent boundaries, plates collide, forming mountains or subduction zones.
At transform boundaries, plates slide past each other, causing earthquakes such as along the San Andreas Fault.
The movement is driven by convection currents in the mantle.
The theory explains the distribution of earthquakes, volcanoes, and mountain ranges.
Alfred Wegener first proposed continental drift, an early version of the idea, in 1912.
""",
"internet_history.txt": """The Internet grew out of ARPANET, a network funded by the United States Department of Defense in 1969.
ARPANET first connected computers at four universities.
The TCP/IP protocol suite, developed by Vinton Cerf and Robert Kahn, became the standard in 1983.
Tim Berners-Lee invented the World Wide Web in 1989 at CERN.
The Web uses HTTP to transfer pages and HTML to structure them.
The first web browser and web server were also created by Berners-Lee.
Domain names are translated into numeric IP addresses by the Domain Name System, or DNS.
The Internet is a network of networks, with no single owner.
""",
"human_heart.txt": """The human heart is a muscular organ that pumps blood throughout the body.
It has four chambers: two atria on top and two ventricles on the bottom.
The right side pumps oxygen-poor blood to the lungs, and the left side pumps oxygen-rich blood to the body.
Valves between the chambers prevent blood from flowing backward.
The heart beats about 100,000 times per day in an average adult.
The natural pacemaker is the sinoatrial node, which sets the rhythm.
Arteries carry blood away from the heart, and veins carry it back.
The circulatory system delivers oxygen and nutrients and removes waste.
""",
"greenhouse_effect.txt": """The greenhouse effect is the process by which certain gases in Earth's atmosphere trap heat.
Sunlight passes through the atmosphere and warms the surface.
The surface radiates heat back as infrared radiation.
Greenhouse gases such as carbon dioxide, methane, and water vapor absorb and re-emit this infrared radiation.
Without the natural greenhouse effect, Earth's average temperature would be far below freezing.
Human activities, especially burning fossil fuels, have increased carbon dioxide levels and enhanced the effect.
This enhanced greenhouse effect is the main driver of modern global warming.
""",
"machine_learning.txt": """Machine learning is a branch of artificial intelligence in which systems learn patterns from data rather than following explicit rules.
Supervised learning trains a model on labeled examples to predict outputs for new inputs.
Unsupervised learning finds structure in unlabeled data, such as clusters.
Reinforcement learning trains an agent to take actions that maximize a reward signal.
Overfitting happens when a model memorizes the training data and fails to generalize.
A dataset is usually split into training, validation, and test sets.
Neural networks are models loosely inspired by the brain and are the basis of deep learning.
Gradient descent is a common optimization method for training models.
""",
"ancient_egypt.txt": """Ancient Egypt was a civilization along the Nile River in northeastern Africa.
Its history spans roughly 3000 BC to 30 BC.
The pharaoh was the political and religious leader, considered a living god.
The Egyptians built pyramids as tombs; the Great Pyramid of Giza was built for the pharaoh Khufu.
They developed a writing system called hieroglyphics.
The Rosetta Stone, discovered in 1799, allowed scholars to decode hieroglyphics.
The Nile's annual flooding deposited fertile soil that supported agriculture.
Egypt was conquered by Alexander the Great and later became a Roman province after Cleopatra's death.
""",
"periodic_table.txt": """The periodic table organizes the chemical elements by increasing atomic number.
Atomic number is the number of protons in an atom's nucleus.
It was arranged by Dmitri Mendeleev in 1869, who left gaps for undiscovered elements.
Elements in the same column, called a group, have similar chemical properties.
Rows are called periods.
Metals are on the left and center, nonmetals on the right, and metalloids between them.
Hydrogen is the lightest element, and it is the most abundant in the universe.
The noble gases, such as helium and neon, are in the rightmost group and rarely react.
""",
"renewable_energy.txt": """Renewable energy comes from sources that are naturally replenished, unlike fossil fuels.
Solar power converts sunlight into electricity using photovoltaic cells.
Wind power uses turbines to convert the kinetic energy of wind into electricity.
Hydroelectric power generates electricity from flowing or falling water.
Geothermal energy taps heat from within the Earth.
Biomass energy comes from burning organic material such as wood or crop waste.
Renewables produce little or no greenhouse gas during operation, helping reduce carbon emissions.
Storage, such as batteries, helps balance supply when the sun is not shining or the wind is not blowing.
""",
}


# --------------------------------------------------------------------------- #
# Task cards. Each: id, question, source_docs, rubric (list of keyword groups;
# a group is satisfied if the answer contains ANY of its synonyms), reference.
# Score = satisfied_groups / total_groups (deterministic, 0-1).
# --------------------------------------------------------------------------- #
def _card(idx, q, docs, rubric, ref):
    return {"id": f"task_{idx:03d}", "question": q, "source_docs": docs,
            "rubric": rubric, "reference": ref}


TASK_SPECS = [
    ("How old is the Solar System and where did it form from?", ["solar_system.txt"],
     [["4.6 billion", "4.6"], ["cloud", "gas", "dust", "nebula"]],
     "About 4.6 billion years ago from a collapsing cloud of gas and dust."),
    ("Which are the four inner rocky planets?", ["solar_system.txt"],
     [["mercury"], ["venus"], ["earth"], ["mars"]],
     "Mercury, Venus, Earth, and Mars."),
    ("Why is Mars called the Red Planet?", ["solar_system.txt"],
     [["iron oxide", "iron", "rust"], ["red", "reddish"]],
     "Iron oxide on its surface gives it a reddish color."),
    ("What is the largest planet and what famous feature does it have?", ["solar_system.txt"],
     [["jupiter"], ["great red spot", "red spot", "storm"]],
     "Jupiter, which has the Great Red Spot, a giant storm."),

    ("What does photosynthesis convert and into what?", ["photosynthesis.txt"],
     [["light", "sunlight", "light energy"], ["chemical energy", "glucose", "sugar"]],
     "It converts light energy into chemical energy (glucose)."),
    ("What are the inputs and outputs of the photosynthesis reaction?", ["photosynthesis.txt"],
     [["carbon dioxide", "co2"], ["water"], ["glucose", "sugar"], ["oxygen"]],
     "Inputs: carbon dioxide and water. Outputs: glucose and oxygen."),
    ("Where does the Calvin cycle occur and what does it do?", ["photosynthesis.txt"],
     [["stroma"], ["carbon", "co2", "carbon dioxide"], ["sugar", "fix", "glucose"]],
     "In the stroma; it fixes carbon dioxide into sugar."),

    ("Who created Python and in what year was it first released?", ["python_language.txt"],
     [["guido", "van rossum"], ["1991"]],
     "Guido van Rossum; first released in 1991."),
    ("What is Python's package installer and where are packages published?", ["python_language.txt"],
     [["pip"], ["pypi", "package index"]],
     "pip; packages are published on PyPI, the Python Package Index."),
    ("Name two popular Python libraries and what they are for.", ["python_language.txt"],
     [["numpy"], ["pandas"], ["numerical", "data analysis", "data"]],
     "NumPy for numerical computing and pandas for data analysis."),
    ("What is Python named after?", ["python_language.txt"],
     [["monty python", "comedy", "monty"]],
     "The comedy group Monty Python, not the snake."),

    ("What drives evaporation in the water cycle?", ["water_cycle.txt"],
     [["sun", "solar", "energy from the sun"], ["evaporation", "evaporate"]],
     "Energy from the Sun drives evaporation of liquid water."),
    ("List the forms precipitation can take.", ["water_cycle.txt"],
     [["rain"], ["snow"], ["sleet"], ["hail"]],
     "Rain, snow, sleet, or hail."),
    ("What is transpiration?", ["water_cycle.txt"],
     [["plants", "plant"], ["water vapor", "vapor", "vapour"]],
     "The release of water vapor from plants."),

    ("When did the Roman Empire begin and who was the first emperor?", ["roman_empire.txt"],
     [["27 bc"], ["augustus"]],
     "27 BC, when Augustus became the first emperor."),
    ("What governed Rome before the empire?", ["roman_empire.txt"],
     [["republic"], ["senate", "elected"]],
     "A republic with elected officials and the Senate."),
    ("When did the Western Roman Empire fall and what continued after it?", ["roman_empire.txt"],
     [["476"], ["byzantine", "eastern"]],
     "It fell in 476 AD; the Eastern half continued as the Byzantine Empire."),

    ("Who discovered the structure of DNA and in what year?", ["dna_genetics.txt"],
     [["watson"], ["crick"], ["1953"]],
     "James Watson and Francis Crick in 1953."),
    ("What are the four DNA bases and how do they pair?", ["dna_genetics.txt"],
     [["adenine"], ["thymine"], ["guanine"], ["cytosine"]],
     "Adenine-thymine and guanine-cytosine."),
    ("What is a gene and what is a genome?", ["dna_genetics.txt"],
     [["gene", "segment"], ["protein"], ["genome", "complete set"]],
     "A gene is a DNA segment coding for a protein; the genome is all the DNA."),

    ("State Ohm's law.", ["electricity_basics.txt"],
     [["voltage", "volt"], ["current", "ampere"], ["resistance", "ohm"]],
     "Voltage equals current times resistance."),
    ("What is measured in volts, amperes, and ohms?", ["electricity_basics.txt"],
     [["voltage", "volt"], ["current", "ampere"], ["resistance", "ohm"]],
     "Voltage in volts, current in amperes, resistance in ohms."),
    ("What is the difference between direct and alternating current?", ["electricity_basics.txt"],
     [["direct", "one direction"], ["alternating", "reverse", "reverses"]],
     "Direct current flows one way; alternating current reverses direction."),

    ("What drives the movement of tectonic plates?", ["plate_tectonics.txt"],
     [["convection"], ["mantle"]],
     "Convection currents in the mantle."),
    ("What happens at divergent versus convergent boundaries?", ["plate_tectonics.txt"],
     [["divergent", "apart", "new crust"], ["convergent", "collide", "collision"]],
     "Divergent: plates move apart and new crust forms. Convergent: plates collide."),
    ("Who proposed continental drift and when?", ["plate_tectonics.txt"],
     [["wegener"], ["1912"]],
     "Alfred Wegener in 1912."),

    ("What did ARPANET become and when did TCP/IP become standard?", ["internet_history.txt"],
     [["arpanet"], ["tcp/ip", "tcp", "ip"], ["1983"]],
     "ARPANET led to the Internet; TCP/IP became the standard in 1983."),
    ("Who invented the World Wide Web and where?", ["internet_history.txt"],
     [["berners-lee", "berners"], ["cern"], ["1989"]],
     "Tim Berners-Lee at CERN in 1989."),
    ("What does DNS do?", ["internet_history.txt"],
     [["domain", "domain name"], ["ip address", "ip", "numeric"]],
     "It translates domain names into numeric IP addresses."),

    ("How many chambers does the human heart have and what are they?", ["human_heart.txt"],
     [["four", "4"], ["atria", "atrium"], ["ventricle", "ventricles"]],
     "Four: two atria and two ventricles."),
    ("What does the natural pacemaker of the heart do?", ["human_heart.txt"],
     [["sinoatrial", "sa node", "pacemaker"], ["rhythm", "beat"]],
     "The sinoatrial node sets the heart's rhythm."),
    ("What is the difference between arteries and veins?", ["human_heart.txt"],
     [["arteries", "artery"], ["veins", "vein"], ["away", "back"]],
     "Arteries carry blood away from the heart; veins carry it back."),

    ("Which gases cause the greenhouse effect?", ["greenhouse_effect.txt"],
     [["carbon dioxide", "co2"], ["methane"], ["water vapor", "vapour", "vapor"]],
     "Carbon dioxide, methane, and water vapor."),
    ("What human activity has enhanced the greenhouse effect?", ["greenhouse_effect.txt"],
     [["fossil fuel", "fossil fuels", "burning"], ["carbon dioxide", "co2"]],
     "Burning fossil fuels, which raises carbon dioxide."),

    ("What is the difference between supervised and unsupervised learning?", ["machine_learning.txt"],
     [["supervised", "labeled", "labelled"], ["unsupervised", "unlabeled", "cluster"]],
     "Supervised uses labeled data; unsupervised finds structure in unlabeled data."),
    ("What is overfitting?", ["machine_learning.txt"],
     [["overfit", "memoriz", "memoris"], ["generalize", "generalise", "new data"]],
     "When a model memorizes training data and fails to generalize."),
    ("What is reinforcement learning?", ["machine_learning.txt"],
     [["agent"], ["reward"], ["action", "actions"]],
     "An agent learns actions that maximize a reward signal."),

    ("What did the Rosetta Stone allow scholars to do, and when was it found?", ["ancient_egypt.txt"],
     [["hieroglyph", "hieroglyphics"], ["1799"]],
     "It let scholars decode hieroglyphics; found in 1799."),
    ("For which pharaoh was the Great Pyramid of Giza built?", ["ancient_egypt.txt"],
     [["khufu"], ["giza", "great pyramid"]],
     "For the pharaoh Khufu (Great Pyramid of Giza)."),

    ("Who arranged the periodic table and how are elements ordered?", ["periodic_table.txt"],
     [["mendeleev"], ["atomic number", "protons", "number"]],
     "Dmitri Mendeleev; ordered by increasing atomic number."),
    ("What are noble gases and give examples.", ["periodic_table.txt"],
     [["noble gas", "noble gases"], ["helium", "neon"], ["rarely react", "unreactive", "inert"]],
     "Unreactive gases like helium and neon in the rightmost group."),

    ("Name three renewable energy sources and what they use.", ["renewable_energy.txt"],
     [["solar", "sunlight"], ["wind"], ["hydro", "water"]],
     "Solar (sunlight), wind (turbines), and hydroelectric (water)."),
    ("Why is storage important for renewable energy?", ["renewable_energy.txt"],
     [["storage", "battery", "batteries"], ["balance", "supply", "not shining", "not blowing"]],
     "Storage balances supply when the sun or wind is unavailable."),
]


def build():
    DOCS.mkdir(parents=True, exist_ok=True)
    TASKS.mkdir(parents=True, exist_ok=True)
    for name, text in CORPUS.items():
        (DOCS / name).write_text(text)
    for i, (q, docs, rubric, ref) in enumerate(TASK_SPECS, start=1):
        card = _card(i, q, docs, rubric, ref)
        (TASKS / f"{card['id']}.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {len(CORPUS)} corpus docs to {DOCS}")
    print(f"wrote {len(TASK_SPECS)} task cards to {TASKS}")


if __name__ == "__main__":
    build()
