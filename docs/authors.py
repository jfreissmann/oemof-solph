# -*- coding: utf-8 -*-

import yaml

with open('CITATION.cff', 'r') as file:
    data = yaml.safe_load(file)

authors = data["authors"]

with open("AUTHORS.tmp", "w") as f:

    f.write("Authors\n")
    f.write("=======\n")
    f.write("\n")
    f.write("--**in alphabetical order**--\n")
    f.write("\n")
    for author in authors[:-1]:
        f.write(
            "* "
            + author["given-names"] + " "
            + author["family-names"] + "\n"
        )
