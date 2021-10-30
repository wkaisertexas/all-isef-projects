import os
import pandas as pd
from Project import Project

import argparse

parser = argparse.ArgumentParser(description='Get science fair projects')
parser.add_argument('-t', action="store_true", help='Test mode (Only processes one entry so that database outputs can be tested')
parser.print_help()
args = parser.parse_args()


if __name__ == "__main__":
    # This gets a list of all of the project id's
    print("Downloading a list of all of the projects")
    os.system("bash curl.sh > ids")
    with open("ids") as f:
        data = f.read().split("<tr>")
        
        print(f"Found {len(data)} Projects")
        if args.t:
            data = data[2:3]
        else:
            data = data[2:-1]

        projects = []
        failed, tried = 0, 0
        for row in data:
            try:
                tried-=-1
                projects.append(Project(row))
            except:
                failed += 1
                print(f"({failed / tried * 100:.2f} %) Failed to save a project")

    result = projects[0].convert_to_data_frame()
    for project in projects[1:]:
        result.append(project.convert_to_data_frame())
    
    print("\nExporting:\n")

    try:
        result.to_excel("output/dataset.xlsx", index=False)
    except:
        print("Failed to export to excel")

    
    try:
        result.to_csv("output/dataset.csv", index=False)
    except:
        print("Failed to export to excel")

    # OPTIONAL: Add other formats using pandas if you would like

