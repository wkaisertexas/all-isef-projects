# ISEF Database
This is a simple web scraper which gets all of the projects and abstract information from [here](abstracts.scienceforsociety.org). 

My goal for this is for someone to get inspired to make a science fair project by doing analysis of this dataset for a "meta" project.
---
## To use:
Simply run the following code:
```
pip3 install -r requirements.txt
python3 main.py
```
This should take a while to run as it will be getting ~20k projects, so just let it run in the background.

At the end of running this program, a folder called `Data` will be created and in this you will find many different representations of the database (excel, pandas, sql, etc.)
