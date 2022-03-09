## ISEF Database

This is a simple web scraper which gets all of the projects and abstract information from [here](https://abstracts.scienceforsociety.org). 

### Goal
I want someone to get inspired to do a "meta" science fair project. Someone could look at what makes science fair projects sucessful *in* in science fair project. Comedy.

---

### To use:
Simply run the following code:
```
pip3 install -r requirements.txt
python3 main.py
```
This should take a while to run as it will be getting ~20k projects, so just let it run in the background.

At the end of running this program, a folder called `output` will be created and in this you will find many different representations of the database (excel, pandas, sql, etc.)

There is an older ISEF database that I already compiled, but this does not have the latest code.

Finally: I wrote this program a really long time ago so please do not judge me for it. If someone actually decides to use this, please ask me if you have any questions. I will be happy to help. To sum, I really want someone to show up with a Meta-Analysis of Science fair Projects.

### FAQ

- Why did I not use beautiful soup or something like that?
I was in Eight grade and did not know that existed.
