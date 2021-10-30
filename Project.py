import requests, re
import pandas as pd


class Project:
    def __init__(self, row, debug=True): 
        self.project_id = int(re.findall("ProjectId=\d+", row)[0].split("=")[1])
        self.abstract_html = self.get_abstract_html(self.project_id)
       
        self.country = self.get_country(row)
        self.state = self.get_state(row)
        self.province = self.get_province(row)
        self.year = self.get_year()
        self.first_names, self.last_names = self.get_names()
        self.project_name = self.get_project_name()
        self.category = self.get_category()
        self.awards = self.get_awards()
        self.awards_in_dollars = self.get_awards_in_dollars(self.awards)

        self.abstract = self.get_abstract()
        self.booth_id = self.get_booth_id()
        
        if debug:
            print("Currently On {}".format(self.project_id))
        
    def get_year(self):
        self.year = int(self.abstract_html.split("Year:")[1].split("<")[2].split(">")[1])

    def get_names(self):
        # this will convert all of these names into arrays of both first and last names
        rough_names = self.abstract_html.split("Finalist Names:")[1].split("<")[2].split(">")[1]

        first_names = []
        last_names = []

        # this will remove the spaces at the start of a name
        while rough_names.find(' ') == 0:
            rough_names = rough_names[1:] # this removes the first character
        
        # I can run more text processing on these names later
        # but the important thing is that they are split by a comma
        for name in rough_names.split('<br>'):
            last_names.append(name.split(', ')[0])
            if not name.find(', ') == -1:
                first_names.append(name.split(', ')[1])
        return first_names, last_names
   
    def get_project_name(self):
        return  self.abstract_html.split("h2")[2].split("<")[0].split(">")[1]


    def get_category(self):
        return self.abstract_html.split("Category:")[1].split("<")[2].split(">")[1]


    def get_country(self, row):
        return row.replace("</td>", "").split("td")[4]


    def get_state(self, row):
        return row.replace("</td>", "").split("<td>")[5]


    def get_province(self, row):
        return row.replace("</td>", "").split("<td>")[6]


    def get_awards(self):
        if self.abstract_html.find("Awards Won:") != -1:
            return self.abstract_html.split("Awards Won:")[1].split("<")[3].split(">")[0]
        else:
            return "";

    def get_awards_in_dollars(self, awards):
        formatted_awards = awards.replace(",", "")
        return [int(award) for award in re.findall("$\d+", awards)]


    def get_abstract_html(self, project_id):
        link = "https://abstracts.societyforscience.org/Home/FullAbstract?ProjectId=" + str(project_id)
        return requests.get(link).text


    def get_booth_id(self):
        return self.abstract_html.split("Booth Id:")[1].split("<")[2].split(">")[1]


    def get_abstract(self):
    	return self.abstract_html.split("Abstract:")[1].split("</p>")[0].split(">")[1]


    def convert_to_data_frame(self):
        return_dict = {
            'year': self.year,
            'first_names': self.first_names,
            'last_names': self.last_names,
            'project_id': self.project_id,
            'project_name': self.project_name, 
            'category': self.category,
            'state': self.state,
            'province': self.province,
            'awards': self.awards,
            'booth_id': self.booth_id,
            'abstract': self.abstract
            }
        return pd.DataFrame(data=return_dict)

    def __str__(self):
        return "{} in {}".format(self.project_name, self.category)   
        
