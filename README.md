
# Sentinews
***************************************************

Sentinews project aims to investigate the salience of topics like immigration, degree of science-oriented/objectivity 
versus conspiracy/fake and political orientation for online news articles. In this project we use an unsupervised dictionary based sentiment analysis technique (using wordvectors) for calculating the degree of negativity of the articles toward the immigrant outgroups in the Netherlands. The dataset is collected from two online news website, "nu.nl" and "geenstijl.nl" for a period of nine months. 

## Relevant links
**********************************************************
- [Dataset](https://Newsdataset.csv)
- [Scraping project](https://github.com/UtrechtUniversity/news-scraping)
- [NU](https://www.nu.nl/)
- [GeenStijl](https://www.geenstijl.nl/)



## Dataset schema
****************************************

Keys description in the dataset:

<style> 
table td, table th, table tr {text-align:left !important;}
</style>


| Key | Data type|Description |Example|
| --- | --- |--- | --- |
|id| string | The unique id of articles |a5154933|
|title|string |Title of the article|RIVM UPDATE: Deze week +4013 besmettingen|
|teaser|string|A short paragraph between title and text|Aantal nieuwe besmettingen STABILISEERT|
|text|string| The full text of the document|U mag kiezen:Optie 1:...|
|category|string| News section if any| null|
|created_at|datetime object |Date and time of scraping|2020-08-19 16:39:35|
|image|string | Dictionary of the image urls|{0: ''https://image.gscdn.nl/image/5f8b9b2526_Schermafbeelding... |
|reactions|string |Number of reactions to each article|308 reacties|
|author|string |Author|@Ronaldo|
|publication_time|string | Time of publication|14:20|
|publication_date|string |Date of publication, format: dd-mm-yy|18-08-20|
|doctype	|string | Source of the news| geenstijl.nl|
|url|string |URL to the article|https://www.geenstijl.nl/5154933/rivm-update-deze-week-4013-besmettingen/|
|tags|string |List of tags|corona, rivm|
|sitemap_url|string |Link to the site's sitemap if any|https://www.geenstijl.nl/sitemap.xml|


In this project we only make use of "text" field in the dataset.


## Usage
****************************************************************

Outgroup's salience is calculated using:

    Salience outgroups = (N articles on [outgroup] / N articles full)*100
    
![alt text](outtable1.JPG "Salience")

![alt text](outtable2.JPG "Salience")


## Installation
****************************************************
This project requires:
  - Python 3.7 or higher
  -  Install the dependencies with the code below
       ```sh
  pip install -r requirements.txt
  ```

## License and citation
****************************************************

The sentinews project is licensed under the terms of the [MIT License](/LICENSE.md). When using sentinews for academic work, please cite:
-	Tubergen, F., Nadi, S., Bagheri, A. (2021).
sentinews - version 0.1.0. url: github.com/ShNadi/sentinews



## Conribution
**************************************

To contribute code to sentinews, please follow these steps:

- Create a branch and make your changes
- Push the branch to GitHub and issue a Pull Request (PR)
- Discuss with us the pull request and iterate of eventual changes
- Read more about pull requests using [GitHub's official documentation](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

## Contact
*****************************************
Do you have any questions, suggestions, or remarks? Feel free to contact "s.nadi@uu.nl"
