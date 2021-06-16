# 1- DataScience Blog Post
___
You can read my Medium Blog Post [here](https://lorenzo-azar.medium.com/data-manipulation-tutorial-using-python-pandas-library-airbnb-dataset-231ccc90d508).

## Libraries
You simply need pandas library to run the notebook. 
Install pandas using:

    pip install pandas

If you would like to add more libraries to do more exploration or visualization, add it to the requirements.txt file on separate lines and run:
    
    pip install -r requirements.txt

## Main Questions to Answer
1. Which area (Between Seattle and Boston) has the highest housing rates? And what is the average difference between them?
2. Which area has more availability in general?
3. What are the times of the year (by month) where there is more availability in housing?

## Answers
1. Boston in general has higher prices than Seattle. The average difference is around 55\$. While the maximum price shows a huge difference (1650\$ in Seattle vs 7163\$ in Boston)
2. In general Boston has more availability, around 50%, while the availability in Seattle is around 33%
3. Disregarding the January month of 2017 which might be an outlier with less data, we can conclude that at all time of the years there is more housing availability in Seattle, and in both Seattle and Boston, the availability goes down in a small percentage moving forward into the year.

## Files in the repository
Datasets in csv format:

    Boston_Airbnb_Calendar_Listing.csv
    Seattle_Airbnb_Calendar_Listing.csv
Requirements file:
    
    requirements.txt
Project's Jupyter Notebook:

    Data Science Blog Post.ipynb

## Acknowledgements
### Airbnb Open Data
The datasets used in this repository were taken from Airbnb's Open Data.
Seattle: [dataset link](https://www.kaggle.com/airbnb/seattle/data)
Boston: [dataset link](https://www.kaggle.com/airbnb/boston)