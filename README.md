# Recommendation-Engine
Book Recommendation Engine:
This project is about building a Book Recommendation Engine. The dataset consist of 2 files. 
metaBooks.json - This file contains all the required information about book. Like Book Id, Title, book description, genre etc. The size of the file is 2 GB.
   reviews_Books_5.json - This file contains the customer reviews for these books. The size of this file is 9 GB.
The Recommendation Engine is developed using Collaborative Filtering. The Engine consists of an interactive UI that will enable the user to review some books. Based on the user's ratings, the system will recommend some similar books to the user based on similar user's ratings.
The Front-end has been developed using Flask and it was developed and deployed as a job on GCP-Dataproc. The Flask application was deployed on Goole Engine.
PySpark ALS has been used to build the model and store the same for further use. When a user asks for recommendation, the system uses the existing model to recommend the same.
If its a new user, it is required to recompile the model with the new user-ratings data. This consumes some good amount of time. So I am currently 
working on content-based filtering using K-Means clustering and topic modelling using LDA.
This project is still work in progress.
