# In this python project, there are 2 methods of predicting rating of a given movie by a given user.

1) User-Item based collaborative filtering : Finding the users similar to the
target user and then taking the weighted average of the ratings given by
them on the target movie.

2) Item-Item based collaborative filtering : Finding the movies similar to the
target movie based on the ratings given by the target user on those
movies.

We have used K-NN to find the top K movies/users for the above methods and the end the top 10 movies are shown to the user as an output for both the methods.
We have used MovieLens Dataset for study purpose whose link is attached in the report.
For reference, a report is attached for further understanding of methods and project.

To compile and run, 
python3 code.py <targetUser> <K> <ratings_file> <movies_file>
 
 Note: targetUser is an integer number from the movieslens dataset
 K is the K-NN value
 Ratings_file is the ratings.csv file from the movieLens Dataset
 movies_file is the movies.csv file from the movieLens dataset
