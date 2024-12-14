# Necessarily Libraries:
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate

#1 Load the IMDb dataset into a pandas DataFrame:-
data=pd.read_csv("C:\\Users\\Dell\\Downloads\\IMDB-Movie-Data.csv")
data

#2. first 5 rows of the dataset:- 
data.head(5)

#3. Overview of column data types and missing values:-
data.info()

#4 Columns with missing values:-
data.isna().sum()

#5. Essential columns for analyzing movie ratings and details:-
data.drop(['Rank','Description'],axis=1)

#6. Average runtime of movies:- 
data['Runtime (Minutes)'].mean()

#7. Number of movies in each genre:-
all_act = data['Genre'].str.cat(sep=',')
act_list = all_act.split(',')

## freuency of every actor:-
act_counts = pd.Series(act_list).value_counts()

#top 3 actors
top_act = act_counts.head()

print(top_act)

#8. Top 5 directors with the most movies in the dataset:-
dic_count=data['Director'].value_counts() 
print(dic_count.head(5))

#9. Line plot showing the number of movies released each year:-
plt.figure(figsize=(15, 5))
fig=data['Year'].value_counts().sort_index()
plt.title('released movies in a year')
plt.xlabel('year')
plt.ylabel('count of movies')
plt.plot(fig.index,fig.values)

#10. Histogram depicting the distribution of movie runtime:-
plt.figure(figsize=(15, 5))
plt.hist(data['Runtime (Minutes)'])

#11. Histogram showcasing the distribution of movie ratings:-
plt.figure(figsize=(15, 5))
plt.hist(data['Rating'])

data.corr()
#12. Correlation coefficient between movie ratings and runtime:-
correlation = data['Rating'].corr(data['Runtime (Minutes)'])
print("Correlation Coefficient:", correlation)

#13. 3 most frequent actors in the datase:-
all_act = data['Actors'].str.cat(sep=', ')
act_list = all_act.split(', ')
#frequency of every actor:-
act_counts = pd.Series(act_list).value_counts()
#top 3 actors:-
top_act = act_counts.head(3)
#top actors and their frequencies:-
print(top_act)

#14 Relationship between box office earnings and movie ratings:-
plt.figure(figsize=(15, 5))
plt.scatter(data['Revenue (Millions)'], data['Rating'], alpha=0.7)
plt.title(' Earnings vs. Movie Ratings')
plt.xlabel(' Earnings (Millions)')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

# correlation coefficient:-
correlation = data['Revenue (Millions)'].corr(data['Rating'])
print("Correlation Coefficient:", correlation)

#16. Histogram of budgets for their distribution:-
plt.figure(figsize=(15, 5))
plt.xlabel('Earnings (Millions)')
plt.ylabel('Count of movie')
plt.hist(data['Revenue (Millions)'])

#19. Word cloud visualization using movie titles or keywords from descriptions.
# Combine movie titles into a string
all_titles = ' '.join(data['Title'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)

# Plot
plt.figure(figsize=(15, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Movie Titles Word Cloud')
plt.show()

# 20Calculate average rating/year:-
avg_ratings = data.groupby('Year')['Rating'].mean()

# Line plot:-
plt.figure(figsize=(15, 5))
plt.plot(avg_ratings.index, avg_ratings.values, marker='o')
plt.title('Average Movie Ratings Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

#22. Most common director-actor pairs:-
data['Actors'] = data['Actors'].str.split(', ')
data = data.explode('Actors')

# Count occurrences of each pair:-
director_actor_counts = data.groupby(['Director', 'Actors']).size().reset_index(name='Count')

# Most common director-actor pairs:-
most_common_pairs = director_actor_counts.sort_values(by='Count', ascending=False).head(5)
print("Most Common Director-Actor Pairs:")
print(most_common_pairs)

#23. Analysis about how movie runtimes have been changed over the years:-
average_runtime_per_year = data.groupby('Year')['Runtime (Minutes)'].mean()

# Line plot
plt.figure(figsize=(15, 5))
plt.plot(average_runtime_per_year.index, average_runtime_per_year.values, marker='o')
plt.title('Average Movie Runtimes ')
plt.xlabel('Year')
plt.ylabel('Average Runtime (minutes)')
plt.grid(True)
plt.show()
overall_trend = 'increasing' if average_runtime_per_year.diff().mean() > 0 else 'decreasing'
print(f"Overall Trend: Movies are {overall_trend} in runtime")

genre_counts_per_year = data.groupby(['Year', 'Genre']).size().unstack(fill_value=0)

# Plot the trends of different genres over the years
plt.figure(figsize=(15, 10))
genre_counts_per_year.plot()
plt.title('Genre Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
#plt.legend(title='Genre')
plt.grid(True)
plt.show()

corr=data['Rating'].corr(data['Runtime (Minutes)'])
corr

data.plot(kind='hist',x='Rating',y='Votes')
plt.show()

data.plot(kind='bar',x='Rating',y='Revenue (Millions)')
plt.show()

data.plot(kind='hist',bins=5,x='Revenue (Millions)',y='Metascore')
plt.show()

# 26 Set the movie titles as the index for better visualization
plt.figure(figsize=(15, 5))

# Bar plot for ratings
data['Rating'].plot(kind='bar', color='blue')
plt.title('Movie Ratings')
plt.ylabel('Rating')


# Bar plot for revenue
plt.figure(figsize=(15, 5))
data['Revenue (Millions)'].plot(kind='bar', color='blue')
plt.title('Movie Revenue')
plt.ylabel('Revenue (Millions)')
plt.tight_layout()
plt.show()

#28 Analyze the relationship between the number of votes and movie ratings
plt.figure(figsize=(15, 5))
plt.scatter(data['Votes'], data['Rating'], color='blue', alpha=0.7)
plt.xlabel('Number of Votes')
plt.ylabel('Rating')
plt.grid(True)
plt.show()


# 29,Explore the distribution of ratings for different genres. Are there variations?
# Get unique genres
unique_genres = data['Genre'].unique()
# Create box plots to explore rating distribution for different genres
plt.figure(figsize=(15, 5))
plt.boxplot([data[data['Genre'] == genre]['Rating'] for genre in unique_genres], labels=unique_genres)
plt.title('Distribution of Ratings')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.show()

#30 find outliers
# Z-scores for movie runtime
z_scores = (data['Runtime (Minutes)'] - data['Runtime (Minutes)'].mean()) / data['Runtime (Minutes)'].std()
outliers = data[abs(z_scores) > 3]
# Box plot for runtime data
plt.figure(figsize=(15, 5))
plt.boxplot(data['Runtime (Minutes)'])
plt.title('Runtime Box Plot')
plt.ylabel('Runtime (minutes)')
plt.show()

# Print the identified outliers
print("Identified Outliers:")
print(outliers)

