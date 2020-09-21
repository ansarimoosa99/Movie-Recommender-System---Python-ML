from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["london paris london", "paris paris london"]

#to count the occurence of word
#countvec.. is a class and hence needed to be initialized.
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)  
#print(count_matrix)
print(count_matrix.toarray())


print("__________separator_________________")


#to find the similarity between 2 sentences .... cosine similarity is method
cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)