import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

tickets=[
    "I was charged twice for subscription",
    "the app crashes when i open settings",
    "please reset my password,cannot log in",
    "refund not received for cancelled order",
    "error 500 when uploading a file",
    "how do i change my registered email"
]
cleaned_ticket=[]
for ticket in tickets:
    ticket=ticket.lower()
    ticket=re.sub(r"[^\w\s]","",ticket)
    words=ticket.split()
    cleaned_ticket.append(" ".join(words))

print("Preprocessed Tickets:")
for ticket in cleaned_ticket:
    print(ticket)
vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform(cleaned_ticket)
print("\nTF-IDF Shape",tfidf_matrix.shape)
words=vectorizer.get_feature_names_out()
scores=tfidf_matrix[0].toarray()[0]
top_indices=scores.argsort()[::-1][:5]
print("\nTop 5 words in Ticket1:")
for index in top_indices:
    print(words[index],":",round(scores[index],3))
query="cannot log in to my account"
query_vector=vectorizer.transform([query])
similarity_scores=cosine_similarity(
    query_vector,tfidf_matrix
)[0]
best_index=similarity_scores.argmax()
print("\n most similar ticket:")
print("ticket number:",best_index+1)
print("ticket:",tickets[best_index])
print("similarity score:",round(similarity_scores[best_index],3))
training_tickets=[
    ("i was charged twice","billing"),
    ("my subscription payment failed","billing"),
    ("wrong amount charged to my card","billing"),
    ("payment was deducted twice","billing"),
    ("i have a billing problem","billing"),
    ("my invoice shows an incorrect charge","billing"),

    ("the app crashes on startup","technical"),
    ("application freezes when","technical"),
    ("error 500 while uploading file","technical"),
    ("the website is not loading","technical"),
    ("app crashes when i open settings","technical"),
    ("i get an error while uploading","technical"),

    ("i cannot log in to my account","login"),
    ("please rest my pasword","login"),
    ("forget my account password","login"),
    ("login fails with the correct password","login"),
    ("unable to sign in","login"),
    ("i need help logging in my account","login"),

    ("my cancelled order refund is missing","refund"),
    ("i have not received my refund","refund"),
    ("please return my money for my order","refund"),
    ("refund is delayed","refund"),
    ("my refund has not arrived yet","refund"),
    ("i need a refund for a cancelled purchase","refund"),

    ("i want to change my registered email","account"),
    ("how can i update my email account","account"),
    ("please change my account details","account"),
    ("i need to update my profile","account"),
    ("how do i edit my registered email","account"),
    ("i want to update my account information","account")
]

X=[]
y=[]
for ticket,label in training_tickets:
    X.append(ticket)
    y.append(label)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("classifier",LogisticRegression(max_iter=1000))
])
model.fit(X_train,y_train)
prediction=model.predict(X_test)
accuracy=accuracy_score(y_test,prediction)
print("\nLogistic Regression Accuracy",round(accuracy,2))