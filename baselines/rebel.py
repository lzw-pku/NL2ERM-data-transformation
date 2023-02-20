from transformers import pipeline

triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')


def getRelation(sentence):
    tmp = triplet_extractor(sentence,
                            return_tensors=True, return_text=False)
    tmp = tmp[0]["generated_token_ids"]['output_ids']
    extracted_text = triplet_extractor.tokenizer.batch_decode(tmp)
    extracted_triplets = extract_triplets(extracted_text[0])
    return extracted_triplets

# We need to use the tokenizer manually since we need special tokens.

# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets



sentences1 = [
    'A library contains libraries, books, authors and patrons.',
    'Libraries are described by library name and location.',
    'Books are described by title and pages.',
    'Authors are described by author name.',
    'Patrons are described by patron name and patron weight.',
    'A library can hold many books.',
    'The book can appear in many libraries.',
    'Beijing is the capital of China.',
    'I am Zhenwen Li.'
]
sentences2 = '''
Bank have Customer.
Banks are identified by a name, code, address of main office.
Banks have branches.
Branches are identified by a branch number, branch name, address.
Customers are identified by name, custome id, phone number, address.
Customer can have one or more accounts.
Accounts are identified by account number, account type, balance.
Customer can avail loans.
Loans are identified by loan id, loan type and amount.
Account and loans are related to bank’s branch.

A library contains libraries, books, authors and patrons.
Libraries are described by library name and location.
Books are described by title and pages.
Authors are described by author name.
Patrons are described by patron name and patron weight.
A library can hold many books.
The book can appear in many libraies.

The university keeps track of each student’s name, student number, Social Security number, current address and phone number, permanent address and phone number, birth date, sex, class (freshman, sophomore, ..., graduate), major department, minor department (if any), and degree program (B.A., B.S., ..., Ph.D.).
Some user applications need to refer to the city, state, and ZIP Code of the student’s permanent address and to the student’s last name.
Both Social Security number and student number have unique values for each student.

Each department is described by a name, department code, office number, office phone number, and college.
Both name and code have unique values for each department.

Each course has a course name, description, course number, number of semester hours, level, and offering department.
The value of the course number is unique for each course.

Each section has an instructor, semester, year, course, and section number.
The section number distinguishes sections of the same course that are taught during the same semester/year; its values are 1, 2, 3, ..., up to the number of sections taught during each semester.

A grade report has a student, section, letter grade, and numeric grade (0, 1, 2, 3, or 4).

The mail order company has employees, each identified by a unique employee number, first and last name, and Zip Code.
Each customer of the company is identified by a unique customer number, first and last name, and Zip Code.

Each part sold by the company is identified by a unique part number, a part name, price, and quantity in stock.
Each order placed by a customer is taken by an employee and is given a unique order number.
Each order contains specified quantities of one or more parts.
Each order has a date of receipt as well as an expected ship date.
The actual ship date is also recorded.

Each movie is identified by title and year of release.
Each movie has a length in minutes.
Each movie has a production company, and each is classified under one or more genres (such as horror, action, drama, and so forth).
Each movie has one or more directors and one or more actors appear in it.
Each movie also has a plot outline.
Finally, each movie has zero or more quotable quotes, each of which is spoken by a particular actor appearing in the movie.
Actors are identified by name and date of birth and appear in one or more movies.
Each actor has a role in the movie.
Directors are also identified by name and date of birth and direct one or more movies.
It is possible for a director to act in a movie (including one that he or she may also direct).
Production companies are identified by name and each has an address.
A production company produces one or more movies.

Authors of papers are uniquely identified by e-mail id.
First and last names are also recorded.
Each paper is assigned a unique identifier by the system and is described by a title, abstract, and the name of the electronic file containing the paper.
A paper may have multiple authors, but one of the authors is designated as the contact author.
Reviewers of papers are uniquely identified by e-mail address.
Each reviewer’s first name, last name, phone number, affiliation, and topics of interest are also recorded.

Each paper is assigned between two and four reviewers.
A reviewer rates each paper assigned to him or her on a scale of 1 to 10 in four categories: technical merit, readability, originality, and relevance to the conference.
Finally, each reviewer provides an overall recommendation regarding each paper.
Each review contains two types of written comments: one to be seen by the review committee only and the other as feedback to the author(s).
'''

'''
[{'head': 'Bank', 'type': 'has part', 'tail': 'Customer'}, {'head': 'Customer', 'type': 'part of', 'tail': 'Bank'}]
[{'head': 'address', 'type': 'part of', 'tail': 'Bank'}]
[{'head': 'branches', 'type': 'part of', 'tail': 'Banks'}]
[{'head': 'address', 'type': 'part of', 'tail': 'Branch'}]
[{'head': 'custome id', 'type': 'has part', 'tail': 'phone number'}, {'head': 'custome id', 'type': 'has part', 'tail': 'address'}, {'head': 'phone number', 'type': 'part of', 'tail': 'custome id'}, {'head': 'address', 'type': 'part of', 'tail': 'custome id'}]
[{'head': 'customer', 'type': 'has part', 'tail': 'accounts'}, {'head': 'accounts', 'type': 'part of', 'tail': 'customer'}]
[{'head': 'balance', 'type': 'subclass of', 'tail': 'Account'}]
[{'head': 'Loan', 'type': 'subclass of', 'tail': 'loan'}]
[{'head': 'loan id', 'type': 'facet of', 'tail': 'Loan'}]
[{'head': 'bank', 'type': 'has part', 'tail': 'branch'}, {'head': 'branch', 'type': 'part of', 'tail': 'bank'}]
[{'head': 'library', 'type': 'has part', 'tail': 'book'}, {'head': 'library', 'type': 'has part', 'tail': 'book'}, {'head': 'book', 'type': 'part of', 'tail': 'library'}, {'head': 'book', 'type': 'part of', 'tail': 'library'}]
[{'head': 'library name', 'type': 'facet of', 'tail': 'Library'}]
[{'head': 'title', 'type': 'facet of', 'tail': 'Book'}, {'head': 'pages', 'type': 'part of', 'tail': 'Book'}]
[{'head': 'author name', 'type': 'used by', 'tail': 'Authors'}]
[{'head': 'patron name', 'type': 'used by', 'tail': 'Patron'}, {'head': 'patron weight', 'type': 'used by', 'tail': 'Patron'}]
[{'head': 'book', 'type': 'part of', 'tail': 'library'}]
[{'head': 'libraies', 'type': 'has part', 'tail': 'librais'}, {'head': 'librais', 'type': 'part of', 'tail': 'libraies'}]
[{'head': 'student number', 'type': 'different from', 'tail': 'Social Security number'}, {'head': 'Social Security number', 'type': 'different from', 'tail': 'student number'}]
[{'head': 'ZIP Code', 'type': 'applies to jurisdiction', 'tail': 'city'}]
[{'head': 'Social Security number', 'type': 'different from', 'tail': 'student number'}, {'head': 'student number', 'type': 'different from', 'tail': 'Social Security number'}]
[{'head': 'office number', 'type': 'different from', 'tail': 'office phone number'}, {'head': 'office phone number', 'type': 'different from', 'tail': 'office number'}]
[{'head': 'name', 'type': 'has part', 'tail': 'code'}, {'head': 'code', 'type': 'part of', 'tail': 'name'}]
[{'head': 'level', 'type': 'subclass of', 'tail': 'course'}]
[{'head': 'course number', 'type': 'part of', 'tail': 'course'}, {'head': 'course', 'type': 'has part', 'tail': 'course number'}]
[{'head': 'instructor', 'type': 'field of this occupation', 'tail': 'course'}, {'head': 'course', 'type': 'practiced by', 'tail': 'instructor'}]
[{'head': 'semester', 'type': 'part of', 'tail': 'year'}, {'head': 'year', 'type': 'has part', 'tail': 'semester'}]
[{'head': 'student', 'type': 'has part', 'tail': 'letter grade'}, {'head': 'letter grade', 'type': 'part of', 'tail': 'student'}]
[{'head': 'Zip Code', 'type': 'instance of', 'tail': 'unique employee number'}]
[{'head': 'customer number', 'type': 'subclass of', 'tail': 'unique'}]
[{'head': 'part number', 'type': 'subclass of', 'tail': 'unique'}]
[{'head': 'customer', 'type': 'has part', 'tail': 'employee'}, {'head': 'employee', 'type': 'part of', 'tail': 'customer'}]
[{'head': 'parts', 'type': 'part of', 'tail': 'order'}]
[{'head': 'expected ship date', 'type': 'subclass of', 'tail': 'date of receipt'}]
[{'head': 'ship', 'type': 'has part', 'tail': 'date'}, {'head': 'date', 'type': 'part of', 'tail': 'ship'}]
[{'head': 'title', 'type': 'part of', 'tail': 'year of release'}]
[{'head': 'minutes', 'type': 'subclass of', 'tail': 'length'}]
[{'head': 'horror', 'type': 'instance of', 'tail': 'genres'}, {'head': 'action', 'type': 'instance of', 'tail': 'genres'}, {'head': 'drama', 'type': 'instance of', 'tail': 'genres'}]
[{'head': 'director', 'type': 'subclass of', 'tail': 'actors'}]
[{'head': 'plot outline', 'type': 'facet of', 'tail': 'movie'}]
[{'head': 'quotable', 'type': 'subclass of', 'tail': 'quotes'}]
[{'head': 'Actor', 'type': 'field of this occupation', 'tail': 'movie'}, {'head': 'movie', 'type': 'practiced by', 'tail': 'Actor'}]
[{'head': 'role', 'type': 'practiced by', 'tail': 'actor'}]
[{'head': 'Director', 'type': 'field of this occupation', 'tail': 'movies'}, {'head': 'movies', 'type': 'practiced by', 'tail': 'Director'}]
[{'head': 'director', 'type': 'product or material produced', 'tail': 'movie'}]
[{'head': 'Production companies', 'type': 'has part', 'tail': 'address'}, {'head': 'address', 'type': 'part of', 'tail': 'Production companies'}]
[{'head': 'production company', 'type': 'product or material produced', 'tail': 'movies'}]
[{'head': 'e-mail id', 'type': 'instance of', 'tail': 'uniquely identified'}]
[{'head': 'First', 'type': 'different from', 'tail': 'last name'}, {'head': 'last name', 'type': 'different from', 'tail': 'First'}]
[{'head': 'title', 'type': 'part of', 'tail': 'abstract'}, {'head': 'abstract', 'type': 'has part', 'tail': 'title'}]
[{'head': 'contact author', 'type': 'subclass of', 'tail': 'author'}]
[{'head': 'e-mail address', 'type': 'subclass of', 'tail': 'uniquely identified'}]
[{'head': 'first name', 'type': 'different from', 'tail': 'last name'}, {'head': 'last name', 'type': 'different from', 'tail': 'first name'}]
[{'head': '2008 Summer Olympics', 'type': 'point in time', 'tail': '2008'}, {'head': '2008 Summer Olympics', 'type': 'location', 'tail': 'Beijing'}, {'head': '2008 Summer Olympics', 'type': 'country', 'tail': 'China'}, {'head': 'Beijing', 'type': 'country', 'tail': 'China'}, {'head': 'China', 'type': 'capital', 'tail': 'Beijing'}]
[{'head': 'technical merit', 'type': 'subclass of', 'tail': 'reviewer'}]
[{'head': 'recommendation', 'type': 'subclass of', 'tail': 'reviewer'}]
[{'head': 'written comments', 'type': 'subclass of', 'tail': 'feedback'}]
'''

for sent in sentences1:
    print(getRelation(sentences1))
exit(0)

sentences2 = sentences2.split('\n')
print(len(sentences2))
for s in sentences2:
    if s != '':
        print(getRelation(s))