import json
from hazm import *
from PostingList import PostingList
import pickle


def findQuery(fullQueryPostingList, userQueryWordsList):
    queryPostingList = {}
    uniqueQueryWords = []

    for word in userQueryWordsList:
        if word in fullQueryPostingList:
            queryPostingList[word] = fullQueryPostingList[word]
            if word not in uniqueQueryWords:
                uniqueQueryWords.append(word)

    firstPhrase = list(queryPostingList[findMinReaptedWord(queryPostingList, uniqueQueryWords)].postingsList.keys())
    for i in range(len(uniqueQueryWords)):
        secondPhrase = list(
            queryPostingList[findMinReaptedWord(queryPostingList, uniqueQueryWords)].postingsList.keys())
        firstPhrase = subscriptPostingsList(firstPhrase, secondPhrase)

    allDocumentsWithQueryWords = firstPhrase
    documentsWithOrderedQueryWords = []
    for document in allDocumentsWithQueryWords:

        firstWord = userQueryWordsList[0]
        firstWordIndexes = queryPostingList[firstWord].postingsList[document]

        for firstWordIndex in firstWordIndexes:
            isOrderedMatched = True
            currentIndex = firstWordIndex + 1

            for i in range(1, len(userQueryWordsList)):
                currentWord = userQueryWordsList[i]
                currentWordIndexesListInDocument = queryPostingList[currentWord].postingsList[document]
                if currentIndex not in currentWordIndexesListInDocument:
                    isOrderedMatched = False
                    break
                else:
                    currentIndex += 1

            if isOrderedMatched:
                documentsWithOrderedQueryWords.append(document)
                break

    # print(allDocumentsWithQueryWords)
    # print(documentsWithOrderedQueryWords)
    return set(allDocumentsWithQueryWords), set(documentsWithOrderedQueryWords)


def findMinReaptedWord(dict, uniqueWords):
    minRepeatedWord = uniqueWords[0]
    for word in uniqueWords:
        if dict[word].numberOfDocs < dict[minRepeatedWord].numberOfDocs:
            minRepeatedWord = word

    uniqueWords.remove(minRepeatedWord)
    return minRepeatedWord


def subscriptPostingsList(firstPostingsList, secondPostingsList):
    i = 0
    j = 0
    subscriptions = []
    while i < len(firstPostingsList) and j < len(secondPostingsList):
        if int(firstPostingsList[i]) == int(secondPostingsList[j]):
            subscriptions.append(firstPostingsList[i])
            i += 1
            j += 1
        elif int(firstPostingsList[i]) > int(secondPostingsList[j]):
            j += 1
        else:
            i += 1
    return subscriptions


# Opening JSON file

with open('titlesDict', encoding='utf-8') as json_file:
    positionalIndexDict = {}
    titlesDict = json.load(json_file)

print('making postinsIndex ...')
with open('dataDict', encoding='utf-8') as json_file:
    data = json.load(json_file)
    # positionalIndexDict = {}
#
#     normalizer = Normalizer()
#     stemmer = Stemmer()
#
#     for doc in data:
#         # for stopWord in stopwords_list():
#         #     data[doc] = data[doc].replace(stopWord, "")
#
#         data[doc] = normalizer.normalize(data[doc])
#         docWordsList = word_tokenize(data[doc])
#         for stopWord in stopwords_list():
#             docWordsList = list(filter(stopWord.__ne__, docWordsList))
#         # print(docWordsList)
#
#         for i in range(len(docWordsList)):
#             # if docWordsList[i] in stopwords_list():
#             #     continue
#
#             docWordsList[i] = stemmer.stem(docWordsList[i])
#             if docWordsList[i] in positionalIndexDict:
#                 if doc in positionalIndexDict[docWordsList[i]].postingsList:
#                     positionalIndexDict[docWordsList[i]].postingsList[doc].append(int(i + 1))
#                 else:
#                     newPostingList = [int(i + 1)]
#
#                     positionalIndexDict[docWordsList[i]].postingsList[doc] = newPostingList
#                     positionalIndexDict[docWordsList[i]].numberOfDocs += 1
#             else:
#                 positionalIndexDict[docWordsList[i]] = PostingList()
#
#                 newPostingList = [int(i + 1)]
#
#                 positionalIndexDict[docWordsList[i]].postingsList[doc] = newPostingList
#                 positionalIndexDict[docWordsList[i]].numberOfDocs += 1
#
#     # for stopWord in stopwords_list():
#     #     if stopWord in positionalIndexDict:
#     #         positionalIndexDict.pop(stopWord)
#     print('Serializeing postingIndex')
#     outputFile = open('PostingList.dat', 'wb')
#     pickle.dump(positionalIndexDict, outputFile)
#     outputFile.close()

print('postingIndex created successfully!')

stemmer = Stemmer()
normalizer = Normalizer()
lemmatizer = Lemmatizer()

inputFile = open('PostingList.dat', 'rb')
postingIndex = pickle.load(inputFile)
inputFile.close()

# test = postingIndex[stemmer.stem('مسلم')].postingsList.keys()
# print(test)

while True:
    userQuery = input("Please enter your query > ")
    userQuery = normalizer.normalize(userQuery)
    splitedUserQuery = userQuery.split()
    splitedUserQueriesWithoutStopWords = []
    for word in splitedUserQuery:
        if word not in stopwords_list():
            splitedUserQueriesWithoutStopWords.append(word)

    userQueryWordsList = list(map(lambda item: stemmer.stem(item), splitedUserQueriesWithoutStopWords))

    print(userQueryWordsList)

    if len(userQueryWordsList) == 0:
        print('All documents')
    elif len(userQueryWordsList) == 1:
        if userQueryWordsList[0] in postingIndex:
            print(postingIndex[userQueryWordsList[0]].postingsList.keys())
        else:
            print("No result found!")
    else:
        allWordExistenceDocuments = {}
        allReleventResultsWithOrderedSubQueries = []

        fullQueryPostingList = {}
        for word in userQueryWordsList:
            if word in postingIndex:
                fullQueryPostingList[word] = postingIndex[word]

        allWordExistenceDocuments, releventResultsWithOrderedSubQueries = findQuery(fullQueryPostingList,
                                                                                    userQueryWordsList)
        allReleventResultsWithOrderedSubQueries.append(releventResultsWithOrderedSubQueries)

        for i in range(1, len(userQueryWordsList) - 1):
            releventResultsWithOrderedSubQueries = set()
            for j in range(i + 1):
                print(userQueryWordsList[j:(len(userQueryWordsList) - i + j)])
                currentSubQueryReleventdocs = \
                    findQuery(fullQueryPostingList, userQueryWordsList[j:(len(userQueryWordsList) - i + j + 1)])[1]
                print(currentSubQueryReleventdocs)
                releventResultsWithOrderedSubQueries.update(currentSubQueryReleventdocs)

            for releventSubQuerySet in allReleventResultsWithOrderedSubQueries:
                releventResultsWithOrderedSubQueries = releventResultsWithOrderedSubQueries - releventSubQuerySet

            allReleventResultsWithOrderedSubQueries.append(releventResultsWithOrderedSubQueries)

        print()
        print('allReleventResultsWithOrderedSubQueries : ', allReleventResultsWithOrderedSubQueries)
        print('allWordExistenceDocuments : ',allWordExistenceDocuments)

        # with open('dataDict', encoding='utf-8') as json_file:
        #     data = json.load(json_file)

        normalizer = Normalizer()
        stemmer = Stemmer()

        releventDocumentsCount = 0
        for i in range(len(allReleventResultsWithOrderedSubQueries)):
            if releventDocumentsCount == 10:
                break
            for docId in allReleventResultsWithOrderedSubQueries[i]:
                if releventDocumentsCount == 10:
                    break
                else:
                    releventDocumentsCount +=1

                documentContent = data[docId]
                realTokenizedSentences = sent_tokenize(documentContent)
                normalizedTokenizedSentences = []

                for j in range(len(realTokenizedSentences)):
                    normalizedSentence = normalizer.normalize(realTokenizedSentences[j])
                    splitedSentence = word_tokenize(normalizedSentence)
                    for stopWord in stopwords_list():
                        splitedSentence = list(filter(stopWord.__ne__, splitedSentence))
                    for t in range(len(splitedSentence)):
                        splitedSentence[t] = stemmer.stem(splitedSentence[t])
                    normalizedTokenizedSentences.append(' '.join(splitedSentence))

                foundedSentencesIndexes = set()

                for j in range(i + 1):
                    print('\n',userQueryWordsList[j:(len(userQueryWordsList) - i + j)])
                    subQuery = ' '.join(userQueryWordsList[j:(len(userQueryWordsList) - i + j)])
                    for t in range(len(normalizedTokenizedSentences)):
                        if subQuery in normalizedTokenizedSentences[t]:
                            foundedSentencesIndexes.add(t)

                print( docId, titlesDict[docId])
                for index in foundedSentencesIndexes:
                    print(realTokenizedSentences[index])

                print('\n\n')
