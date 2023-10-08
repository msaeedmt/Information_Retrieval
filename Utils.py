import numpy as np
import matplotlib.pyplot as plt
import math
import heapq


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
        firstWordIndexes = queryPostingList[firstWord].postingsList[document].occurrences

        for firstWordIndex in firstWordIndexes:
            isOrderedMatched = True
            currentIndex = firstWordIndex + 1

            for i in range(1, len(userQueryWordsList)):
                currentWord = userQueryWordsList[i]
                currentWordIndexesListInDocument = queryPostingList[currentWord].postingsList[document].occurrences
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


def mergeSort(arr):
    if len(arr) > 1:

        mid = len(arr) // 2

        L = arr[:mid]

        R = arr[mid:]

        mergeSort(L)

        mergeSort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] > R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


def countWordFreq(dict):
    count = 0
    for doc in dict:
        count += len(dict[doc])
    return count


def estimate_coef(x, y):
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    y_pred = b[0] + b[1] * x

    plt.plot(x, y_pred, color="g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def calcLTC(tf_td, df_t, docsCount):
    if tf_td == 0:
        return 0

    w_td = (1 + math.log(tf_td, 10)) * (math.log(docsCount / df_t))
    return w_td


def calcLNC(tf_td):
    if tf_td == 0:
        return 0

    w_td = 1 + math.log(tf_td, 10)
    return w_td


def FirstKelements(arr, k):
    # indices = np.array(arr).argsort()[-k:][::-1]
    # for index in indices:
    #     print(arr[index])
    # return indices

    indices = list(map(lambda x: x[0], heapq.nlargest(k, list(enumerate(arr)), key=lambda x: x[1])))
    return indices

def docsSimilarity(doc1,doc2):
    similarityScore = np.dot(doc1,doc2)/(np.linalg.norm(doc1)*np.linalg.norm(doc2))
    return (similarityScore+1)/2
