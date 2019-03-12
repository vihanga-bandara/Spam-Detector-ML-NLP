import pandas as pd
import csv

## THIS IS FOR ONE TYPE OF DATASET PROCESSING
# # open tweets containing file for processing
# with open("spamtweets(socalled).txt") as fp:
#     # create two arrays that store the split values
#     my_list = list()
#     my_list2 = list()
#
#     # enumerate lines to get the tweets
#     for i, line in enumerate(fp):
#         my_list = line.split(",")
#         sizeOfArray = len(my_list)
#         # get the target element dynamically - 12th Element of each line after splitting
#         targetElem = sizeOfArray - 4
#         # write to a new list and the label
#         my_list3 = my_list[targetElem].strip().split("\'")
#         targetElem = len(my_list3) - 2
#         new_line = my_list3[targetElem]
#         # print(new_line)
#         my_list2.append([new_line, "spam"])
#         print(i)
#     # check if csv file is empty or not
#     df = True
#     if df:
#         # write the list to a CSV file
#         with open('Tweets(Spam).csv', 'w') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerows(my_list2)
#     else:
#         # write the list to a CSV file
#         with open('Tweets(Spam).csv', 'w') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerow(my_list2)
#     csvFile.close()

# Another type of dataset processing
# open tweets containing file for processing
with open("full-corpus.csv") as fp:
    # create two arrays that store the split values
    my_list = list()
    my_list2 = list()

    # enumerate lines to get the tweets
    for i, line in enumerate(fp):
        my_list = line.split(",")
        sizeOfArray = len(my_list)
        # get the target element dynamically - 12th Element of each line after splitting
        targetElem = sizeOfArray - 1
        # write to a new list and the label
        new_line = my_list[targetElem].strip()
        new_line = new_line.replace('"', '')
        # print(new_line)
        my_list2.append([new_line, "ham"])
        print(i)
    # check if csv file is empty or not
    df = True
    if df:
        # write the list to a CSV file
        with open('Tweets(Normal_from_full_corpus.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(my_list2)
    csvFile.close()
