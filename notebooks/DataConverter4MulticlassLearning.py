#!/usr/bin/env python3
#Convert file format from that used by the AG's News Topic Classification
# Dataset ('newsSpace' files, downloadable as a bzip2 file from
# http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) into
# the format expected by the multiclass classifiers in 'Deep Learning for
# Natural Language Processing: A Gentle Introduction' (Surdeanu and
# Valenzuela-Escárcega, 2024).

#Run from directory "gentlenlp/notebooks/"; the subdirectory data/ag_news_csv
# is assumed to exist, and to contain the file newsSpace.bz2.

#Input format:
#Input is a single file, default name 'newsSpace.bz2', as provided by
# the unipi.it website.  The fields in the unzipped version of that file
# format are documented on that website; they are in tab-separated format.
# However, newlines in the original news articles are preserved, but escaped by
# a single backslash, so these newlines + '\' are replaced in the output by a
# space character.
# Note 1: The articles do not contain hyphenated morphemes, i.e. there are no
# words that are broken across lines by hyphens, with the exception of phrases
# which are ordinarily written with hyphens, like "full-term".  Depending
# on the tokenization used, this may result in such a phrase sometimes
# being represented by one token ("full-term") and sometimes by two tokens
# ("full-" and "term").  In the data downloaded in April 2025, this affects
# only a handful of phrases.
# Note 2: The book's programs expect four categories: World, Sports, Business,
# and Sci/Tech.  These categories are hard-coded in this program, but can be
# changed: see the definition of 'Classes' near the bottom of this file.
# We output equal numbers of records for each such category, and the total
# number of records to be output is 25,000, evenly split between train and
# test, and also evenly split among the four categories.
# Note 3: As of April 2025, the newsSpace file contains over a million records.
# Since the test and training programs require only 12,500 records each, this
# program only outputs that number of records.  However, input records
# containing non-Unicode characters (of which there are a few thousand) are
# skipped (with a warning).

#Output format:
# The data is output in three files, as required by the various multiclass
# learning programs.  The filenames are hardcoded here, since they are also
# hardcoded in the learning programs:
#train.csv: 12,500 records (lines), each consisting of three comma-separated
#     fields: class (as an int), title, and description.
# test.csv: 12,500 lines, same format as train.csv
# classes.txt: Hard coded as the names of four classes, one class per line.

#Bugs:
#1:The data comes from multiple news sources.  It is not clear that all these
#  sources use the same criteria for categorizing articles into World, Sports,
#  Business and Sci/Tech.  Further, the article describing how the data was
#  processed says that some of the tags were supplied by Bayesian inference,
#  rather than (presumably) human taggers.  The result is visible inconsistency
#  in tagging.  For example, the story "British leader defends war amid sinking
#  ratings" appears five times in the source data from two different sources
#  (theglobeandmail.com and seattlepi.newsource.com), with three different tags:
#  Entertainment, Top Stories, and World.  As it happens, this program only
#  accepts the 'World' tag (because the target programs from GentleNLP only use
#  that tag).  So in this particular case, the inconsistent tagging doesn't
#  matter; but it does raise the question of how consistently tags were used
#  in the dataset.

#Written by Mike Maxwell (mmaxwell@umd.edu).  Freely licensed.


from math import ceil #Used to round up numbers of records
from bz2 import open as bzopen
from lazyreader import lazyread
from csv import writer as CSVWriter
from random import shuffle
from sys import stderr  #For warning msgs about data


def ProcessRecord(sInRecord, iLine, dictClasses):
    """Process a single valid record from the input file, and return the desired
       fields of the record in a list.  The desired columns of the input record
       are hard-coded here.
       If the record is invalid or doesn't belong to one of the specified
       classes in dictClasses, return None.
       The iLine arg is used in the warning message.
    """
    Fields = sInRecord.split('\t')
    try:
        sClass = Fields[4]
        sTitle = Fields[2]
        sDescription = Fields[5]
    except IndexError:
        stderr.write(f"Skipping record ending at line {iLine} because missing field(s).\n")
        return None
    if sClass in dictClasses:
        return [sClass, sTitle, sDescription]
    else:
        return None


def GetRecords(fNewSpace, iRecordsPerClass, dictClasses):
    """Find exactly iRecordsPerClass valid records for each class in dictClasses
       from file fNewSpace, shuffle them together and return them them as
       a tuple of lists of training and test records.
       We only return records whose class matches one of the target classes
       in dictClasses, and then only if there are no errors in the record
       (currently we check for Unicode errors, of which there are many--some
       of the older records appear to be non-Unicode).
       If we can't obtain iRecordsPerClass records for each class, we raise
       a warning, and return a tuple (None, None).
    """
    #Prepare a dictionary to record how many records we've found for each class:
    dictRecordsFound = {}
    for sClass in dictClasses:
        dictRecordsFound[sClass] = 0
    TestRecords  = []
    TrainRecords = []
    iAllRecords  = 0
    sInRecord    = ""
    bValidRecord = True #Will be set to False within a record if that record
        # contains an invalid Unicode char (and set back to True at the end
        # of that record).
    for (iLine, bLine) in enumerate(lazyread(fNewSpace, delimiter=b"\n")):
        try:
            sLine = bLine.decode('utf-8').rstrip()  #Remove trailing nl
        except UnicodeDecodeError as Error:
            #Inform the user of the problem:
            stderr.write(f"Skipping record: Unicode error:"
                         f" {Error.reason}:"
                         f" {Error.object[Error.start:Error.end]}"
                         f" at line {iLine}\n")
            #...flag this record as invalid:
            bValidRecord = False
            #...and create a dummy Unicode-compliant line:
            sLine = ""
        #Check for continuation line vs. end of record using the binary
        # form, in case this wasn't a valid record:
        if bLine.endswith(b'\\\n'): #Not yet at the end of this record
            sInRecord = sInRecord + " " + sLine.rstrip('\\')  #Omit the trailing '\'
        else: #End of record
            if bValidRecord:
                sInRecord = sInRecord + " " + sLine
                OutRecord = ProcessRecord(sInRecord, iLine, dictClasses)
                if OutRecord: #ProcessRecord() returns None if invalid record
                              # or if it doesn't belong to one of dictClasses
                    dictRecordsFound[OutRecord[0]] += 1
                    if dictRecordsFound[OutRecord[0]] <= iRecordsPerClass:
                        #Only retain this record if we need more for this class
                        iAllRecords += 1
                        sClass = OutRecord[0]
                        #Replace the class with the class ID for output:
                        OutRecord[0] = dictClasses[OutRecord[0]]
                        if dictRecordsFound[sClass] % 2 == 0:
                            TestRecords.append(OutRecord)
                        else:
                            TrainRecords.append(OutRecord)
                    if iAllRecords >= 4 * iRecordsPerClass:
                        #We're all done collecting records; shuffle them, in
                        # case there's variability over time in what was
                        # collected:
                        shuffle(TestRecords)
                        shuffle(TrainRecords)
                        return (TestRecords, TrainRecords)
            #Whether that record was valid or not, set up for next record:
            sInRecord = ""
            bValidRecord = True
    #If we get here, we probably didn't find enough records.  Raise a warning,
    # and return a tuple of Nones:
    for sClass in dictRecordsFound:
        if dictRecordsFound[sClass] < iRecordsPerClass:
            stderr.write(f"Only found {dictRecordsFound[sClass]} valid records"
                         f" for class {sClass}.\n")
        return (None, None)


#---------------------Main---------------------
dictClasses = {'World':1, 'Sports':2, 'Business':3, 'Sci/Tech':4}
    #1-based, as expected by target programs
iRecordsPerClass = ceil(25000 / len(dictClasses))
    #iRecordsPerClass gives the intended number of records for each of the
    # classes in dictClasses.
    # We will quit once we have found at least iRecordsPerClass for each class,
    # then truncate the records for each class to that number so the classes are
    # evenly distributed.  (This includes both training and test records, which
    # we will evenly split.)
sDirPath = './data/ag_news_csv/'
    #This hard coded path is the same as that used in the learning programs.
#First write out the classes to a file:
with open(sDirPath + 'classes.txt', mode='w') as fClasses:
    fClasses.write('\n'.join(list(dictClasses)))
#Then read the data in from the bzipped file, and write out the training and
# test data to the respective output files:
with bzopen(sDirPath + 'newsSpace.bz2', mode='rb') as fNewSpace, \
     open(sDirPath + 'train.csv', mode='w') as fTraining, \
     open(sDirPath + 'test.csv', mode='w') as fTest:
    #Open the .bz2 file as bytes, rather than text, because there are invalid
    # Unicode chars, and opening as text raises an error that we can't get by--
    # instead, we deal with Unicode errors in the function GetRecords():
    (TestRecords, TrainRecords) = \
        GetRecords(fNewSpace, iRecordsPerClass, dictClasses)
    #Now write the records out as two CSV files, one for test and one for train:
    if TestRecords:
        #If there weren't enough records for at least one class, GetRecords()
        # will have returned None for TestRecords (and for TrainRecords)
        CSVWriter(fTraining).writerows(TestRecords)
        CSVWriter(fTest).writerows(TrainRecords)
