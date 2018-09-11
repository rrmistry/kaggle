
import traceback
import datetime
import csv
import os
import shutil

BATCH_SIZE = 1000000
FILE_PATH = r'../input/'
FILE_NAME = r'train'
FILE_EXTN = r'.csv'

TRAIN_TEST_SPLIT_RATIO = 0.8

# Read main input file (e.g. 'train.csv' ~5.6 Gb)
with open(FILE_PATH + FILE_NAME + FILE_EXTN, "r") as csvinputfile:

    datareader = csv.reader(csvinputfile)

    # Duplicate header for all batch files
    headerRow = next(datareader)

    print('Header Row: ', headerRow)

    batchFilePath = FILE_PATH + FILE_NAME + '\\'
    trainBatchFileName = ''
    testBatchFileName = ''

    batchCount = 0

    row = headerRow

    rowsWritten = 1

    if os.path.exists(batchFilePath):
        shutil.rmtree(path=os.path.dirname(batchFilePath))
    os.makedirs(os.path.dirname(batchFilePath))

    def WriteToBatchFile(batchFileName, outputSize, row):

        rowsWritten = 0

        with open(batchFileName, 'w+', newline='') as csvTrainOutputFile:
            datawriter = csv.writer(csvTrainOutputFile)

            datawriter.writerow(headerRow)

            for _ in range(outputSize):

                try:
                    row = next(datareader)
                    rowsWritten = rowsWritten + 1
                except StopIteration:
                    row = None
                    break

                datawriter.writerow(row)

        return rowsWritten

    print ('Starting Processing: ', datetime.datetime.now())

    while rowsWritten > 0:

        print ('Processing batch: ', batchCount)

        # Write train file (first x % of the batch for training, e.g. 80%)
        batchFileName = batchFilePath + 'train-' + str(batchCount) + FILE_EXTN
        rowsWritten = WriteToBatchFile(batchFileName, int(BATCH_SIZE * TRAIN_TEST_SPLIT_RATIO), row)

        # Write test file (next (1-x) % of the batch for testing, e.g. 20%)
        batchFileName = batchFilePath + 'test-' + str(batchCount) + FILE_EXTN
        rowsWritten = WriteToBatchFile(batchFileName, int(BATCH_SIZE * (1-TRAIN_TEST_SPLIT_RATIO)), row)

        batchCount  = batchCount + 1

    print ('Finished Processing: ', datetime.datetime.now())
