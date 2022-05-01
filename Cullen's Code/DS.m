% Author -- Cullen Fitzgerald

% This file contains the final model used for the competition


trainData = readtable("C:\Users\Cullen\Downloads\train.csv\train.csv");
testData = readtable("C:\Users\Cullen\Downloads\test.csv\test.csv");

% Preprocess -- Map narratives to embedded vectors

% Create embedded vectors for narratives (test and train)
emb = fastTextWordEmbedding;

% Make embeddings and encodings for train data
wordsTrain = trainData.NARRATIVE;
wordsTokTrain = tokenizedDocument(wordsTrain);
enc = wordEncoding(wordsTokTrain);
encSeqTrain = doc2sequence(enc,wordsTokTrain);
sequencesTrain = doc2sequence(emb, wordsTokTrain, 'Length', 5);

% Make embeddings for test data
wordsTest = testData.NARRATIVE;
wordsTokTest = tokenizedDocument(wordsTest);
encSeqTest = doc2sequence(enc,wordsTokTest);
sequencesTest = doc2sequence(emb, wordsTokTest, 'Length', 5);

% Ok, now append this to the training and testing data
trainData.EMBEDDINGS = sequencesTrain;
testData.EMBEDDINGS = sequencesTest;

% Ok, now append this to the training and testing data
trainData.ENCODINGS = encSeqTrain;
testData.ENCODINGS = encSeqTest;

% Remove any rows with empty narratives
trainData = rmmissing(trainData);

% Ok, now train classifier off of the embeddings for test data

% This didn't work very well -- 60% accuracy -- Let's try something else

A = cell2mat(trainData.ENCODINGS);
T = cell2mat(testData.ENCODINGS);

trainData.ENCODINGS2 = A;
testData.ENCODINGS2 = T;

% Convert date data into strings 
trainData.BEGDATE = datestr(trainData.BEGDATE);

% Remove missing values (not optimal)
trainNoMis = rmmissing(trainData);

% Predict with Bagged Tree model -- best model so far
yfit = baggedTrees.predictFcn(testData);

submission = testData;
submission.CRIMETYPE = yfit;

writetable(submission, 'submission.csv');
