Objective
Use admission dataset above and train the model on a random 80% sample of the data and then cross-validate it against the other 20% to see how good the model is.

The way to do this is to separate the data into training and validation sets. Then for each target in the validation set, check if the prediction was correct (e.g. if the prediction was >.5 and the answer was 1 for admitted, then it was correct, but if the answer was 0 then it was wrong. Analogous for if the prediction was < .5).

Answer this question: how accurate is the model?
