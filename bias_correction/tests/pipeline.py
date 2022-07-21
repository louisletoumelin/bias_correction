# Things to tests

"""
https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn

1. Variables are created but never used (usually because of copy-paste errors);
2. Expressions for gradient updates are incorrect;
3. Weight updates are not applied;
4. The loss is not appropriate for the task (for example, using categorical cross-entropy loss for a regression task).
5. Dropout is used during testing, instead of only being used for training.
6. Check that loss is correct
7. NaN in input data
8. Creating NaN in the network
9. Accidentally assigning the training data as the testing data;
10. When using a train/tests split, the model references the original, non-split data instead of the training partition or the testing partition.
11. Forgetting to scale the testing data
12. Scaling the testing data using the statistics of the tests partition instead of the train partition;
13. Forgetting to un-scale the predictions (e.g. pixel values are in [0,1] instead of [0, 255]).

14. Scale data
15. Batch or layer normalization

16. Network initialization
17. Activation function: leaky relu?
https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons

18. Residual connections

19. Learning rate
20. Gradient clipping
21. Learning rate scheduling
22. Optimizer
23. Add regularization after
"""