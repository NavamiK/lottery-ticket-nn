# lottery-ticket-nn

The recent “Lottery ticket hypothesis” by Frankle and
Garbin [4] demonstrated a way to find trainable subnets of
neural networks that achieve same or better accuracy as the
original unpruned network. These networks, dubbed winning tickets, are identified by training a neural net, prun-
ing smallest-magnitude weights and resetting the remaining
weights to their original initializations. We examine if these
tickets are trainable only because it has seen the same training data in the previous pruning iteration. As the process of
uncovering a ticket is slow and tedious, we explore a faster
alternative by using a fraction of the dataset for pruning iterations and examine its performance when retrained with
the entire dataset. We observe that a speed-up of 7.5x can
be achieved by using subset (10%) of training data to generate winning tickets while achieving the same accuracy when
retrained on the full dataset. We also discover a winning
ticket for Shufflenet, a network architecture with 48 layers,
that makes use of depthwise separable convolutions.
