# Final Project
By : Connor Laughland, Jaehyun Yoon, Stratis Aloimonos

# Introduction
The point of the experiment was to use written data, taken from posts made
on Twitter by 250 different people, to identify whether these people were con-
sidered neurotic. The data came from a dataset called myPersonality. This
dataset includes the user ID's of people who post, the posts themselves, the nu-
meric (numbers between 1 and 5) and nominal (numbers converted to yes/no)
measures of Big-5 personality traits (OCEAN: openness, conscientiousness, ex-
traversion, agreeableness and neuroticism), as well as the date the post was
made, among other statistics about the posts.

# 6 Conclusions
#### 6.1 Classifiers
Based on these experiments, which are simple to understand and perform, one
can get an idea of how difierent classifiers perform for the data. One of the first
choices we had was whether to split or combine the user posts. It is worth noting
that when the posts are not combined into one post per user, the accuracies of
the SVM with a linear kernel, the SVM with an RBF kernel and the logistic
regressor are roughly the same, but when they are combined into one post per
user, the SVM with an RBF kernel shows a noticeably higher accuracy. With
that said, all of the classifiers performed better when the posts were separated,
suggesting that the consolidation of their posts might result in posts that are
easy to classify being diluted by other posts that are harder to classify. A
next step for analysis might be to compare post classifications for users between
consolidation and separation, and identify those posts that might be affecting
the classification accuracy.<br>

### 6.2 Feature Detection

The different feature detection methods revealed many word categories and
ngrams that are intuitively associated with neuroticism. However, it's clear by
our inability to yield a classification accuracy higher than approximately 60
percent (for consolidated posts) that while those features might exist, the post
won't be classified correctly if they aren't present. That speaks to the broader
issue of social media posts having such wildly different topics, and being so
dependent upon current events and subcultures, that personality assessments
are dificult to make based of of arbitrary moments in time.<br>

### 6.3 Limitations
One of the greatest limitations posed by the MyPersonality data was the lack
of clear criteria for what kinds of posts warrant the classification of neuroti-
cism. Because of this experiment's exploratory nature, the indicators of neuroti-
cism weren't guaranteed to exist within the data. We had to test to determine
15 whether they exist. So, despite how many combinations of feature vectors and
classifiers we paired, we couldn't force an association that didn't exist. This
seems to be the explanation for why BERT, which is supposed to be one of
the higher end NLP text classification models, could only reach a maximum
evaluation accuracy of 60 percent, no better than the other classifiers we used. <br>
One might wonder whether classification would be improved by a larger amount
of data for each user, or by each user's data being filtered down to particular
kinds of posts that might be better indicators of neuroticism. For example, if
posting about something regularly and obsessively is an indicator, you could
easily take a large series of posts from an individual user and count how many
times, and with what language, they chose to discuss it. Similarly, if emotional
instability is an indicator, one could try to track their emotional 
uctuations (positive vs. negative emotions) and turn that into a feature metric. The
intersection of data analysis and psychology poses a fascinating potential for
better identifying self-destructive and life threatening behaviors, independent
of the ethical implications (for which there are many!).
