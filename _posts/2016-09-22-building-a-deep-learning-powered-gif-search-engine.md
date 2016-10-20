---
layout: post
title: "Building a Deep Learning Powered GIF Search Engine"
description: "When all you have is deep learning, everything looks like a dataset. Deep learning driven GIF search with convolutional and recurrent neural networks, a shallow embedding matrix, and fast-approximate nearest neighbor search."
date: 2016-09-22
tags: []
comments: true
share: true
---

They say a picture's worth a thousand words, so GIFs are worth at least an order of magnitude more. But what are we to do when the experience of finding the right GIF is like searching for the right ten thousand words in a library full of books, and your only aide is the [Dewey Decimal System](http://giphy.com/categories)?

![Searching](https://media4.giphy.com/media/l0HlAgJTVaAPHEGdy/giphy.gif)

Why we build a thesaurus of course! But where a thesaurus translates one word you know into another you don't know, we'll take a sentence you know, and translate it into a GIF you quite likely don't!

The experience we're looking for is one where a user inputs a sentence, and gets the GIF that would be best described by that sentence. This best matches the ideal of having watched every GIF on the internet, and being able to recall which one is most appropriate from memory (I swear some Redditors come quite close)

I imagined it would look something like this:

![Ideal Search](https://media4.giphy.com/media/iqW0TbyuKKCMo/giphy.gif)

And without further ado, here's the final result:
[DeepGIF](http://deepgif.tarzain.com/)

# How does it work?

![Main Figure](/images/mainfig.001.png "Main Figure")

## Overview

I use pretrained models and existing datasets liberally here. By formulating the problem generally enough we can get away with only having to learn a shallow neural network model that embeds the hidden layer representations from two pretrained models - the VGG16 16-layer CNN pretrained on ImageNet, and the Skip Thoughts GRU RNN pretrained on the BooksCorpus - into a joint space where videos from the Microsoft Video Description Corpus and their captions are close together. This constrains the resulting model to only working well with live-action videos that are similar to the videos in the MVDC, but the pre-trained image and sentence models help it generalize to pairings in that domain it has never before seen.

If you're unfamiliar with much of the CNN-RNN image captioning work - there are dozens of thorough, interesting and well written posts on the specific machinery behind deep learning I will direct you to.
Instead I'll only cover how they work at a high level and how they fit together to make this GIF search engine work. Watch out for a subsequent post explaining the nitty gritty details with a different perspective soon.

Like I said above, there are three pieces we put together to make this work. First, we have a convolutional neural network, pretrained to classify the objects found in images. At a high level, a convolutional neural network is a deep neural network with a specific pattern of parameter reuse that enables it to scale to large inputs (read: images). Researchers have trained convolutional neural networks to exhibit near-human performance in classifying objects in images, a landmark achievement in computer vision and artificial intelligence in general.

Now what does this have to do with GIFs? Well, as one may expect, the "skills" learned by the neural network in order to classify objects in an image should generalize to other tasks requiring understanding images. If you taught a robot to tell you what's in an image for example, and then started asking it to draw the boundaries of such objects (a vision task that is strictly harder, but requires much of the same knowledge), you'd hope it would pick up this task more quickly than if it had started on this new task from scratch. 

## Transfer Learning

We can use an understanding of how neural networks function to figure out exactly how to achieve that. Deep neural networks are so called because they contain layers of composed pieces - each layer is simply a matrix multiplication followed by some sort of function, like $$ f(x) = \frac{1}{1+e^{-x}} $$ or $$ f(x) = \tanh{(x)} $$. This means, for a given input, we multiply it by a matrix, then pass it through one of those functions, then multiply it by another matrix, then pass it through one of those functions again, until we have the numbers we want. In classification, the numbers we want are a probability distribution over classes/categories, and this is necessarily far fewer numbers than in our original input.

It is well understood that matrix multiplications simply parametrize transformations of a space of information - e.g. for images you can imagine that each matrix multiplication warps the image a bit so that it is easier to understand for subsequent layers, amplifying certain features to cover a wider domain, and shrinking others that are less important. You can also imagine that, based on the shape of the common activation functions (they "saturate" at the limits of their domain from $$ {-\infty} \, to \, {+\infty} $$, and only have a narrow range around their center when they aren't strictly one number or another), they are utilized to "destroy" irrelevant information by shifting and stretching their narrow range of non-linearity to the region of interest in the data. Further, by doing this many times rather than only once, the network can combine features from disparate parts of the image that are relevant to one another.

When you put these pieces together what you really have is a simple, but powerful, layered machine that destroys, combines, and warps information from an image until you only have the information relevant to the task at hand. When the task at hand is classification, then it transforms the image information until only the information critical to making a class decision is available. We can leverage this understanding of the neural network to realize that just prior to the layer that outputs class probabilities we have a layer that does *most* of the dirty work in understanding the image *except* reducing it to class labels.

For classification problems like the one we just discussed, our last layer's output needs to have dimensionality equal to the number of available classes, because we want it to output a distribution of probabilities from 0 to 1 for each of these classes (where 1 is complete certainty that the object in the image is of that class, and 0 is total certainty that it isn't). But the layer just prior to that one contains all of the information required to make that decision (what object is in this image) and likely a bit more (it has a higher dimensionality, so unless our computer has perfect numerical precision information is being lost in the transformation to object classes).

We leverage this understanding to reuse learned abilities from this previous task, and thereby to generalize far beyond our limited training data for this new task. More concretely, for a given image, we recognize that this penultimate layer's output may be a more useful representation than the original (the image itself) for a new task if it requires similar skills. For our GIF search engine this means that we'll be using the output of the [VGG-16 CNN from Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) trained on the task of classifying images in the [ImageNet](http://image-net.org/) dataset as our representation for GIFs (and thereby the input to the machine learning model we'd like to learn).

## Representing Text
What is the equivalent representation for sentences? That brings us to the second piece of our puzzle, the SkipThoughts GRU (gated-recurrent-unit) RNN (recurrent neural network) trained on the Books Corpus.

large inputs (read: images) by reusing model capacity with weight sharing across the image. 
* neural networks are a 'simple' (relatively) machine learning algorithm, consisting of subsequent matrix multiplications with "activation" functions interleaved.
* typically, a neural network trained to, let's say, classify a black and white image of a digit as a particular number would take as input the image data (black or white brightness values for every pixel in the image)

To learn more about how convolutional neural networks work in detail, check out Chris Olah's fantastic post [here](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/). 

## Search

We frame the problem as one of ranking/search. Naively, all we would need for this is a function that will, for a given query (in our case, a sentence description of the GIF we're looking for), return the relevance score of a particular GIF with that query (i.e. how likely is this GIF to be described by this sentence?)

Naively, we could apply the function to the query sentence and every image in our dataset, then return the highest relevance one. As simple as that sounds, it would be quite computationally expensive/time consuming. Obviously we'd have to do some computation for every sentence and every image at some point, but ideally we'd frontload the computation / memoize it so it doesn't need to happen every time someone searches for a GIF.

## Distance Ranking

In data analysis parlance, the typical solution to a problem like this is called projection/embedding. Rather than redundantly running this function on every image and query pair, we can split the function into 3 parts: a complex function we run once for every sentence, a complex function we run once for every image, and a simple function we can run (at search time) for every image-sentence pair. If the simple function we run for every image-sentence pair is a distance function, we can understand our process as projecting points in the space of images and the space of sentences into a space in which they both exist, which we'll call the multimodal embedding space. Then for a given sentence, the GIF most likely to be described by it is the one that is closest to it in this multimodal embedding space.

![Measuring Distance](https://media4.giphy.com/media/RtZag69AFzTRm/giphy.gif)

## Projection and Embedding

Now we unfortunately don't already have this embedding space where GIFs and the sentences that describe them are near each other (wouldn't that be swell), but we can make that our optimization criterion, and use it to learn, from some examples of course, a set of transformations from GIFs and sentences into a common space.

Again, the machinery behind this deserves a post of its own, so I'll direct you to a few blog posts by others in the field if you're looking for more detail on the implementations, but I'll cover how they work at a high level and how they fit together to make this GIF search engine work. Watch out for a separate post explaining the nitty gritty details soon.

# Dense Representations

Now given we have GIFs and sentences that describe them, how do we actually compare them to each other? Neural networks are famously effective as arbitrary function approximators (given sufficiently many training examples) - can we train one to put GIFs and sentences into a joint space? Well, it turns out we can - so long as we craft our objective properly.




# Transforming Numbers Between Spaces

We'll use the intuition of spaces of numbers, and transforms between them, to discuss all of the machinery and how it combines to enable simply incredible things. This may seem a bit long-winded, but I think this is very important to understand and thereby deserves a thorough explanation.

First of all we're concerned with two domains, GIFs and sentences. GIFs are 3D images, just as images are 2D - they are arrays of numbers with a width and height - GIFs are arrays of arrays with dimensions: width, height, and time. Video clips and movies are also 3D arrays, while a song is 1D (it's just a sequence of amplitudes, where the only dimension is time). This general characterization of the data as simply N-dimensional arrays will be very important, it allows us to not only do operations on a piece of media but also to make associations between them.

![It's all numbers](https://media4.giphy.com/media/v7WM6sLcnGIc8/giphy.gif)

While characterizing a GIF as a 3D array may be somewhat intuitive, transforming sentences into the same general space won't be. This is because while you can easily treat the brightness/color of a pixel in an image as a number on some range, the same doesn't seem as intuitive for words. Words are _discrete_ while the colors of pixels are _continuous_.

## Discrete vs Continuous
What does this mean? Roughly, it means that the space is fully occupied - there is always a value in between some other values. Take two different color values for a pixel; there is always a color value in between them. We can't say the same for words. Where there are infinitely many colors "between" [blue-green block] and [yellow-green block], we can't say the same for the words between "dog" and "cat". Now note, this doesn't mean that having more words would make it continuous (though it does help bridge the gap) - instead it means that the words are distinct from one another, like categories.

I like to think of the difference somewhat like that between buttons and dials on a boombox. There are some inputs that just make more sense as buttons than dials (on vs off, radio tuner vs auxiliary input mode, etc.) and some make more sense as dials than buttons (volume, frequency, bass level, etc.).

##### Some more examples of discrete, categorical spaces include:
* Blood Types (A, B, AB, or O)
* States (CA, PA, ID, HI)
* Political party
* Dog breed

Now importantly (and necessarily for our application), these definitions aren't as incontrovertible as they may seem. While the words themselves are certainly distinct, they represent ideas that aren't necessarily so black and white. There may not be any words between cat and dog, but we can certainly think of concepts between them. And for some pairs of words, there actually are plenty of words in between (like the word warm between hot and cold).

![Colors vs. Color Names](/images/DeepGIF_figures.003.png "Colors vs. Color Names")

Back to colors, the space of colors is certainly continuous, but the space of named colors is not. There are infinitely many colors between black and white, but we really only have a few words for them (grey, steel, charcoal, etc.). What if we could find that space of colors from the words for them, and use that space directly? We would only require that the words for colors that are similar are also close to each other in the color space. Then, despite the color names being a discrete space, for any operations we want to do on or between the colors we can convert them to the continuous space first!

## Word Vectors

Our method for bringing the discrete world of language into a continuous space like GIFs involves a step exactly like that. We will find the space of meaning behind the words, by finding embeddings for every word such that words that are similar in meaning are close to one another.

Our friends over at Google Brain did exactly this, with their software system [Word2Vec](https://code.google.com/archive/p/word2vec/). The realization key to their implementation is that, although words don't have a continuous definition of meaning we can use for the distance optimization, they do approximately obey a simple rule popular in the Natural Language Processing Literature 

##### The Distributional Hypothesis - _succinctly, it states_ :

> "a word is characterized by the company it keeps"

At a high level, this means that rather than optimizing for similar words to be close together, they assume that words that are often in similar contexts have similar meanings, and optimize for that directly instead.

![Some vectors](https://adriancolyer.files.wordpress.com/2016/04/word2vec-gender-relation.png?w=600)

Once this optimization has been completed, the resulting word vectors have exactly the property we wanted them to have - amazing! We'll see that this is exactly the pattern of success here - it doesn't take much, just a good formulation of your objective, and tools sufficient to get you there. The initial Word2Vec results contained some pretty astonishing figures - in particular, they showed that (when embedded in a human-readable, 2D space) not only were similar words near each other, but that the dimensions of variability were consistent with simple geometric operations.

$$ king - man + woman = queen $$

## Back to Spaces

Now that we have a way to convert words from human-readable sequences of letters into computer readable sequences of N-dimensional vectors. We can now consider our sentences as arrays too - with dimensions: dimensionality of the word vectors, and length of sentence.

This leaves us with our sentences looking somewhat like rectangles, with durations and heights, and our GIFs looking like cuboids / rectangular prisms, with durations, heights, and widths.

![sentences and gifs](/images/DeepGIF_figures.002.png "sentences and GIFs")

Now we just need to find the transformations that will put them in a common space, and for that we'll start and end with matrices. You'll discover in your machine learning odyssey that vectors and matrices go together like peas and carrots. A matrix is simply an ordered set of vectors, and so you can imagine we are parametrizing the transformation of our vectors simply by some coefficients. 

$$ A \times B = C $$

where $$\hat{A}$$ is a vector, $$ \mathbf{B} $$ a matrix, and the result $$ \hat{C} $$ another vector

As you can imagine, you'll only learn simple scalings/multiplications this way, but the key insight with neural networks is twofold: 1) by adding what we call an activation function, we can express more complex functions than just a simple multiplication
and 2) once we have these functions which aren't just multiplications, we can make them arbitrarily complex by composing them together. I.e. if we're learning f(x) and g(x) we can combine them by putting one inside the other -> f(g(x))

## Hierarchical Functions

What does this mean for transforming spaces? It basically means that if a correspondence could exist, then we can express it with a neural network of some depth and size - thereby our main challenge is finding the neural network. 

It means that while, for a given pixel value in an image, we could change it slightly and the image will remain reasonable, we cannot change a letter in a word and expect the word to remain reasonable. There isn't even an elementary concept of small perturbation to be made, with pixel values we could add, subtract, multiply or divide by some small amount, but there i
We'll be using a convolutional neural network for the GIFs, which are state of the art for image classification / many other computer vision tasks, and a recurrent neural network, which are state of the art for sentence classification / many other natural language processing tasks, for the sentence descriptions. Each are essentially a set of matrix multiplications (applie)
$$A^2 + B^2 = C^2 $$

To learn such a set of functions from scratch would require a prohibitively large dataset, 


Cool, an English to GIF translator. Sounds straightforward, but no one wants to sit around writing out a sentence for each GIF, and there are multiple possible sentences for each GIF - it would take ages!


Problem setup:
Your friend sends you a photo from his phone of the menu for dinner at the cool new restaurant in town, but it's sideways. Now you have to awkwardly rotate your phone to actually read the menu, and want to demonstrate to him how awkward this is, but how?

Emoji won't really do the situation justice - trust me, I've tried. But maybe a short video clip would... if only you could find the right one.

DeepGIF to the rescue!

