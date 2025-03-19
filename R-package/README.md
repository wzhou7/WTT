# Collocated Topic Modeling R Implementation

This repositoty implements the collocated topic modeling method in R. This method is described in Jung, Zhou, & Smith (2024) with an illustrative study. There is also a Python implementation available at the [WTTP](https://github.com/wzhou7/WTTP) repository.

# How to Use

To use this package, you will need the `devtools` library to install it from github. 
For example, 

```
install.packages("devtools") # install devtools
library(devtools) # load devtools
install_github("wzhou7/WTTR") # install WTTR
```

The above code needs to be run just once.
Then, you can load the `WTTR` package each time of use:

```
library(WTTR) # load WTTR
```

Then, you can use the implemented functions to explore your data. See example study.

# Citation

If you use our code in your study, please cite:

> Jung, J., Zhou, W., and Smith, A. (2024). **From Textual Data to Theoretical Insights: Introducing and Applying the Word-Text-Topic Extraction Approach.** *Organizational Research Methods (ORM)*.

or via BibTeX code:

```
@ARTICLE{jung2024textual,
  title =        {From Textual Data to Theoretical Insights: Introducing and
                  Applying the Word-Text-Topic Extraction Approach},
  author =       {Jung, Jaewoo and Zhou, Wenjun and Smith, Anne},
  journal =      {Organizational Research Methods (ORM)},
  pages =        {forthcoming},
  year =         {2024}
}
```


# Using Google Pretrained Word Vectors in R

While loading Google's pretrained word vectors is quite easy in Python, doing so in R isn't quite as easy. You will need to download the binary file from Google, then then load the vectors locally for your data analysis needs. 

To obtain Google Pretrained Word Vectors:

1. Go to [Google's word2vec archive page](https://code.google.com/archive/p/word2vec/), search for "`GoogleNews-vectors-negative300.bin.gz`" to find its download [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Download it and and unzip locally into your working directory. You will have a file `GoogleNews-vectors-negative300.bin`. (Note the .gz file is more than 1GB, and when extracted, the .bin file will be more than 3GB.)

To load the vectors, here's my quick fix based on Windows.

1. Download [distance.c](https://github.com/mukul13/rword2vec/blob/master/src/distance.c) from the `rword2vec` repository and save it in your working directory.
2. Compile the C program file into a DLL dynamic library. This step is platform dependent. After ensuring you have R and Rtools installed, in Windows command line, navigate into the working directory, and then run “R CMD SHLIB distance.c”. This step will generate several files in the same directory. One of them is `"distance.dll"`.
3. In R scripts, you can use the `".C"` function after calling `dyn.load("distance.dll")`. E.g.,

```
dyn.load("distance.dll")

file_name = "GoogleNews-vectors-negative300.bin"
search_word = "good"
num=5
OUT <- .C("distance",
   rfile_name=as.character(file_name),
   search_word=as.character(search_word),
   rN=as.integer(num),
   rbestw=as.character(rep("",num)),
   rbestd=as.double(rep(0,num)))

data=as.data.frame(cbind(OUT$rbestw,OUT$rbestd))
colnames(data)=c("word","dist")
data
```
Note: the [rword2vec](https://github.com/mukul13/rword2vec) package would have been a good choice to load these vectors. However, that repository appears to be unmaintained at the moment I was checking. 
