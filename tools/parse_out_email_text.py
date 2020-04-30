#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

 
# other stemmer exist

    
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string   "to be uncomment if needed

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        words = ' '.join([stemmer.stem(word) for word in text_string.split()])
#        split_list = text_string.split();
#        words = ''
#        for e in split_list:
#            words += stemmer.stem(e) + " "
        
#        text_string = text_string.replace("\n", "").replace("\r", "")
#        text_string = text_string.replace("  "," ")
#        texts = text_string.split(" ",)
#        stemmer = SnowballStemmer("english")
#        i = 0
#        
#        while i < len(texts):
#            
#            if len(stemmer.stem(texts[i])) > 0:
#                words += stemmer.stem(texts[i]).strip()
#                
#            i+=1
#            if i < len(texts):
#                words += " "
    return words
    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()
