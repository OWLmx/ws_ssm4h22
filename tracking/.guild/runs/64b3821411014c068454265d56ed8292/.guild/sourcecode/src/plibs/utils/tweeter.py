import re
import emoji
import html

# def clean_tweet(tweet, demojize=True, normalize_usertags='@USER', normalize_urls='HTTPURL', normalize_hashtags=''):
def clean_tweet(tweet, demojize=True, normalize_usertags='@USER', normalize_urls='URL', normalize_hashtags='_content_'):
    clean_tweet = str(tweet).replace('\n','')
    clean_tweet = clean_tweet.replace('&amp;',' and ')
    
    clean_tweet = html.unescape(clean_tweet)
    if not (normalize_usertags is None): 
        clean_tweet = re.sub("@[A-Za-z0-9_]+",f" {normalize_usertags} ", clean_tweet)
    if not (normalize_urls is None): 
        clean_tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',f" {normalize_urls} ", clean_tweet)        
    if not (normalize_hashtags is None):
        if normalize_hashtags =='_content_':
            clean_tweet = re.sub("#([A-Za-z0-9_]+)",r"\1", clean_tweet)
        else:
            clean_tweet = re.sub("#[A-Za-z0-9_]+",f" {normalize_hashtags} ", clean_tweet)
    if demojize:
        clean_tweet = emoji.demojize(clean_tweet)
    
    
    clean_tweet = re.sub("\s\s+"," ", clean_tweet)
    
    return clean_tweet