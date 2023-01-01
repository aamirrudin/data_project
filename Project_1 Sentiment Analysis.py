from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re

def abb_en(line):
    abbreviation_en = {
        'u': 'you',
        'thr': 'there',
        'asap': 'as soon as possible',
        'lv' : 'love',
        'c' : 'see'
    }

    abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
    return (abbrev)

def remove_features(data_str):
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')
    mention_re = re.compile(r'@|#(\w+)')
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)
    data_str = data_str.lower()
    data_str = url_re.sub(' ', data_str)
    data_str = mention_re.sub(' ', data_str)
    data_str = num_re.sub(' ', data_str)
    return data_str

def sentiment_text(value):
    if value>0:
        return "Positive"
    elif value<0:
        return "Negative"
    else:
        return "Neutral"

#creating ETL functions
def main(sc,filename):
    sentiment_analysis1 = sc.textFile(filename).map(lambda x:x.split(",")).filter(lambda x: len(x) == 9)\
    .filter(lambda x: len(x[0])>1)

    mydata1 = sentiment_analysis1.map(lambda x: x[1]).map(lambda x:remove_features(x)).map(lambda x: abb_en(x))\
    .map(lambda x: TextBlob(x).sentiment.polarity).map(lambda x:sentiment_text(x))

    project_1 = sentiment_analysis1.zip(mydata1).map(lambda x:str(x).replace("'","")).map(lambda x: str(x).replace('"',""))

    print(project_1.take(5))

    project_1.saveAsTextFile("Project_1")

if __name__ == "__main__":
    conf = SparkConf().setMaster("local[1]").setAppName("project1")
    sc = SparkContext(conf = conf)

filename = "blockchain.csv"
main(sc,filename)

sc.stop()