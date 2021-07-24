import streamlit as st
import json
import pandas as pd
from pandas.io.json import json_normalize
import base64
from google_play_scraper import app, Sort, reviews_all
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import spacy
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import string
import gensim
import gensim.corpora as corpora
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# from spacy import load


#####################################################
# TOPIC MODELING
#####################################################



def get_stopwords(language):
    #     nltk.download("stopwords")
    stop_words=[]
#     stop_words = stopwords.words(language)
    if language=="fr":
        stop_words=['a','à','acré','adieu','afin','ah','ai','aie','aïe','aient','aies','ailleurs','ains','ainsi','ait','alentour','alentours','alias','alléluia','allo','allô','alors','amen','an','ans','anti','après','arrière','as','ase','assez','atchoum','au','aube','aucun','aucune','aucunement','aucunes','aucuns','audit','auparavant','auprès','auquel','aura','aurai','auraient','aurais','aurait','auras','aurez','auriez','aurions','aurons','auront','aussi','autant','autour','autours','autre','autrefois','autres','autrui','aux','auxdits','auxquelles','auxquels','avaient','avais','avait','avant','avants','avec','avez','aviez','avions','avoir','avons','ayant','ayez','ayons','b','badabam','badaboum','bah','balpeau','banco','bang','basta','baste','bé','beaucoup','bcp','ben','berk','bernique','beu','beuark','beurk','bien','biens','bigre','bim','bing','bis','bof','bon','bonne','bonnes','bons','boudiou','boudu','bouf','bougre','boum','boums','bravo','broum','brrr','bye','c','ça','ca','calmos','car','caramba','ce','ceci','cela','celle','celles','celui','cependant','certain','certaine','certaines','certains','certes','ces','cet','cette','ceux','chacun','chacune','chaque','chez','chic','chiche','chouette','chut','ci','ciao','cinq','cinquante','clac','clic','combien','comme','comment','concernant','contre','contres','couic','crac','cré','crénom','cristi','croie','croient','croies','croira','croirai','croiraient','croirais','croirait','croiras','croire','croirez','croiriez','croirions','croirons','croiront','crois','croit','croyaient','croyais','croyait','croyant','croyez','croyiez','croyions','croyons','cru','crue','crûmes','crurent','crus','crusse','crussent','crusses','crut','crût','d','da','dans','davantage','dc','de','debout','dedans','dehors','déjà','demain','demains','demi','demie','demies','demis','depuis','derrière','des','dès','desdites','desdits','desquelles','desquels','dessous','dessus','deux','devaient','devais','devait','devant','devants','devers','devez','deviez','devions','devoir','devoirs','devons','devra','devrai','devraient','devrais','devrait','devras','devrez','devriez','devrions','devrons','devront','dia','diantre','différents','dig','ding','dira','dirai','diraient','dirais','dirait','diras','dire','dirent','dires','direz','diriez','dirions','dirons','diront','dis','disaient','disais','disait','disant','dise','disent','dises','disiez','disions','disons','dissent','dit','dît','dite','dites','dîtes','dits','divers','diverses','dix','dm','dois','doit','doive','doivent','doives','dommage','donc','dong','dont','douze','dring','du','dû','dudit','due','dues','dûmes','duquel','durant','durent','dus','dusse','dussent','dusses','dussiez','dussions','dut','dût','e','eh','elle','elles','en','encore','enfin','ensuite','entre','envers','environ','environs','es','ès','est','et','étaient','étais','était','etait','étant','été','êtes','étiez','étions','être','eu','eue','eues','euh','eûmes','eurêka','eurent','eus','eusse','eussé','eussent','eusses','eussiez','eussions','eut','eût','eûtes','eux','excepté','extra','extras','f','faire','fais','faisaient','faisais','faisait','faisant','faisiez','faisions','faisons','fait','faite','faites','fallait','falloir','fallu','fallut','fallût','fasse','fassent','fasses','fassiez','fassions','faudra','faudrait','faut','fera','ferai','feraient','ferais','ferait','feras','ferez','feriez','ferions','ferons','feront','fi','fichtre','fîmes','firent','fis','fisse','fissent','fissiez','fissions','fit','fît','fîtes','flac','floc','flop','font','force','fors','fort','forte','fortes','fortissimo','forts','fouchtra','franco','fûmes','furent','fus','fusse','fussent','fusses','fussiez','fussions','fut','fût','fûtes','g','gare','gares','gnagnagna','grâce','gué','gy','ha','haha','hai','halte','hardi','hare','hé','hein','hélas','hello','hem','hep','heu','hi','hic','hip','hisse','ho','holà','hom','hon','hop','hormis','hors','hou','houhou','houlà','houp','hourra','hourras','hue','hugh','huit','hum','hurrah','icelle','icelles','icelui','ici','il','illico','ils','in','inter','inters','itou','j','jadis','jamais','jarnicoton','je','jouxte','jusqu','jusqu_à','jusqu_au','jusque','juste','justes','l','la','là','lala','laquelle','las','le','lendemain','lendemains','lequel','les','lès','lesquelles','lesquels','leur','leurs','lez','loin','longtemps','lors','lorsqu','lorsque','lui','m','ma','macarel','macarelle','madame','maint','mainte','maintenant','maintes','maints','mais','mal','male','males','malgré','mâtin','maux','mazette','mazettes','me','même','meme','mêmes','merci','merdasse','merde','merdre','mes','mesdames','messieurs','meuh','mézig','mézigue','mi','miam','mien','mienne','miennes','miens','mieux','mil','mille','milles','million','millions','mince','ml','mlle','mm','mme','moi','moindre','moindres','moins','mon','monseigneur','monsieur','morbleu','mordicus','mordieu','motus','mouais','moyennant','n','na','ne','néanmoins','neuf','ni','niet','non','nonante','nonobstant','nos','notre','nôtre','nôtres','nous','nul','nulle','nulles','nuls','o','ô','octante','oh','ohé','ok','olé','ollé','on','ont','onze','or','ou','où','ouah','ouais','ouf','ouh','oui','ouiche','ouille','oust','ouste','outre','outres','pa','palsambleu','pan','par','parbleu','parce','pardi','pardieu','pardon','parfois','parmi','partout','pas','pasque','patapouf','patata','patati','patatras','pchitt','pendant','pendante','pendantes','pendants','personne','personnes','peste','peu','peuchère','peuh','peut','peuvent','peux','pff','pfft','pfutt','pianissimo','pianissimos','pis','plein','ploc','plouf','plupart','plus','plusieurs','point','points','pollope','polope','pouah','pouce','pouf','pouh','pouic','pour','pourquoi','pourra','pourrai','pourraient','pourrais','pourrait','pourras','pourrez','pourriez','pourrions','pourrons','pourront','pourtant','pouvaient','pouvais','pouvait','pouvant','pouvez','pouviez','pouvions','pouvoir','pouvoirs','pouvons','près','presque','primo','pristi','prosit','prout','pschitt','psitt','pst','pu','puis','puisqu','puisque','puissamment','puisse','puissent','puisses','puissiez','puissions','pûmes','purent','pusse','pussent','pusses','pussiez','put','pût','pûtes','qu','quand','quant','quarante','quasi','quatorze','quatre','que','quel','quelconque','quelconques','quelle','quelles','quelque','quelquefois','quelques','quels','qui','quiconque','quinze','quoi','quoique','rantanplan','rasibus','rataplan','rebelote','recta','revoici','revoilà','rez','rien','riens','s','sa','sachant','sache','sachent','saches','sachez','sachiez','sachions','sachons','sacrebleu','sacrédié','sacredieu','sais','sait','salut','sans','saperlipopette','sapristi','sauf','saufs','saura','saurai','sauraient','saurais','saurait','sauras','sauront','savaient','savais','savait','savent','savez','saviez','savions','savoir','savoirs','savons','scrogneugneu','se','sécolle','secundo','seize','selon','sept','septante','sera','serai','seraient','serais','serait','seras','serez','seriez','serions','serons','seront','ses','sézigue','si','sic','sien','sienne','siennes','siens','sinon','six','skaal','snif','sniff','soi','soient','sois','soit','soixante','sommes','son','sons','sont','soudain','soudaine','soudaines','soudains','sous','souvent','soyez','soyons','splash','su','subito','suis','suivant','sûmes','sur','sure','surent','sures','surnombre','surs','surtout','surtouts','sus','susse','sussent','sut','sût','t','ta','tacatac','tacatacatac','tagada','taïaut','tant','tap','taratata','tard','tchao','te','té','tel','telle','telles','tels','tertio','tes','tézig','tézigue','tien','tienne','tiennes','tiens','tintin','to','toi','ton','tons','toujours','tous','tout','toute','toutefois','toutes','touts','treize','trente','très','trois','trop','tu','tudieu','turlututu','u','un','une','unes','uns','v','van','vans','ventrebleu','vers','versus','vertubleu','veuille','veuillent','veuilles','veuillez','veulent','veut','veux','via','vingt','vite','vivat','vive','vlan','vlouf','voici','voilà','voire','volontiers','vos','votre','vôtre','vôtres','voudra','voudrai','voudraient','voudrais','voudrait','voudras','voudrez','voudriez','voudrions','voudrons','voudront','voulaient','voulais','voulait','voulant','voulez','vouliez','voulions','vouloir','vouloirs','voulons','voulu','voulue','voulûmes','voulurent','voulus','voulusse','voulussent','voulut','voulût','vous','vroom','vroum','wouah','x','y','yeah','youp','youpi','yu','zou','zut','zzz','zzzz','cest','etre','ouii','ouiiii','hahah','hahaha','hahahah','hahahaha','hahahahaha','hahahahahaha','hahahahaahaha','hahaaa','hahaaahaaaa','hahahaahhaaaaa','ahahaahahaha','ahahah','ahahahah','ahahahahah','ahahahahahahah','ahaha','ahah','aha','http','https','www','p','r','ouai','étée','étées','étés','étante','étants','étantes','ayante','ayantes','ayants']
    if language=="en":
        stop_words=["able","about","above","according","accordingly","across","actually","after","afterwards","again","against","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","came","can","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","course","currently","d","definitely","described","despite","did","different","do","does","doing","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","happens","hardly","has","have","having","he","hello","help","hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","it","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","like","liked","likely","little","look","looking","looks","ltd","m","made","mainly","make","makes","making","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","take","taken","takes","taking","tell","tends","th","than","that","thats","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","theres","thereupon","these","they","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","way","we","welcome","well","went","were","what","whatever","when","whence","whenever","where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","wonder","would","x","y","yes","yet","you","your","yours","yourself","yourselves","z","zero","a","you're","you've","you'll","you'd","she's","it's","that'll","don","don't","should've","ll","ve","ain","aren","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't","haven","haven't","isn","isn't","ma","mightn","mightn't","mustn","mustn't","needn","needn't","shan","shan't","shouldn","shouldn't","wasn","wasn't","weren","weren't","won","won't","wouldn","wouldn't"]
    return stop_words



@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer()

@Language.factory('pos')
def create_pos_tagger(nlp, name):
    return POSTagger()

def tokenize_text(df,col_name,lang):
    lemma = []
    if lang=='fr':
        nlp = spacy.load('fr_core_news_sm')
        nlp.add_pipe('pos', name='pos', after='parser')
        nlp.add_pipe('french_lemmatizer', name='lefff', after='pos')
       
    if lang=='en':
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('pos', name='pos', after='parser')
    i=0    
    
    for doc in nlp.pipe(df[col_name].astype('unicode').values, batch_size=50,n_process=5):
        i=i+1
        if doc.is_parsed:          
            lemma.append([n.lemma_.lower().translate(str.maketrans('', '', string.punctuation+'’')) for n in doc if n.pos_ in ["VERB","NOUN","ADJ","PROPN","ADV","SYM"]])
        else:
            lemma.append(None)

#     df['lemma']=[' '.join(map(str, l)) for l in lemma]
    df['lemma']=lemma
    return df


def pipeline_nlp(df_sample,lang,stop_words,no_topics):
    df_sample=tokenize_text(df_sample,'content',lang)
    
    tokenized_data_bigrams,tokenized_data_trigrams=create_bigrams_trigrams(df_sample['lemma'])
    df_sample=detokenization(df_sample,tokenized_data_trigrams)
    vectorizer,document_matrix,feature_names=vectorize(df_sample['detokenized_text'],5000,stop_words)    

    nmf_model = NMF(n_components=no_topics, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd',max_iter=1000).fit(document_matrix)
    
    df_sample,nmf_topic_values=get_topics(df_sample,nmf_model,document_matrix)
    df_topics=display_topics(nmf_model, feature_names, 0.4, 3)
    
    #preparation données du wordcloud
    dense = document_matrix.todense()
    lst1 = dense.tolist()
    df_tfidf = pd.DataFrame(lst1, columns=feature_names)
    
    
    #génération du wordcloud
    Cloud = WordCloud(background_color="white", max_words=50,width=800, height=500).generate_from_frequencies(df_tfidf.T.sum(axis=1))
    
    return df_sample,df_topics,df_tfidf, document_matrix,feature_names,Cloud


# Define functions for creating bigrams and trigrams.

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def create_bigrams_trigrams(docs):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(docs, min_count=5, threshold=10) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    tokenized_data_bigrams = make_bigrams(docs,bigram_mod)
    
    trigram = gensim.models.Phrases(bigram[docs], threshold=10)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    tokenized_data_trigrams = make_trigrams(docs,bigram_mod,trigram_mod)
    
    return tokenized_data_bigrams,tokenized_data_trigrams


def detokenization(df,tokenized_data_trigrams):
    # de-tokenization, combine tokens together
    detokenized_data = []
    for i in range(len(df)):
        t = ' '.join(tokenized_data_trigrams[i])
        detokenized_data.append(t)
    df['detokenized_text']= detokenized_data
    documents = df['detokenized_text']
    return df


def vectorize(documents,no_terms,stop_words):
    # NMF uses the tf-idf count vectorizer
    # Initialise the count vectorizer with the English stop words
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=no_terms, stop_words=stop_words)
    # Fit and transform the text
    document_matrix = vectorizer.fit_transform(documents)
    #get features
    feature_names = vectorizer.get_feature_names()
    return vectorizer,document_matrix,feature_names


def get_topics(df,nmf_model,document_matrix):
    #Use NMF model to assign topic to papers in corpus
    nmf_topic_values = nmf_model.transform(document_matrix)
    df['NMF Topic'] = nmf_topic_values.argmax(axis=1)
    df['NMF Proba']=nmf_topic_values.max(axis=1)
    return df,nmf_topic_values


def display_topics(model, feature_names, seuil, no_top_words):
 
    l_topics=[]
    for topic_idx, topic in enumerate(model.components_):
        topic_str=''
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            if topic[i]>seuil:
                topic_str=topic_str+feature_names[i]+', '

        l_topics.append(topic_str[:-2])
    
    df_topics=pd.DataFrame(l_topics,columns=["topic_title"])
    df_topics['index']=df_topics.index
    
    return df_topics



######################################
# DATA VIZ
######################################


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])



def process_histogram(result_app):
    df_histogram=pd.DataFrame(result_app['histogram'],columns=['reviews'])
    df_histogram['Note']= df_histogram.index +1 
    df_histogram["%reviews"]=(df_histogram['reviews']/df_histogram['reviews'].sum()) * 100
    df_histogram["%reviews"]=df_histogram["%reviews"].round(1).astype(str)+'%'
    
    return df_histogram


def histogram_score(df,x,y,text,marker_color):
    x_data=list(df[x])
    y_data=list(df[y])
    text_data=list(df[text].astype(str))
    
    fig = go.Figure(go.Bar(
                x=x_data,
                y=y_data,
                orientation='h', marker_color=marker_color,text=text_data),

                layout=go.Layout(title=go.layout.Title(text="Répartition des avis"))
                   )  

    fig.update_traces(textposition='inside')
    
    return fig



def barchart_sentiment_relative(df_reviews):
#     df_gb=df_reviews.groupby(["month"]).agg({"reviewId":"nunique","score":"mean"}).reset_index().sort_values(by="month",ascending=False)
    
    df_gb=df_reviews.groupby(["month"]).agg({"reviewId":"nunique","score":"mean"}).reset_index()
    df_sentiment=df_reviews.groupby(["month","sentiment"]).agg({"reviewId":"nunique"})
    df_sentiment['%reviews']=df_sentiment.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
    df_sentiment=df_sentiment.reset_index()
#     df_sentiment=df_sentiment.reset_index().sort_values(by="month",ascending=False)
    df_sentiment=df_sentiment.pivot(index="month", columns="sentiment", values="%reviews").reset_index()
    df_sentiment["reviews_count"]=df_gb['reviewId']
    df_sentiment["score"]=df_gb['score']   

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    
    
    if "positif" in df_sentiment.columns:
        fig.add_trace(go.Bar(
            y=df_sentiment["positif"],
            x=df_sentiment["month"],
            name="positif",
            marker=dict(
                color='#57bb8a',
                line=dict(color='rgba(0,128,0, 0.5)', width=0.05)
            )
        ))
    if "neutre" in df_sentiment.columns:
        fig.add_trace(go.Bar(
            y=df_sentiment["neutre"],
            x=df_sentiment["month"],
            name="neutre",
            marker=dict(
                color='#ffcf02',
                line=dict(color='rgba(0,0,255, 0.5)', width=0.05)
            )
        ))
        
    if "négatif" in df_sentiment.columns:
        fig.add_trace(go.Bar(
            y=df_sentiment["négatif"],
            x=df_sentiment["month"],
            name="négatifs",
            marker=dict(
                color='#ff6f31',
                line=dict(color='rgba(128,0,0, 0.5)', width=0.05)
            )
        ))
    fig.add_trace(go.Scatter(
            x=df_sentiment["month"], 
            y=df_sentiment["score"],
            mode='lines+markers',
            name='#score moyen',
            line=dict(color='LightSlateGray', width=2)
                ),
          secondary_y=True)
    
    fig.update_layout(
            yaxis=dict(
            title_text="Reviews(%)",
            ticktext=["0%", "20%", "40%", "60%","80%","100%"],
            tickvals=[0, 20, 40, 60, 80, 100],
            tickmode="array",
            titlefont=dict(size=15),
        ),
        autosize=False,
        width=1000,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': "Evolution du rating",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        barmode='relative') 
    
    fig.update_yaxes(title_text="Score moyen", secondary_y=True,range=[0, 5])

    return fig,df_sentiment


def barchart_dev_replies(df):
    

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        y=df["Réponses"],
        x=df["month"],
        name="Réponses du développeur",
        marker=dict(
            color='#F08080',
            line=dict(color='rgba(0,128,0, 0.5)', width=0.05)
        )
    ))

    fig.add_trace(go.Scatter(
            x=df["month"],
            y=df["Taux de réponse"],
            mode='lines+markers',
            name='Taux de réponse',
            line=dict(color='LightSlateGray', width=2)
                ),
          secondary_y=True)
    
    fig.update_layout(
            yaxis=dict(
            title_text="Reviews",
#             ticktext=["0%", "20%", "40%", "60%","80%","100%"],
#             tickvals=[0, 20, 40, 60, 80, 100],
            tickmode="array",
            titlefont=dict(size=15),
        ),
        autosize=False,
        width=1000,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': "Evolution des réponses du développeur",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        barmode='stack') 
    
    fig.update_yaxes(title_text="Taux de réponse", secondary_y=True,range=[0, 1],            ticktext=["0%", "20%", "40%", "60%","80%","100%"],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])

    return fig






def query_app(app_id,lang,country):
    return app(app_id,lang=lang, country=country)


def query_reviews(app_id,lang,country):
    return reviews_all(app_id,sleep_milliseconds=0, lang=lang, country=country, sort=Sort.MOST_RELEVANT, filter_score_with=None)


def sample_reviews(df_reviews,dt_min_date):
    if len(df_reviews)>1000:
        df_reviews_subset=df_reviews[df_reviews['at']>dt_min_date]
        if len(df_reviews_subset)>1000:
            df_sample=df_reviews_subset.sample(n=1000, random_state=42)
        else:
            df_sample=df_reviews_subset
    else:
        df_sample=df_reviews
        
    return df_sample


def parsing_reviews(result_reviews,app_id,country):
    
    #parsing des reviews
    df_reviews=json_normalize(result_reviews)

    df_reviews['sentiment']= np.where(df_reviews['score']>3,"positif",np.where(df_reviews['score']<3,"négatif","neutre"))
    df_reviews['url']="https://play.google.com/store/apps/details?id="+str(app_id)+"&gl="+country+"&reviewId="+df_reviews['reviewId']   
    df_reviews['Réponses']=np.where(df_reviews["replyContent"].str.len()>0,1,0)
    
    
    
    return df_reviews


def define_no_topics(df):
    no_topics=20
    size=len(df)
    
    if size<750:
        no_topics=15
        if size<500:
            no_topics=10
            if size<250:
                no_topics=8       
    
    return no_topics
        


def main():
    st.set_page_config(
        page_title="Google Play Analysis",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    ###################################
    # PARAMETRES DE LA SIDEBAR
    st.sidebar.title('Paramètres')          
    app_id=st.sidebar.text_input("Entrez l'ID de l'application à analyser", value='com.nytimes.android', max_chars=None, key=None, type='default') 
    lang=st.sidebar.selectbox("Sélectionnez la langue des reviews à capturer",['fr','en'], index=0) 
    country=st.sidebar.selectbox("Sélectionnez la zone géographique du Google Play Store",['fr','gb','us'], index=0) 
#     no_topics = st.sidebar.number_input("Combien de topics à détecter", min_value=0, max_value=None,value=20, step=1)
    
    if st.sidebar.button("Valider"):
#         if lang=='en':
#             import en_core_web_sm
#         else:
#             import fr_core_web_sm

        
        try:
            result_app=query_app(app_id,lang,country)
            df_app=json_normalize(result_app)

            if result_app["free"] is True:
                free="App Gratuite"
            else:
                free="App payante : "+str(result_app["price"])+result_app["currency"]
            if result_app['containsAds'] is True:
                containsAds="Contient des publicités"
            else:
                containsAds="Ne contient pas de publicités"
            if result_app["inAppProductPrice"] is None:
                inApp="Pas d'achat in App"
            else:
                inApp="Achats in App : "+str(result_app["inAppProductPrice"])
            if result_app['ratings'] is not None:
                rating=human_format(result_app['ratings'])
            else:
                rating=0

            st.write("<div style=\"background-color: #e1e1e1;float:left;padding:10px 10px 10px 10px;width:100%;border-radius:5px;\"><div><div style=\"float: left;width:20%;\"><a href=\""+result_app['url']+"\" target=\"_blank\"><img src=\""+result_app['icon']+"\" style=\"border-radius: 50%;\" width=\"150\"/></img></a></div><div style=\"float: left;width:80%;\"><h1>"+result_app['title']+"</h1><br/>"+result_app['summaryHTML']+"<br/><hr><button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">"+result_app['genre']+"</button>&nbsp;<button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">Développé par "+result_app['developer']+"</button>&nbsp;<button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">"+result_app['contentRating']+"</button></div></div></div>",unsafe_allow_html=True)

            col1, col2 = st.beta_columns(2)
            with col1:

                st.write("<h3>Key metrics & modèle économique</h3>", unsafe_allow_html=True)
                st.write("<table style=\"border-collapse: collapse;margin: 25px 0;font-size: 0.9em;font-family: sans-serif;min-width: 400px;width:100%;box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);\"><thead style=\"background-color: #F63366;color: #ffffff;text-align: left;\"><tr><th>KPI</th><th>Depuis la création de l'app</th></tr></thead><tbody><tr><td><b>#reviews</b></td><td>"+str(human_format(result_app['reviews']))+" reviews</td></tr><tr><td><b>#ratings</b></td><td>"+str(rating)+" ratings</td></tr><tr><td><b>Installations</b></td><td>"+str(result_app['installs'])+"</td></tr><tr><td><b>Score moyen</b></td><td>"+str(round(result_app['score'],2))+"</td></tr><tr><td><b>Modèle économique</b></td><td>"+free+"</td><tr><td><b>Intégration e-commerce</b></td><td>"+inApp+"</td></tr><tr><td><b>Inclus de la publicité</b></td><td>"+containsAds+"</td></tbody></table>", unsafe_allow_html=True)
            with col2:
                try:
                    df_histogram=process_histogram(result_app)
                    fig=histogram_score(df_histogram,'reviews','Note','%reviews',['#ff6f31','#ff9f02','#ffcf02','#9ace6a','#57bb8a'])
                    st.plotly_chart(fig, use_container_width=False, sharing='streamlit') 
                except:
                    pass
                    st.info("Il n'y a pas de reviews")
       
        except:
            pass
            st.error("Impossible de collecter les infos sur cette application")
            
            
        try:
            with st.spinner("Collecte des reviews en cours"):
                # Récupération des reviews
                result_reviews = query_reviews(app_id,lang,country)

            if len(result_reviews)>0:

                try:
                    # Mise en forme des données
                    df_reviews=parsing_reviews(result_reviews,app_id,country)

                    #calcul de la date correspondant au dernier trimestre et au dernier semestre
                    dt_min_date=(df_reviews["at"].max()-timedelta(days=90))
                    dt_min_date_12months=(df_reviews["at"].max()-timedelta(days=365))


                    df_reviews.sort_values(by='at', ascending=True,inplace=True)
                    df_reviews["month"]=pd.to_datetime(df_reviews['at']).dt.strftime('%Y-%m')

                    #agrégation des données (réponses du développeur)
                    df_dev=df_reviews[['month','Réponses']].groupby("month").agg({'Réponses':'sum'}).reset_index()
                    df_dev['Total reviews']=df_reviews.groupby("month").agg({'reviewId':'nunique'}).reset_index()['reviewId']
                    df_dev['Taux de réponse']=df_dev['Réponses']/df_dev['Total reviews']    


                    st.subheader("Commentaires sur Google Play Store")
                    fig,df_sentiment=barchart_sentiment_relative(df_reviews) 

                    st.write(str(len(df_reviews))+ " commentaires ont été posté depuis le "+str(df_reviews["at"].min().strftime('%d-%m-%Y'))+" ("+str(len(df_reviews[df_reviews["at"]>dt_min_date_12months]))+" sur les 12 derniers mois) . La notation moyenne la plus basse était de "+str(round(df_sentiment['score'].min(),2))+" le "+ str(df_sentiment[df_sentiment['score'] == min(df_sentiment['score'])]['month'].values[0])+" . La notation la plus élevée était de "+str(round(df_sentiment['score'].max(),2))+ " le "+ str(df_sentiment[df_sentiment['score'] == max(df_sentiment['score'])]['month'].values[0]), unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
                except:
                    pass
                    st.info("Impossible d'analyser les reviews")




                # Analyse des réponses du developpeur    

                st.subheader("Réponses du développeur")
                if len(df_reviews[df_reviews["replyContent"].str.len()>0])>0:
                    df_replies=df_reviews[df_reviews["replyContent"].str.len()>0]
                    nb_replies=len(df_replies)
                    nb_replies_12months=len(df_replies[df_replies["at"]>dt_min_date_12months])

                    per_replies=(nb_replies/len(df_reviews))*100

                    st.write(str(nb_replies)+ " commentaires ont été posté depuis le "+str(df_replies["at"].min().strftime('%d-%m-%Y'))+" ("+str(nb_replies_12months)+" sur les 12 derniers mois) . Le taux de réponse moyen s'élève à "+str(round(per_replies,1))+"%.", unsafe_allow_html=True)
                    fig=barchart_dev_replies(df_dev)
                    st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
                else:
                    st.info("Le développeur n'a répondu a aucun commentaires")

                # on splitte nos reviews selon le sentiment. On retient un max de 1000 reviews sur les 3 derniers mois   

                df_negative_reviews=sample_reviews(df_reviews[df_reviews['score']<4],dt_min_date)
                df_positive_reviews=sample_reviews(df_reviews[df_reviews['score']>3],dt_min_date)
                stop_words=get_stopwords(lang)



                if len(df_negative_reviews)>100:  
                    with st.spinner("Analyse de "+str(len(df_negative_reviews))+" reviews récentes en cours - un peu de patience ! :)"):
                        no_topics=define_no_topics(df_negative_reviews)
                        df_negative_reviews, df_neg_topics, df_neg_tfidf, document_matrix_neg, feature_names_neg, neg_cloud = pipeline_nlp(df_negative_reviews,lang,stop_words,no_topics)

                        neg_is_ok=True

                else:
                    neg_is_ok=False

                if len(df_positive_reviews)>100:
                    with st.spinner("Analyse de "+str(len(df_positive_reviews))+" reviews récentes en cours - un peu de patience ! :)"):
                        no_topics=define_no_topics(df_positive_reviews)
                        df_positive_reviews, df_pos_topics, df_pos_tfidf, document_matrix_pos, feature_names_pos, pos_cloud = pipeline_nlp(df_positive_reviews,lang,stop_words,no_topics)
                        pos_is_ok=True
                else:
                    pos_is_ok=False



                if pos_is_ok is True or neg_is_ok is True:
                    st.subheader("Termes spécifiques")
                    st.write("<p>Les 50 termes les plus spécifiques aux reviews récentes</p><br/>",unsafe_allow_html=True)             
                    col1, col2 = st.beta_columns(2)
                    with col1:

                        st.write("<h4>Reviews négatives</h4><br/><br/>",unsafe_allow_html=True)  
                        if neg_is_ok is True :
                            plt.imshow(neg_cloud, interpolation='bilinear')
                            plt.axis("off")
                            st.image(neg_cloud.to_array(),use_column_width='auto')

                        else:
                            st.info("Il n'y a pas suffisamment de reviews négatives à analyser")

                    with col2:
                        st.write("<h4>Reviews positives</h4><br/><br/>",unsafe_allow_html=True)
                        if pos_is_ok is True :
                            plt.imshow(pos_cloud, interpolation='bilinear')
                            plt.axis("off")
                            st.image(pos_cloud.to_array(),use_column_width='auto')
                        else:
                            st.info("Il n'y a pas suffisamment de reviews positives à analyser")


                    st.subheader("Sujets principaux")
                    st.write("Les reviews récentes sont classées en 10 sujets principaux.")
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        st.write("<h4>Reviews négatives</h4><br/><br/>",unsafe_allow_html=True)
                        if neg_is_ok is True :
                            df_negative_reviews=pd.merge(df_negative_reviews,df_neg_topics, how='left', left_on='NMF Topic', right_on='index')

                            df_pie_neg = df_negative_reviews[["topic_title","reviewId"]].groupby(["topic_title"]).agg({"reviewId":"nunique"}).reset_index().sort_values(by='reviewId',ascending=False)

                            fig = px.pie(df_pie_neg, values='reviewId', names='topic_title', title='Principaux pain points')
                            st.plotly_chart(fig, use_container_width=True, sharing='streamlit')

                        else:
                            st.info("Il n'y a pas suffisamment de reviews négatives à analyser")

                    with col2:
                        st.write("<h4>Reviews positives</h4><br/><br/>",unsafe_allow_html=True)

                        if pos_is_ok is True :
                            df_positive_reviews=pd.merge(df_positive_reviews,df_pos_topics, how='left', left_on='NMF Topic', right_on='index')

                            df_pie_pos = df_positive_reviews[["topic_title","reviewId"]].groupby(["topic_title"]).agg({"reviewId":"nunique"}).reset_index().sort_values(by='reviewId',ascending=False)

                            fig = px.pie(df_pie_pos, values='reviewId', names='topic_title', title='Principaux points d\'appréciation')
                            st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
                        else:
                            st.info("Il n'y a pas suffisamment de reviews positives à analyser")


                    if pos_is_ok is True :
                        st.subheader("Verbatims positifs")
                        st.write("Consultez les reviews les plus pertinentes par sujet") 
                        # POUR CHAQUE TOPIC, ON AFFICHE LES ARTICLES CLASSES            
                        for n in sorted(df_positive_reviews['NMF Topic'].unique()):
                            d=df_positive_reviews[df_positive_reviews['NMF Topic']==n].sort_values(by='NMF Proba',ascending=False)

                            with st.beta_expander("Sujet n°"+str(n+1)+" - "+d['topic_title'].min()+" - "+str(round(len(d)/len(df_positive_reviews)*100,1))+"% des reviews récentes - score moyen : "+str(round(d['score'].mean(),1))):
                                 st.table(d[['content','sentiment']][:15].assign(hack='').set_index('hack'))


                    if neg_is_ok is True :
                        st.subheader("Verbatims négatifs")
                        st.write("Consultez les reviews les plus pertinentes par sujet") 
                        # POUR CHAQUE TOPIC, ON AFFICHE LES ARTICLES CLASSES            
                        for n in sorted(df_negative_reviews['NMF Topic'].unique()):
                            d=df_negative_reviews[df_negative_reviews['NMF Topic']==n].sort_values(by='NMF Proba',ascending=False)

                            with st.beta_expander("Sujet n°"+str(n+1)+" - "+d['topic_title'].min()+" - "+str(round(len(d)/len(df_negative_reviews)*100,1))+"% des reviews récentes - score moyen : "+str(round(d['score'].mean(),1))):
                                 st.table(d[['content','sentiment']][:15].assign(hack='').set_index('hack'))    




                else:
                    st.info("Il n'y a pas suffisamment de reviews à analyser")

            else:
                    st.info("Il n'y a pas de reviews à analyser")


    
        except:
            pass
            st.error("Impossible de récupérer les reviews pour cette application")
    
if __name__ == "__main__":
    main()    