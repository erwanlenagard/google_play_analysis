####################################
#                                  #
# FONCTIONS NECESSAIRES AU PROJET  #
#                                  #
####################################
import streamlit as st
import json
import pandas as pd
from twitchAPI.twitch import Twitch
from twitchAPI.types import TimePeriod,SortMethod,VideoType
from datetime import datetime
import plotly.express as px
import re
import base64

@st.cache()
def create_instance(key,secret):
    twitch = Twitch(key, secret)
    twitch.authenticate_app([])
    return twitch

@st.cache()
def collect_video(twitch,user,periode,classement,video_type):

    #On requête
    video_data=dict()
    video_data["data"]=[]
    cursor=None
    videos_json=twitch.get_videos(user_id=user,first=100,period=periode,sort=classement,video_type=video_type)

    if len(videos_json['data'])>0:
        video_data["data"]=videos_json['data']
    if 'pagination' in videos_json:
        if 'cursor' in videos_json['pagination']:
            cursor=videos_json['pagination']['cursor']
        else:
            cursor=None

    while cursor is not None:
        videos_json=twitch.get_videos(user_id=user, after=cursor,first=100,period=periode,sort=classement,video_type=video_type)
        if len(videos_json['data'])>0:
            video_data["data"]=video_data["data"]+videos_json['data']

        if 'pagination' in videos_json:
            if 'cursor' in videos_json['pagination']:
                cursor=videos_json['pagination']['cursor']
            else:
                cursor=None
    return video_data


@st.cache()
def parsing_user(users):
    all_users=[]
    for user in users["data"]:
        user_id=user['id']
        login=user['login']
        type_user=user['type']
        description=user['description']
        broadcaster_type=user['broadcaster_type']
        view_count=user['view_count']
        created_at=user['created_at']
        current_user=(user_id,login,type_user,description,broadcaster_type,view_count,created_at)
        all_users.append(current_user)

    df_users=pd.DataFrame.from_records(all_users,columns=['user_id','login','user_type','description','broadcaster_type','view_count','created_at'])

    return df_users

@st.cache()
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

@st.cache()
def parsing_videos(video_json):
    df_videos=pd.DataFrame()
    all_vids=[]

    if 'data' in video_json:
        videos=video_json['data']
        for video in videos:
            video_id=video["id"]
            user_id=video['user_id']
            user_login=video['user_login']
            user_name=video['user_name']
            title=video["title"]
            description=video['description']
            created_at=video['created_at']
            published_at=video['published_at']
            url=video['url']
            if video['thumbnail_url'] is None:
                thumbnail_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Circle-icons-image.svg/240px-Circle-icons-image.svg.png"
            else:
                thumbnail_url=video['thumbnail_url'].replace("%{width}","200").replace("%{height}","200")
            viewable=video['viewable']
            view_count=video['view_count']
            language=video['language']
            vid_type=str(video['type'])
            duration=video['duration']
            
            current_vid=(video_id,user_id,user_login,user_name,title,description,created_at,published_at,url,thumbnail_url,
                         viewable,view_count,language,vid_type,duration)
            all_vids.append(current_vid)
        df_videos=pd.DataFrame.from_records(all_vids,columns=['video_id','user_id','login','user_name','title','description',
                                                              'created_at','published_at','url','thumbnail_url','viewable',
                                                              'view_count','language','video_type','duration'])
        
        df_videos["month"]=pd.to_datetime(df_videos['created_at']).dt.strftime('%Y-%m')
        df_videos[["hours","min","sec"]]=df_videos["duration"].str.extract(r"((\d+)h)?((\d+)m)?((\d+)s)",expand=True)[[1,3,5]].fillna(0)
        df_videos["duration_s"]=df_videos['hours'].astype(int)*3600+df_videos['min'].astype(int)*60+df_videos["sec"].astype(int)
        df_videos["heures de videos"]=df_videos['duration_s'].astype(int)/3600  
        
    return df_videos


@st.cache()
def get_table_download_link(df,channel):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    names=df.columns
    csv = df.to_csv(header=names, sep=';',encoding='utf-8',index=False, decimal=",")
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="video_{channel}.csv"><button style=\"background-color: #14A1A1;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">Télécharger le CSV</button></a>'
    return href

def create_barchart_12month(df,x,y,color,title):
    if len(df)>0:
        if len(df)>=12:
            st.subheader(title+" - 12 derniers mois d'activité")
            fig = px.bar(df[:12], x=x, y=y,color=color)
            st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
        else:
            st.subheader(title+" - derniers mois d'activité connus")
            fig = px.bar(df, x=x, y=y,color=color)
            st.plotly_chart(fig, use_container_width=True, sharing='streamlit')     

    else:
        st.subheader(title+" - derniers mois d'activité connus")
        st.info("La chaine est inactive")
        
    return fig
   
def main():
    st.set_page_config(
        page_title="Twitch - vidéos",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    ###################################
    # PARAMETRES DE LA SIDEBAR
    st.sidebar.title('Paramètres')          
    channel=st.sidebar.text_input("Entrez un nom de chaine Twitch", value='gotaga', max_chars=None, key=None, type='default') 
    st.sidebar.write("Entrez vos clés API, après avoir enregistré une <a href=\"https://dev.twitch.tv/console/apps\" target=\"_blank\">application</a>",unsafe_allow_html=True)
    key=st.sidebar.text_input("API KEY", value='xxxx', max_chars=None, key=None, type='default')   
    secret=st.sidebar.text_input("SECRET KEY", value='xxxx', max_chars=None, key=None, type='default')
    
    # initialisation des paramètres de requête API
    periode=TimePeriod.ALL
    video_type=VideoType.ALL
    classement=SortMethod.TIME

    
    if st.sidebar.button("Valider"):       
        with st.spinner("Connexion à Twitch"):
            try:    
                twitch=create_instance(key,secret)
            except:
                st.error("Impossible de se connecter à l'API Twitch, vérifiez vos identifiants")
               
        with st.spinner("Récupération des infos de la chaine"):
            user_info=dict()
            user_info["data"]=[]
            try:
                users_json=twitch.get_users(logins=channel)
                user_info["data"]=user_info["data"]+users_json['data']
                                
                st.write("<div style=\"background-color: #e1e1e1;float:left;padding:10px 10px 10px 10px;width:100%;border-radius:5px;\"><div><div style=\"float: left;width:20%;\"><a href=\"http://www.twitch.com/"+user_info["data"][0]["login"]+"\" target=\"_blank\"><img src=\""+user_info["data"][0]["profile_image_url"]+"\" style=\"border-radius: 50%;\" width=\"150\"/></img></a></div><div style=\"float: left;width:80%;\"><h1>"+user_info["data"][0]["display_name"]+"</h1><br/>"+user_info["data"][0]["description"]+"<br/><hr><button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">"+str(human_format(user_info["data"][0]["view_count"]))+" vues cumulées</button>&nbsp;<button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">"+user_info["data"][0]["broadcaster_type"]+"</button>&nbsp;<button style=\"background-color: #F63366;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 12px;margin: 4px 2px;border-radius: 8px;\">Chaine créée le "+datetime.strptime(user_info["data"][0]["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%m-%Y")+"</button></div></div></div>",unsafe_allow_html=True)
                
        
                df_videos=pd.DataFrame()
                with st.spinner("Analyse des vidéos en cours"):
                    try:
                        video_data=collect_video(twitch,user_info["data"][0]["id"],periode,classement,video_type)
                        df_videos=parsing_videos(video_data)

                        # on crée des df agrégés
                        df_vid=df_videos.groupby(["month","video_type"]).agg({"video_id":"nunique"}).reset_index()\
                        .sort_values(by="month",ascending=False)   
                        df_vid["video_type"]=df_vid["video_type"].str.replace("VideoType.","")
                        df_month=df_videos.groupby("month").agg({"heures de videos":"sum","view_count":"sum"}).reset_index().sort_values(by="month",ascending=False) 

                        # on calcule le mois le plus ancien sur une période de 12 derniers mois d'activité
                        min_month=df_vid["month"][:12].min()  

                        ###################################
                        # AFFICHAGE DES KEY METRICS
                        st.subheader("Key metrics")
                        st.write("<table style=\"border-collapse: collapse;margin: 25px 0;font-size: 0.9em;font-family: sans-serif;min-width: 400px;width:100%;box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);\"><thead style=\"background-color: #F63366;color: #ffffff;text-align: left;\"><tr><th>KPI</th><th>Depuis la création de la chaine</th><th>12 derniers mois d'activité</th></tr></thead><tbody><tr><td><b>#vidéos</b></td><td>"+str(len(df_videos["video_id"].unique()))+" vidéos</td><td>"+str(df_vid["video_id"][:12].sum())+" vidéos</td></tr><tr><td><b>#vues</b></td><td>"+str(human_format(df_videos["view_count"].sum()))+" vues</td><td>"+str(human_format(df_month["view_count"][:12].sum()))+" vues</td></tr><tr><td><b>Moyenne de vues par vidéo</b></td><td>"+str(round(df_videos["view_count"].mean()))+" vues</td><td>"+str(round(df_videos[df_videos['month']>=min_month]["view_count"].mean()))+" vues</td></tr><tr><td><b>Durée cumulée des vidéos</b></td><td>"+str(round(df_videos["heures de videos"].sum()))+" heures</td><td>"+str(round(df_videos[df_videos['month']>=min_month]["heures de videos"].sum()))+" heures</td></tr></tbody></table>", unsafe_allow_html=True)

                        ###################################
                        # AFFICHAGE DES DIAGRAMMES BARRES

                        create_barchart_12month(df_vid,'month','video_id',"video_type","#Vidéos")    
                        create_barchart_12month(df_month,'month','heures de videos',None,"#Durée des contenus")    
                        create_barchart_12month(df_month,'month','view_count',None,"#Vues")

                        ###################################
                        # AFFICHAGE DES VIDEOS LES PLUS CONSULTEES
                        st.subheader("Vidéos les plus consultées (12 derniers mois) :")
                        html_str=""
                        df_top=df_videos[df_videos["month"]>=min_month].sort_values(by="view_count", ascending=False)[:5].reset_index()
                        for i,vid in df_top.iterrows():
                            html_str=html_str+"<tr style=\"width:100%\"><td><b>"+str(i+1)+"</b></td><td><img src=\""+vid.thumbnail_url+"\" width=\"100\"></img></td><td><a href=\""+vid.url+"\" target=\"_blank\">"+vid.title+"</a><br/>Créée le : "+datetime.strptime(vid.created_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%d-%m-%Y %H:%M:%S")+"<br/>Durée : "+vid.duration+"<br/><b>"+str(human_format(vid.view_count))+" vues</b></td></tr>"                

                        st.write("<table style=\"border-collapse: collapse;margin: 25px 0;font-size: 0.9em;font-family: sans-serif;min-width: 400px;width:100%;box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);\"><thead style=\"background-color: #F63366;color: #ffffff;text-align: left;\"><tr><th>Rang</th><th>thumbnail</th><th>Vidéos les plus vues</th></tr></thead><tbody>"+html_str+"</tbody></table>",unsafe_allow_html=True)

                        ###################################
                        # AFFICHAGE DES DONNEES COLLECTEES
                        st.subheader("Aperçu des vidéos :")
                        cols=['video_id','login','title','description','published_at','url','view_count','video_type','duration']
                        st.markdown(get_table_download_link(df_videos[cols],channel), unsafe_allow_html=True)
                        st.write(df_videos[cols])

                    except:
                        pass
                        st.error("Impossible de récupérer les vidéos")

            except:
                pass
                st.error("Impossible de récupérer les infos de cette chaine") 
    st.sidebar.write("<br/><br/><p><center><a href=\"http://www.erwanlenagard.com\" target=\"_blank\" style=\"color:#434444;\">@Erwan Le Nagard</a></center></p>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()    