# Google Play Analysis

Cet outil permet d'analyser les commentaires déposés sur le Google Play Store pour une application mobile. 

### Application

<a href="https://share.streamlit.io/erwanlenagard/google_play_analysis/main/app.py" target="_blank">Un prototype</a> est accessible en ligne. 

### Description
<p>Les métadonnées de l'application et les reviews récentes sont capturées à l'aide de <a href="https://github.com/JoMingyu/google-play-scraper" target="_blank">Google Play Scraper</a>. Un rapport est généré présentant les principales métriques d'activité de la chaine ainsi qu'une analyse automatique des reviews récentes :
<ul>
  <li>Key metrics de l'app: note moyenne, nombre d'installations, total de reviews...</li>
  <li>Reviews : évolution mensuelle, évolution du score, taux de réponse du développeur...</li>
  <li>Nuages de mots clés : les termes les plus spécifiques aux reviews négatives et aux reviews positives</li>
  <li>Topic Modeling : analyse des pain points (les sujets propres aux reviews négatives) et des points d'appréciation (les sujets propres aux reviews positives)</li>
  <li>Tableau de l'ensemble des métadonnées capturées, téléchargeable au format csv</li>
</ul></p>

### Notes

<p>L'outil Google Play Analysis est un prototype, il présente certaines limites :</p>
<ul>
  <li>La librairie Google Play Scraper ne retourne pas l'exhaustivité des données présentes sur Google Play Store, néanmoins celles-ci sont suffisamment pertinentes pour effectuer un audit de l'app.</li>
  <li>Si le nombre de reviews dépasse les 1000 commentaires, un échantillon aléatoire est sélectionné pour l'analyse NLP parmi les reviews les plus récentes.</li>
</ul>
