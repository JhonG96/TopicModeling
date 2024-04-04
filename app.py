from flask import Flask, render_template, request
from PyPDF2 import PdfReader
#run in python console
import nltk
nltk.download('stopwords')
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('spanish')
stop_words.extend(['del', 'por', 're', 'a', 'le','su','https','of','in','the','los','la','on','are','in','to','and','y','por','algún',","'alguna',","'algunas',","'alguno',","'algunos',","'ambos',","'ampleamos',","'ante',","'antes',","'aquel',","'aquellas',","'aquellos',","'aqui',","'arriba',","'atras',","'bajo',","'bastante',","'bien',","'cada',","'cierta',","'ciertas',","'cierto',","'ciertos',","'como',","'con',","'conseguimos',","'conseguir',","'consigo',","'consigue',","'consiguen',","'consigues',","'cual',","'cuando',","'dentro',","'desde',","'donde',","'dos',","'el',","'ellas',","'ellos',","'empleais',","'emplean',","'emplear',","'empleas',","'empleo',","'en',","'encima',","'entonces',","'entre',","'era',","'eramos',","'eran',","'eras',","'eres',","'es',","'esta',","'estaba',","'estado',","'estais',","'estamos',","'estan',","'estoy',","'fin',","'fue',","'fueron',","'fui',","'fuimos',","'gueno',","'ha',","'hace',","'haceis',","'hacemos',","'hacen',","'hacer',","'haces',","'hago',","'incluso',","'intenta',","'intentais',","'intentamos',","'intentan',","'intentar',","'intentas',","'intento',","'ir',","'la',","'largo',","'las',","'lo',","'los',","'mientras',","'mio',","'modo',","'muchos',","'muy',","'nos',","'nosotros',","'otro',","'para',","'pero',","'podeis',","'podemos',","'poder',","'podria',","'podriais',","'podriamos',","'podrian',","'podrias',","'por',","'por qué',","'porque',","'primero',","'puede',","'pueden',","'puedo',","'quien',","'sabe',","'sabeis',","'sabemos',","'saben',","'saber',","'sabes',","'ser',","'si',","'siendo',","'sin',","'sobre',","'sois',","'solamente',","'solo',","'somos',","'soy',","'su',","'sus',","'también',","'teneis',","'tenemos',","'tener',","'tengo',","'tiempo',","'tiene',","'tienen',","'todo',","'trabaja',","'trabajais',","'trabajamos',","'trabajan',","'trabajar',","'trabajas',","'trabajo',","'tras',","'tuyo',","'ultimo',","'un',","'una',","'unas',","'uno',","'unos',","'usa',","'usais',","'usamos',","'usan',","'usar',","'usas',","'uso',","'va',","'vais',","'valor',","'vamos',","'van',","'vaya',","'verdad',","'verdadera',","'verdadero',","'vosotras',","'vosotros',","'voy',","'yo',","'él',","'ésta',","'éstas',","'éste',","'éstos',","'última',","'últimas',","'último',","'últimos',","'a',","'añadió',","'aún',","'actualmente',","'adelante',","'además',","'afirmó',","'agregó',","'ahí',","'ahora',","'al',","'algo',","'alrededor',","'anterior',","'apenas',","'aproximadamente',","'aquí',","'así',","'aseguró',","'aunque',","'ayer',","'buen',","'buena',","'buenas',","'bueno',","'buenos',","'cómo',","'casi',","'cerca',","'cinco',","'comentó',","'conocer',","'consideró',","'considera',","'contra',","'cosas',","'creo',","'cuales',","'cualquier',","'cuanto',","'cuatro',","'cuenta',","'da',","'dado',","'dan',","'dar',","'de',","'debe',","'deben',","'debido',","'decir',","'dejó',","'del',","'demás',","'después',","'dice',","'dicen',","'dicho',","'dieron',","'diferente',","'diferentes',","'dijeron',","'dijo',","'dio',","'durante',","'e',","'ejemplo',","'ella',","'ello',","'embargo',","'encuentra',","'esa',","'esas',","'ese',","'eso',","'esos',","'está',","'están',","'estaban',","'estar',","'estará',","'estas',","'este',","'esto',","'estos',","'estuvo',","'ex',","'existe',","'existen',","'explicó',","'expresó',","'fuera',","'gran',","'grandes',","'había',","'habían',","'haber',","'habrá',","'hacerlo',","'hacia',","'haciendo',","'han',","'hasta',","'hay',","'haya',","'he',","'hecho',","'hemos',","'hicieron',","'hizo',","'hoy',","'hubo',","'igual',","'indicó',","'informó',","'junto',","'lado',","'le',","'les',","'llegó',","'lleva',","'llevar',","'luego',","'lugar',","'más',","'manera',","'manifestó',","'mayor',","'me',","'mediante',","'mejor',","'mencionó',","'menos',","'mi',","'misma',","'mismas',","'mismo',","'mismos',","'momento',","'mucha',","'muchas',","'mucho',","'nada',","'nadie',","'ni',","'ningún',","'ninguna',","'ningunas',","'ninguno',","'ningunos',","'no',","'nosotras',","'nuestra',","'nuestras',","'nuestro',","'nuestros',","'nueva',","'nuevas',","'nuevo',","'nuevos',","'nunca',","'o',","'ocho',","'otra',","'otras',","'otros',","'parece',","'parte',","'partir',","'pasada',","'pasado',","'pesar',","'poca',","'pocas',","'poco',","'pocos',","'podrá',","'podrán',","'podría',","'podrían',","'poner',","'posible',","'próximo',","'próximos',","'primer',","'primera',","'primeros',","'principalmente',","'propia',","'propias',","'propio',","'propios',","'pudo',","'pueda',","'pues',","'qué',","'que',","'quedó',","'queremos',","'quién',","'quienes',","'quiere',","'realizó',","'realizado',","'realizar',","'respecto',","'sí',","'sólo',","'se',","'señaló',","'sea',","'sean',","'según',","'segunda',","'segundo',","'seis',","'será',","'serán',","'sería',","'sido',","'siempre',","'siete',","'sigue',","'siguiente',","'sino',","'sola',","'solas',","'solos',","'son',","'tal',","'tampoco',","'tan',","'tanto',","'tenía',","'tendrá',","'tendrán',","'tenga',","'tenido',","'tercera',","'toda',","'todas',","'todavía',","'todos',","'total',","'trata',","'través',","'tres',","'tuvo',","'usted',","'varias',","'varios',","'veces',","'ver',","'vez',","'y',","'ya''i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your','yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are','was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y','ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
from pprint import pprint
import gensim.corpora as corpora
import os.path
import glob


# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route("/")
def home():
    return render_template('index.html')
# ‘/’ URL is bound with hello_world() function.
@app.route('/upload', methods=['POST'])
def upload():
   if request.method == 'POST':   
        f = request.files['file'] 
        f.save(f.filename)   
        return render_template('index.html', name = f.filename)



def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]






@app.route('/topic', methods=['POST'])
def topic():
     folder_path = glob.glob(r"C:\Users\J h o n\TopicModelling\*")
     max_file = max(folder_path, key=os.path.getctime)
     file= os.path.basename(max_file)
     reader = PdfReader(file)
     #reader = PdfReader('Certificado de inscripción agencia de empleo.pdf')
     page = reader.pages[0]
     text = page.extract_text()
     docs = text.split()
     #data = text.paper_text_processed.values.tolist()
     data_words = list(sent_to_words(docs))
     # remove stop words
     data_words = remove_stopwords(data_words)
     data_words = [item for item in data_words if item]
     id2word = corpora.Dictionary(data_words)
     # Create Corpus
     texts = data_words
     # Term Document Frequency
     corpus = [id2word.doc2bow(text) for text in texts]
     num_topics = 10
     # Build LDA model
     lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)
     # Print the Keyword in the 10 topics
     topics = lda_model.show_topics(num_topics=1, num_words=5, log=False, formatted=False)
     topic_id1 = []
 
     dit = []
     for topic_id, topic in topics:
      for word in topic:
         s=''.join(map(str, word[:1]))
         dit.append(s)
     ' '.join(map(str, dit))

     
    
     return render_template('index.html',texto=topic)




# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()