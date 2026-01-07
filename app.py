from flask import Flask, render_template, request, jsonify, session
import random
import secrets
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# configurazione dei file delle parole
WORD_FILES = {
    'it': "parole_it.txt",
    'en': "words_en.txt"
}
MAX_ATTEMPTS = 6
SCORES_FILE = "classifica.csv"
WORD_STATS_FILE = "word_statistics.json"
PLAYERS_FILE = "players.json"
ANALYTICS_FILE = "analytics.npy"

# caricamento e gestione salvataggi, giocatori
def load_players():
    """legge il file JSON che contiene tutti i giocatori registrati"""
    if os.path.exists(PLAYERS_FILE):
        with open(PLAYERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_players(players):
    """scrive i dati dei giocatori all'interno del file JSON"""
    with open(PLAYERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(players, f, indent=2, ensure_ascii=False)


def calculate_player_analytics(username):
    """calcola le statistiche avanzate del giocatore usando numpy e pandas"""
    if not os.path.exists(SCORES_FILE):
        return None
    
    #tramite pandas caricare i file e i risultati
    df_scores = pd.read_csv(SCORES_FILE)
    player_data = df_scores[df_scores['player'] == username]
    
    if len(player_data) == 0:
        return None
    
    #tramite numpy analizziamo le statistiche dei giocatori 
    scores_array = player_data['score'].values
    
    analytics = {
        'mean_score': float(np.mean(scores_array)),
        'min_score': int(np.min(scores_array)),
        'max_score': int(np.max(scores_array)),
        'total_games': len(player_data),
        'win_rate': float(player_data['won'].mean() * 100),
    }
    
    return analytics

def update_player_stats(username, won, attempts, lang):
    """aggiorna le statistiche del giocatore dopo ogni partita ed utilizza numpy per calcoli statistici complessi"""
    players = load_players()
    
    if username not in players:
        return False
    
    player = players[username]
    
    #aumentiamo il numero di partite giocate
    player['games_played'] += 1
    
    #in caso di vittoria aumentareil numero di partite vinte 
    if won:
        player['games_won'] += 1
    
    #umentare il numero di tenativi totali
    player['total_attempts'] += attempts
    player['last_played'] = datetime.now().isoformat()
    
    #tramite pandas gestire le statistciche del giocatore in base alla lingua selezionata
    if lang not in player['lang_stats']:
        player['lang_stats'][lang] = {'played': 0, 'won': 0}
    
    player['lang_stats'][lang]['played'] += 1
    if won:
        player['lang_stats'][lang]['won'] += 1
    
    #calacolare il punteggio 
    score = 100 - (attempts * 10)
    if won:
        score += 50
    
    player['total_score'] += score
    player['best_score'] = max(player.get('best_score', 0), score)
    
    #tramite numpy calcolare la media dei punteggi per ricavare la media
    if os.path.exists(SCORES_FILE):
        df = pd.read_csv(SCORES_FILE)
        user_scores = df[df['player'] == username]['score'].values
        if len(user_scores) > 0:
            player['average_score'] = float(np.mean(user_scores))
    
    #gestire il numero di vittorie consecutive
    if won:
        player['current_streak'] = player.get('current_streak', 0) + 1
        player['best_streak'] = max(player.get('best_streak', 0), player['current_streak'])
    else:
        player['current_streak'] = 0
    
    save_players(players)
    
    return True

#funzioni per gestire le staistiche delle parole 
def load_word_statistics():
    """carica le statistiche di quante volte vengono utilizzate le parole all'interno del file JSON"""
    if os.path.exists(WORD_STATS_FILE):
        with open(WORD_STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'it': {}, 'en': {}}


def save_word_statistics(stats):
    """salva le statistiche delle parole utilizzate nel file JSON"""
    with open(WORD_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def analyze_word_frequency(lang='it'):
    """analizza con che frequenza escono certe parole usando pandas"""
    stats = load_word_statistics()
    
    if lang not in stats or not stats[lang]:
        return pd.DataFrame()
    
    #utilizziamo Dataframe per calcolare le analisi
    words_data = []
    for word, data in stats[lang].items():
        words_data.append({
            'word': word,
            'count': data['count'],
            'first_used': data.get('first_used', 'N/A'),
            'last_used': data.get('last_used', 'N/A')
        })
    
    df = pd.DataFrame(words_data)
    
    if len(df) > 0:
        #aggiunta delle colonne per le statistiche
        df['frequency_rank'] = df['count'].rank(ascending=False, method='dense')
        df['usage_percentage'] = (df['count'] / df['count'].sum()) * 100
    
    return df

def increment_word_count(word, lang='it'):
    """incrementa il contatore di utilizzo per una specifica parola"""
    stats = load_word_statistics()
    
    if lang not in stats:
        stats[lang] = {}
    
    if word in stats[lang]:
        stats[lang][word]['count'] += 1
        stats[lang][word]['last_used'] = datetime.now().isoformat()
    else:
        stats[lang][word] = {
            'count': 1,
            'first_used': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat()
        }
    
    
    save_word_statistics(stats)
    

def get_top_words(lang='it', limit=10):
    """restituisce le tot parole più utilizzate usando pandas per ordinarle"""
    df = analyze_word_frequency(lang)
    
    if df.empty:
        return []
    
    #ordina in base a qunate volte sono uscite le parole e prendi le prime
    top_df = df.nlargest(limit, 'count')
    
    return top_df.to_dict('records')

#gestione parole
def get_random_word(lang='it'):
    """seleziona casualmente una parola usando numpy random più efficiente per grandi dataset"""
    file_parole = WORD_FILES.get(lang, WORD_FILES['it'])
    
    if not os.path.exists(file_parole):
        print(f"[ERROR] File {file_parole} non trovato!")
        return "ERROR"
    
    with open(file_parole, "r", encoding='utf-8') as file:
        parole = file.read().strip().split(",")
        
    #tramite numpy facciamo una selezione randomica
    idx = np.random.randint(0, len(parole))
    word = parole[idx].upper()
    
    increment_word_count(word, lang)
    
    return word

#gestione della classifica e dei sitemi di punteggio

def save_score(player_name, attempts, won, lang):
    """salva il punteggio e utilizzando pandas gestire in modo efficiente il file CSV"""
    score = 100 - (attempts * 10)
    if won:
        score += 50
    
    #creazione o caricamento Dataframe
    if os.path.exists(SCORES_FILE):
        df = pd.read_csv(SCORES_FILE)
    else:
        df = pd.DataFrame(columns=['player', 'score', 'attempts', 'won', 'lang', 'timestamp'])
    
    # aggiungi nuova statistica giocatore
    new_row = pd.DataFrame([{
        'player': player_name,
        'score': score,
        'attempts': attempts,
        'won': won,
        'lang': lang,
        'timestamp': datetime.now().isoformat()
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)
    
    #ordina in base al punteggio
    df = df.sort_values(by='score', ascending=False)
    
    #tramite pandas salva e trasforma in un file csv
    df.to_csv(SCORES_FILE, index=False)
    
#utilizzo di numpy per i calcoli più complicati
class Game:
    """gestisce una singola partita del gioco e usa numpy per operazioni su array di lettere"""
    
    def __init__(self, lang='it', secret_word=None, attempts=0, guesses = [], game_over=False, won=False):
        self.lang = lang
        self.secret_word = get_random_word(lang) if secret_word is None else secret_word
        self.attempts = attempts
        self.guesses = guesses
        self.game_over = game_over
        self.won = won
        
    def check_guess(self, guess):
        """controlla un tentativo usando numpy per confronti veloci"""
        guess = guess.upper().strip()
        
        #verificare se l'input è valido
        if len(guess) != 5:
            return {'success': False, 'error': 'La parola deve essere di 5 lettere!'}
        
        if not guess.isalpha():
            return {'success': False, 'error': 'Inserisci solo lettere!'}
        
        if self.game_over:
            return {'success': False, 'error': 'Il gioco è già finito!'}
        
        #analizzare la parola
        results = self._check_word_logic(guess)
        self.attempts += 1
        self.guesses.append({'word': guess, 'results': results})
        
        #verificare se il giocatore ha vinto oppure ha finito i tenativi
        self.won = guess == self.secret_word
        self.game_over = self.won or self.attempts >= MAX_ATTEMPTS
        
        return {
            'success': True,
            'results': results,
            'attempts': self.attempts,
            'max_attempts': MAX_ATTEMPTS,
            'game_over': self.game_over,
            'won': self.won,
            'secret_word': self.secret_word if self.game_over else None,
        }
    
    
    def _check_word_logic(self, guess):
        """
        Algoritmo di confronto usando numpy per operazioni vettoriali.
        """
        #converitre stringhe in array numpy
        guess_arr = np.array(list(guess))
        secret_arr = np.array(list(self.secret_word))
        
        results = []
        used = np.zeros(5, dtype=bool)
        
        #controllare se le lettere sono nella posizione giusta e segnalrle in verde
        correct_mask = guess_arr == secret_arr
        
        for i in range(5):
            if correct_mask[i]:
                results.append({
                    'letter': guess_arr[i],
                    'status': 'correct',
                    'position': i
                })
                used[i] = True
            else:
                results.append({
                    'letter': guess_arr[i],
                    'status': 'absent',
                    'position': i
                })
        
        #controllare se le lettere sono presenti nella parola e segnalarle in giallo
        for i in range(5):
            if not correct_mask[i]:
                for j in range(5):
                    if not used[j] and guess_arr[i] == secret_arr[j]:
                        results[i]['status'] = 'present'
                        used[j] = True
                        break
        
        return results
    
    def get_state(self):
        """Restituisce lo stato completo con analytics numpy"""
        return {
            'attempts': self.attempts,
            'max_attempts': MAX_ATTEMPTS,
            'guesses': self.guesses,
            'game_over': self.game_over,
            'won': self.won,
            'lang': self.lang,
            'secret_word': self.secret_word
        }

#intaurare le rotte per l'html
@app.route('/')
def root():
    """Homepage - Redirect alla home"""
    return render_template('home.html')


@app.route('/home')
def home():
    """Pagina di login/registrazione"""
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard personale post-login"""
    player_session = session.get('player')
    if not player_session:
        return render_template('home.html')
    return render_template('dashboard.html')


@app.route('/game')
def index():
    """Pagina di gioco principale"""
    player_session = session.get('player')
    if not player_session:
        return render_template('home.html')
    return render_template('index.html')


@app.route('/profile')
def profile():
    """Pagina profilo utente"""
    player_session = session.get('player')
    if not player_session:
        return render_template('home.html')
    return render_template('profile.html')


@app.route('/word-statistics')
def word_statistics():
    """Pagina statistiche parole utilizzate"""
    return render_template('word_statistics.html')


@app.route('/player-statistics')
def player_statistics_page():
    """Pagina statistiche e classifiche giocatori"""
    return render_template('player_statistics.html')

#autenticazione degli api, login, logout e registrazione

@app.route('/players', methods=['POST'])
def create_player():
    """Crea un nuovo account giocatore"""
    data = request.get_json()
    nome = data.get('nome', '').strip()
    username = data.get('username', '').strip()
    
    if not nome or not username:
        return jsonify({
            'success': False,
            'error': 'Nome e username obbligatori'
        }), 400
    
    players = load_players()
    
    if username in players:
        return jsonify({
            'success': False,
            'error': 'Username già esistente'
        }), 409
    
    #creare un nuovo profilo di un giocatore
    players[username] = {
        'nome': nome,
        'username': username,
        'created_at': datetime.now().isoformat(),
        'last_played': None,
        'games_played': 0,
        'games_won': 0,
        'total_attempts': 0,
        'total_score': 0,
        'average_score': 0.0,
        'best_score': 0,
        'current_streak': 0,
        'best_streak': 0,
        'lang_stats': {}
    }
    
    save_players(players)
    
    session['player'] = {
        'nome': nome,
        'username': username
    }
    
    return jsonify({
        'success': True,
        'player': players[username],
        'message': f'Giocatore {username} creato con successo!'
    })


@app.route('/login', methods=['POST'])
def login():
    """Effettua il login di un giocatore esistente"""
    data = request.get_json()
    username = data.get('username', '').strip()
    
    if not username:
        return jsonify({
            'success': False,
            'error': 'Username obbligatorio'
        }), 400
    
    players = load_players()
    
    if username not in players:
        return jsonify({
            'success': False,
            'error': 'Username non trovato'
        }), 404
    
    player = players[username]
    
    session['player'] = {
        'nome': player['nome'],
        'username': username
    }
    
    return jsonify({
        'success': True,
        'player': player,
        'message': f'Benvenuto {player["nome"]}!'
    })


@app.route('/logout', methods=['POST'])
def logout():
    """Effettua il logout e pulisce la sessione"""
    username = session.get('player', {}).get('username', 'Unknown')
    
    session.pop('player', None)
    session.pop('game_state', None)

    return jsonify({
        'success': True,
        'message': 'Logout effettuato'
    })


@app.route('/check-session', methods=['GET'])
def check_session():
    """Verifica se esiste una sessione attiva valida"""
    player = session.get('player')
    
    if not player:
        return jsonify({
            'authenticated': False,
            'message': 'Nessuna sessione attiva'
        })
    
    players = load_players()
    username = player.get('username')
    
    if username not in players:
        session.pop('player', None)
        return jsonify({
            'authenticated': False,
            'message': 'Giocatore non trovato'
        })
    
    return jsonify({
        'authenticated': True,
        'player': players[username]
    })

#gestione dei tenativi e delle partite

@app.route('/new-game', methods=['POST'])
def new_game():
    """Inizia una nuova partita"""
    player_session = session.get('player')
    
    if not player_session:
        return jsonify({
            'success': False,
            'error': 'Devi essere autenticato per iniziare una partita!'
        }), 401
    
    data = request.get_json()
    lang = data.get('lang', 'it')
    
    game = Game(lang)
    session['game_state'] = game.get_state()
    
    return jsonify({
        'success': True,
        'message': f'Nuova partita iniziata in {lang}!',
        'max_attempts': MAX_ATTEMPTS
    })


@app.route('/check-word', methods=['POST'])
def check_word():
    """Controlla un tentativo del giocatore"""
    player_session = session.get('player')
    
    if not player_session:
        return jsonify({
            'success': False,
            'error': 'Devi essere autenticato per giocare!'
        }), 401
    
    data = request.get_json()
    guess = data.get('word', '')
    username = player_session.get('username')
    
    game_state = session.get('game_state')
    
    if not game_state:
        return jsonify({
            'success': False,
            'error': 'Nessuna partita attiva!'
        })
    
    #ricreare lo stato del gioco una volta terminata la sessione e riaperta
    game = Game(game_state['lang'], game_state['secret_word'], game_state['attempts'],game_state['guesses'], game_state['game_over'], game_state['won'])
    
    result = game.check_guess(guess)
    session['game_state'] = game.get_state()
    
    #in caso di partita terminata aggiornare le satistiche 
    if result.get('game_over'):
        update_player_stats(username, game.won, game.attempts, game.lang)
        save_score(username, game.attempts, game.won, game.lang)
        
        #calcolare le statistiche avanzate 
        analytics = calculate_player_analytics(username)
        result['analytics'] = analytics

    return jsonify(result)


@app.route('/get-secret-word', methods=['GET'])
def get_secret_word():
    """Mostra la parola segreta (cheat per debug/aiuto)"""
    game_state = session.get('game_state')
    
    if not game_state:
        return jsonify({
            'success': False,
            'error': 'Nessuna partita attiva!'
        })
    
    return jsonify(game_state)

@app.route('/api/word-stats/all', methods=['GET'])
def api_all_word_stats():
    """Restituisce le statistiche per tutte le lingue con pandas"""
    limit = int(request.args.get('limit', 10))
    stats = load_word_statistics()
    result = {}
    
    for lang in stats.keys():
        result[lang] = get_top_words(lang, limit)
    
    return jsonify({
        'success': True,
        'statistics': result
    })

@app.route('/player-stats', methods=['GET'])
def player_stats():
    """Restituisce le statistiche del giocatore loggato con analytics"""
    player_session = session.get('player')
    
    if not player_session:
        return jsonify({
            'success': False,
            'error': 'Non autenticato'
        }), 401
    
    username = player_session.get('username')
    players = load_players()
    
    if username not in players:
        return jsonify({
            'success': False,
            'error': 'Giocatore non trovato'
        }), 404
    
    #statistiche avanzate
    analytics = calculate_player_analytics(username)
    
    return jsonify({
        'success': True,
        'stats': players[username],
        'analytics': analytics
    })

@app.route('/top-players', methods=['GET'])
def top_players():
    """Restituisce la classifica dei migliori giocatori con pandas"""
    limit = int(request.args.get('limit', 10))
    players = load_players()
    
    #crazione Dataframe
    df = pd.DataFrame([
        {
            'username': username,
            'nome': data['nome'],
            'total_score': data['total_score'],
            'games_played': data['games_played'],
            'games_won': data['games_won'],
            'best_score': data['best_score'],
            'best_streak': data.get('best_streak', 0),
        }
        for username, data in players.items()
    ])
    
    if df.empty:
        return jsonify({
            'success': True,
            'total_players': 0,
            'leaderboard': []
        })
    
    #tramite numpy calcolare il rapporto vittorie-sconfitte
    df['win_rate'] = np.where(
        df['games_played'] > 0,
        np.round((df['games_won'] / df['games_played']) * 100, 2),
        0
    )
    
    #ordina le parole e fai un limite di parole con un massimo di 10
    df = df.sort_values(by='total_score', ascending=False).head(limit)
    
    return jsonify({
        'success': True,
        'total_players': len(players),
        'leaderboard': df.to_dict('records')
    })

@app.route('/rules', methods=['GET'])
def rules():
    """Restituisce le regole del gioco"""
    rules_text = """
Benvenuto al gioco delle parole a 5 lettere!

Regole del gioco:

1. Devi indovinare la parola segreta composta da 5 lettere.

2. Hai un massimo di 6 tentativi per indovinare la parola.

3. Dopo ogni tentativo riceverai un feedback per ciascuna lettera:
   - CORRECT (Verde): la lettera è corretta e nella posizione giusta.
   - PRESENT (Giallo): la lettera è presente nella parola, ma in un'altra posizione.
   - ABSENT (Grigio): la lettera non è presente nella parola.

4. Il punteggio finale dipende dal numero di tentativi:
   - Vittoria: +50 punti
   - Ogni tentativo: -10 punti
   - Minor numero di tentativi = punteggio più alto

5. Puoi giocare in Italiano o Inglese selezionando la lingua all'inizio.

6. Le tue statistiche vengono salvate automaticamente:
   - Partite giocate
   - Partite vinte
   - Punteggio totale
   - Miglior punteggio
   - Serie di vittorie consecutive

Buon divertimento!  
"""
    return jsonify({'rules': rules_text})

#tramite beautifulsoup popolare il databese delle parole da indovinare prese da un sito utilizzanod il web scraping

def prendi_parole_italiane():
    """Scarica parole italiane da listediparole.it tramite web scraping"""
    import requests
    from bs4 import BeautifulSoup
    
    print("Inizio download parole italia...")
    lista_parole = []
    
    try:
        url = "https://www.listediparole.it/5lettereparole.htm"
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.content, 'html.parser')
        parole = soup.find(class_="mt")
        
        if parole:
            lista_parole = parole.text.strip().split(" ")
        
        for pagenumber in range(2, 18):
            url = f"https://www.listediparole.it/5lettereparolepagina{pagenumber}.htm"
            
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, 'html.parser')
            parole = soup.find(class_="mt")
            
            if parole:
                nuove_parole = parole.text.strip().split(" ")
                lista_parole += nuove_parole
        
        #creare array di numpy e controllare che vengano salvate soltanto le parole con 5 lettere
        parole_array = np.array([p.strip().upper() for p in lista_parole if p.strip() and len(p.strip()) == 5])
        lista_parole = np.unique(parole_array).tolist()
        
        return lista_parole
    
    except Exception as e:
        return [
            'CARNE', 'PALLA', 'FORNO', 'PORTA', 'SEDIA',
            'PIANO', 'LIBRO', 'CAMPO', 'TEMPO', 'MONDO'
        ]


def prendi_parole_inglesi():
    """Scarica parole inglesi dal dataset Stanford"""
    import requests
    
    print("[SCRAPING] Inizio download parole inglesi...")
    lista_parole = []
    
    try:
        url = "https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            parole = response.text.strip().split('\n')
            
            parole_array = np.array([p.strip().upper() for p in parole if len(p.strip()) == 5])
            lista_parole = parole_array.tolist()
            
        if len(lista_parole) < 100:
            parole_aggiuntive = [
                'APPLE', 'HOUSE', 'TABLE', 'PLANT', 'WATER', 'LIGHT', 'WORLD', 'SOUND',
                'GREAT', 'SMALL', 'FOUND', 'STILL', 'LEARN', 'WRITE', 'SPELL', 'THEIR',
                'WOULD', 'ABOUT', 'AFTER', 'AGAIN', 'PLACE', 'THINK', 'THREE', 'WHERE',
                'COULD', 'RIGHT', 'FIRST', 'WHICH', 'THERE', 'THESE', 'OTHER',
                'UNDER', 'EVERY', 'POINT', 'LARGE', 'BEING', 'MIGHT', 'NEVER', 'STORY',
                'ALONG', 'ASKED', 'BEGAN', 'BRING', 'EARTH', 'GIVEN', 'HEARD', 'HORSE',
                'LIVED', 'TAKEN', 'TODAY', 'TRIED', 'YOUNG', 'ABOVE', 'AMONG',
                'BELOW', 'BUILT', 'CATCH', 'DRAWN', 'EARLY', 'FIGHT', 'FRONT',
                'GLASS', 'HEART', 'LAUGH', 'REACH', 'RIVER', 'SHALL', 'SLEEP', 'STOOD',
                'WATCH', 'WHOLE', 'WOMAN', 'WROTE', 'BRAIN', 'BREAD', 'BREAK', 'BRIEF'
            ]
            lista_parole.extend(parole_aggiuntive)
        
        #tramite numpy rimuovere i duplicati
        lista_parole = np.unique(np.array([p for p in lista_parole if p.isalpha() and len(p) == 5])).tolist()
        
        return lista_parole

    except Exception as e:
        print("Impossibile scaricare parole inglesi: " + str(e))
        return [
            'APPLE', 'HOUSE', 'TABLE', 'PLANT', 'WATER',
            'LIGHT', 'WORLD', 'SOUND', 'GREAT', 'SMALL'
        ]


def salva_parole(lang='it', file_parole="parole_it.txt"):
        """Scarica e salva le parole per la lingua specificata"""
        if not os.path.exists(file_parole):
            
            if lang == 'it':
                parole = prendi_parole_italiane()
            elif lang == 'en':
                parole = prendi_parole_inglesi()
            else:
                parole = prendi_parole_italiane()
            
            with open(file_parole, "w", encoding='utf-8') as file:
                file.write(",".join(parole))
            
        else:
            print("File " + file_parole + " già esistente")
    #avvio del gioco

if __name__ == '__main__':
        print("\n" + "="*60)
        print(" "*15 + "WORDLE GAME - INIZIALIZZAZIONE")
        print("="*60 + "\n")
        
        #creare e sistemare i file delle parole per ogni lingua
        for lang, file_name in WORD_FILES.items():
            salva_parole(lang, file_name)
            print()
        
        #creare e sistemare il file dei giocatori
        if not os.path.exists(PLAYERS_FILE):
            print("Creazione file database giocatori: " + PLAYERS_FILE)
            save_players({})
        else:
            print("Database giocatori trovato: " + PLAYERS_FILE)
        
        #creare e sistemare il file con le statistiche delle parole
        if not os.path.exists(WORD_STATS_FILE):
            print("Creazione file statistiche parole: " + WORD_STATS_FILE)
            save_word_statistics({'it': {}, 'en': {}})
        else:
            print("File statistiche parole trovato: " + WORD_STATS_FILE)
        
        app.run(debug=True, port=5000, host='0.0.0.0')
