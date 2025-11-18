

from datetime import datetime, timedelta
import sqlite3
import uuid
import random
import time
import tempfile
import os
import sys
import threading
import subprocess
import io
import json

# Optional libs
STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
    from streamlit import session_state
except Exception:
    STREAMLIT_AVAILABLE = False

TRANSFORMERS_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    TRANSFORMERS_AVAILABLE = False

GTTs_AVAILABLE = True
try:
    from gtts import gTTS
except Exception:
    GTTs_AVAILABLE = False

import pandas as pd

# ----------------- Config -----------------
DB_FILE = "mindcheck_single_improved.db"
SIMULATION_MODE = True
DEFAULT_USER_ID = str(uuid.uuid4())[:6]
ALERT_HR = 110
ALERT_HRV = 25
ALERT_MOOD = -1.5

# Mood check intervals in minutes
MOOD_CHECK_INTERVALS = {
    'frequent': 30,    # Every 30 minutes
    'normal': 60,      # Every hour
    'relaxed': 120,    # Every 2 hours
    'minimal': 240     # Every 4 hours
}

EMOJI_LABELS = {
    "😊": (2, "Feliz"),
    "🙂": (1, "Contente"),
    "😐": (0, "Neutro"),
    "😔": (-1, "Triste"),
    "😰": (-2, "Ansioso"),
    "😡": (-2, "Irritado"),
    "😴": (-1, "Cansado")
}

WELLNESS_TIPS = [
    "Alongue-se por 2 minutos.",
    "Beba um copo de água e respire fundo.",
    "Faça uma caminhada rápida de 5 minutos.",
    "Desconecte-se por 2 minutos e observe sua respiração.",
    "Tente ajustar sua postura por 1 minuto.",
    "Lembre-se de respirar profundamente 3 vezes.",
    "Faça uma pausa para um chá ou água.",
    "Ouça uma música que você gosta por alguns minutos."
]

POSITIVE_FEEDBACK = [
    "Que ótimo! Continue assim! 🌟",
    "Excelente! Seu bem-estar está em alta! 🚀",
    "Maravilhoso! Você está no caminho certo! 💫",
    "Incrível! Sua energia positiva é contagiante! 🌈"
]

NEUTRAL_FEEDBACK = [
    "Tudo bem ter dias neutros. Pequenas ações fazem diferença!",
    "Dia equilibrado é um bom dia para observar e aprender.",
    "As vezes uma pausa pode trazer novos insights.",
    "Mantenha a consistência - bons hábitos se constroem assim."
]

ENCOURAGEMENT_FEEDBACK = [
    "Você é mais forte do que imagina! 💪",
    "Dias difíceis não duram para sempre. Você consegue! 🌦️",
    "Cada pequeno passo importa. Continue! 👣",
    "Respire fundo - você já superou desafios antes! 🌬️"
]

# --------------- Database -----------------

def init_db(path=DB_FILE):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS responses (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        timestamp TEXT,
        question TEXT,
        emoji TEXT,
        text TEXT,
        sentiment REAL,
        mood_score INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS physio (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        timestamp TEXT,
        source TEXT,
        heart_rate INTEGER,
        hrv REAL,
        sleep_hours REAL,
        steps INTEGER,
        stress_index REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS reminders (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        timestamp TEXT,
        reminder_type TEXT,
        message TEXT,
        dismissed BOOLEAN DEFAULT FALSE
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        user_id TEXT PRIMARY KEY,
        mood_check_interval TEXT,
        popup_enabled BOOLEAN DEFAULT TRUE,
        system_notifications BOOLEAN DEFAULT TRUE,
        last_mood_check TEXT
    )''')
    conn.commit()
    return conn

conn = init_db()

# --------------- System Notifications ----------------

def show_system_notification(title, message):
    """Show system notification (works on Windows, macOS, Linux)"""
    try:
        if sys.platform == "win32":
            subprocess.Popen(['powershell', '-WindowStyle', 'Hidden', '-Command',
                            f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show("{message}", "{title}")'])
        elif sys.platform == "darwin":
            os.system(f"osascript -e 'display notification \"{message}\" with title \"{title}\"'")
        else:
            os.system(f'notify-send "{title}" "{message}"')
    except Exception as e:
        print(f"Error showing system notification: {e}")


def show_popup_reminder(user_id):
    message = "💭 Hora do check-in de humor! Como você está se sentindo agora?"
    show_system_notification("MindCheck - Lembrete", message)
    create_reminder(user_id, 'periodic_check', message)

# --------------- Settings Management ----------------

def get_user_settings(user_id):
    c = conn.cursor()
    c.execute('SELECT * FROM settings WHERE user_id = ?', (user_id,))
    row = c.fetchone()
    if row:
        return {
            'user_id': row[0],
            'mood_check_interval': row[1],
            'popup_enabled': bool(row[2]),
            'system_notifications': bool(row[3]),
            'last_mood_check': row[4]
        }
    else:
        default_settings = {
            'user_id': user_id,
            'mood_check_interval': 'normal',
            'popup_enabled': True,
            'system_notifications': True,
            'last_mood_check': None
        }
        save_user_settings(default_settings)
        return default_settings


def save_user_settings(settings):
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO settings 
                 (user_id, mood_check_interval, popup_enabled, system_notifications, last_mood_check) 
                 VALUES (?, ?, ?, ?, ?)''',
              (settings['user_id'], settings['mood_check_interval'],
               settings['popup_enabled'], settings['system_notifications'],
               settings['last_mood_check']))
    conn.commit()

# --------------- Sentiment (improved with robust fallbacks) -----------------

def load_sentiment():
    if not TRANSFORMERS_AVAILABLE:
        return None

    # Prefer a Portuguese-capable model if available, then multilingual models
    candidates = [
        'pierreguillou/bert-base-cased-sentiment-portuguese',
        'nlptown/bert-base-multilingual-uncased-sentiment',
        'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    ]
    for model in candidates:
        try:
            pipe = pipeline('sentiment-analysis', model=model)
            return {'pipe': pipe, 'model': model}
        except Exception:
            continue
    # Final fallback: default pipeline
    try:
        pipe = pipeline('sentiment-analysis')
        return {'pipe': pipe, 'model': 'default'}
    except Exception:
        return None

sentiment_model = load_sentiment()


def analyze_sentiment(text):
    if not text:
        return 0.5
    if sentiment_model and sentiment_model.get('pipe'):
        try:
            out = sentiment_model['pipe'](text)
            if isinstance(out, list) and len(out) > 0:
                o = out[0]
                # Hugging Face pipelines vary: some return {'label':'POSITIVE','score':0.99}, others give 'stars'
                label = o.get('label', '').lower()
                score = float(o.get('score', 0.5))
                # Map labels to 0..1 more robustly
                if 'neg' in label or '1' in label:
                    return max(0.0, min(1.0, 0.1 * score + 0.0))
                if 'neu' in label or 'mixed' in label:
                    return 0.5
                if 'pos' in label or '5' in label or '4' in label:
                    return max(0.5, min(1.0, 0.5 + 0.5 * score))
                # default: return score
                return max(0.0, min(1.0, score))
        except Exception:
            pass
    # Fallback heuristic for Portuguese/Brazilian Portuguese
    lower = text.lower()
    neg_words = ['triste', 'cansad', 'estress', 'ansios', 'mau', 'ruim', 'difícil', 'problema', 'deprim', 'sofr', 'solit']
    pos_words = ['bem', 'feliz', 'ótimo', 'excelente', 'maravilha', 'content', 'alegr', 'animad', 'gratid']
    neg = any(w in lower for w in neg_words)
    pos = any(w in lower for w in pos_words)
    if neg and not pos:
        return 0.2
    elif pos and not neg:
        return 0.85
    elif pos and neg:
        return 0.5
    return 0.5

# ------------- Reminder System -------------

def create_reminder(user_id, reminder_type, message):
    c = conn.cursor()
    rid = str(uuid.uuid4())
    c.execute('INSERT INTO reminders (id, user_id, timestamp, reminder_type, message) VALUES (?,?,?,?,?)',
              (rid, user_id, datetime.utcnow().isoformat(), reminder_type, message))
    conn.commit()
    return rid


def get_pending_reminders(user_id):
    c = conn.cursor()
    c.execute('SELECT id, reminder_type, message FROM reminders WHERE user_id = ? AND dismissed = FALSE ORDER BY timestamp DESC LIMIT 10',
              (user_id,))
    return c.fetchall()


def dismiss_reminder(reminder_id):
    c = conn.cursor()
    c.execute('UPDATE reminders SET dismissed = TRUE WHERE id = ?', (reminder_id,))
    conn.commit()


def check_reminders_needed(user_id):
    c = conn.cursor()
    c.execute('SELECT timestamp, mood_score FROM responses WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (user_id,))
    last_mood = c.fetchone()

    reminders = []

    if last_mood:
        try:
            last_time = datetime.fromisoformat(last_mood[0])
        except Exception:
            last_time = datetime.utcnow()
        mood_score = last_mood[1]

        if datetime.utcnow() - last_time > timedelta(hours=4):
            reminders.append(("checkin_reminder", "💭 Hora do check-in! Como você está se sentindo agora?"))

        if mood_score is not None and mood_score <= -1 and datetime.utcnow() - last_time > timedelta(hours=2):
            reminders.append(("followup_reminder", "❤️‍🩹 Verificando... Como está se sentindo após o último check-in?"))

    physio = recent_physio_summary(user_id, minutes=240)
    if physio:
        if physio.get('stress_index', 0) > 0.7:
            reminders.append(("stress_reminder", "🧘‍♂️ Níveis de estresse altos detectados. Que tal uma pausa para respirar?"))
        if physio.get('steps', 0) < 1000:
            reminders.append(("activity_reminder", "🚶‍♂️ Pouca atividade hoje. Uma pequena caminhada pode ajudar!"))

    return reminders

# ------------- Periodic Mood Check System -------------

class MoodCheckScheduler:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self, user_id):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._scheduler_loop, args=(user_id,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _scheduler_loop(self, user_id):
        while self.running:
            try:
                settings = get_user_settings(user_id)
                if settings['popup_enabled']:
                    self._check_and_trigger_mood_check(user_id, settings)
                time.sleep(60)
            except Exception as e:
                print(f"Error in scheduler loop: {e}")
                time.sleep(60)

    def _check_and_trigger_mood_check(self, user_id, settings):
        if not settings.get('last_mood_check'):
            settings['last_mood_check'] = datetime.utcnow().isoformat()
            save_user_settings(settings)
            return

        last_check = datetime.fromisoformat(settings['last_mood_check'])
        interval_minutes = MOOD_CHECK_INTERVALS.get(settings['mood_check_interval'], 60)

        if datetime.utcnow() - last_check > timedelta(minutes=interval_minutes):
            if settings['system_notifications']:
                show_popup_reminder(user_id)
            settings['last_mood_check'] = datetime.utcnow().isoformat()
            save_user_settings(settings)

mood_scheduler = MoodCheckScheduler()

def simulate_fitbit():
    hr = random.randint(58, 110)
    hrv = max(5, random.gauss(50 - (hr - 60)*0.4, 10))
    sleep = round(random.gauss(7, 1.2), 1)
    steps = random.randint(0, 12000)
    stress = round(min(1, max(0, (120-hr)/200 + random.random()*0.3)), 2)
    return {'source': 'fitbit', 'heart_rate': int(hr), 'hrv': float(hrv), 'sleep_hours': float(sleep), 'steps': int(steps), 'stress_index': float(stress)}


def simulate_googlefit():
    hr = random.randint(55, 100)
    hrv = max(5, random.gauss(45, 12))
    sleep = round(random.gauss(6.5, 1.8), 1)
    steps = random.randint(0, 10000)
    stress = round(random.random(), 2)
    return {'source': 'googlefit', 'heart_rate': int(hr), 'hrv': float(hrv), 'sleep_hours': float(sleep), 'steps': int(steps), 'stress_index': float(stress)}


def simulate_apple():
    hr = random.randint(52, 98)
    hrv = max(5, random.gauss(48, 11))
    sleep = round(random.gauss(7.1, 1.0), 1)
    steps = random.randint(0, 9000)
    stress = round(max(0, (100-hr)/160 + random.random()*0.2), 2)
    return {'source': 'apple', 'heart_rate': int(hr), 'hrv': float(hrv), 'sleep_hours': float(sleep), 'steps': int(steps), 'stress_index': float(stress)}


def fetch_wearable(user_id, source='fitbit'):
    if SIMULATION_MODE:
        if source == 'fitbit':
            return simulate_fitbit()
        if source == 'googlefit':
            return simulate_googlefit()
        if source == 'apple':
            return simulate_apple()
        return simulate_fitbit()
    raise NotImplementedError('Real connectors not implemented in prototype')


def save_physio(user_id, sample):
    c = conn.cursor()
    sid = str(uuid.uuid4())
    c.execute('INSERT INTO physio VALUES (?,?,?,?,?,?,?,?,?)', (sid, user_id, datetime.utcnow().isoformat(), sample.get('source'), sample.get('heart_rate'), sample.get('hrv'), sample.get('sleep_hours'), sample.get('steps'), sample.get('stress_index')))
    conn.commit()


def recent_physio_summary(user_id, minutes=180):
    since = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    c = conn.cursor()
    c.execute('SELECT heart_rate, hrv, sleep_hours, steps, stress_index FROM physio WHERE user_id = ? AND timestamp >= ?', (user_id, since))
    rows = c.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['heart_rate','hrv','sleep_hours','steps','stress_index'])
    return df.mean().to_dict()


def compose_feedback(user_id, emoji, text, voice=False, thresholds=None):
    mood_score = EMOJI_LABELS.get(emoji, (0, ''))[0]
    sentiment = analyze_sentiment(text)
    phys = recent_physio_summary(user_id)
    alerts = []
    suggestions = []

    if mood_score >= 1:
        suggestions.append(random.choice(POSITIVE_FEEDBACK))
    elif mood_score == 0:
        suggestions.append(random.choice(NEUTRAL_FEEDBACK))
    else:
        suggestions.append(random.choice(ENCOURAGEMENT_FEEDBACK))

    if mood_score <= -2:
        alerts.append('mood_very_low')
        suggestions.append('Seu humor está muito baixo — considere falar com alguém de confiança.')
        create_reminder(user_id, 'urgent_care', '💙 Verifique como está se sentindo. Sua saúde mental importa!')
    elif mood_score == -1:
        suggestions.append('Parece um dia difícil. Faça uma pausa curta e respire profundamente.')
    elif mood_score >= 1:
        suggestions.append('Continue com o que tem funcionado!')

    th = thresholds or {'hr': ALERT_HR, 'hrv': ALERT_HRV}
    if phys:
        hr = phys.get('heart_rate', 0)
        hrv = phys.get('hrv', 100)
        sleep_h = phys.get('sleep_hours', 7)
        steps = phys.get('steps', 0)
        stress_idx = phys.get('stress_index', 0)

        if hr >= th['hr'] and mood_score <= -1:
            alerts.append('high_hr')
            suggestions.append('Freq. cardíaca elevada sem atividade. Sente-se e respire por 2 minutos.')
            create_reminder(user_id, 'high_hr', '💓 Frequência cardíaca elevada. Respire profundamente.')
        if hrv <= th['hrv'] and stress_idx > 0.5:
            alerts.append('low_hrv')
            suggestions.append('HRV baixo detectado — experimente relaxamento guiado por 3 minutos.')
        if sleep_h < 5.5 and mood_score <= 0:
            suggestions.append('Sono insuficiente detectado. Priorize descanso hoje se possível.')
        if steps < 500:
            suggestions.append('Pouca atividade hoje — uma breve caminhada de 5 minutos pode ajudar.')

    # sentiment nuance: if sentiment low but emoji positive
    if sentiment < 0.35 and mood_score >= 0:
        suggestions.append('Apesar do emoji, o texto sugere desconforto — obrigado por compartilhar.')
    if sentiment < 0.25 and mood_score <= -1:
        alerts.append('text_negative')

    if random.random() > 0.3:
        suggestions.append(random.choice(WELLNESS_TIPS))

    feedback = ' '.join(suggestions[:5]) if suggestions else 'Obrigado por compartilhar.'

    audio_path = None
    if voice and GTTs_AVAILABLE:
        try:
            tts = gTTS(feedback, lang='pt-br')
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(tmp.name)
            audio_path = tmp.name
        except Exception:
            audio_path = None

    return {
        'mood_score': mood_score,
        'sentiment': sentiment,
        'physio': phys,
        'alerts': alerts,
        'feedback': feedback,
        'audio': audio_path
    }


def show_reminders(user_id):
    if not STREAMLIT_AVAILABLE:
        return

    reminders = get_pending_reminders(user_id)

    for reminder_id, reminder_type, message in reminders:
        if 'urgent' in reminder_type or 'high_hr' in reminder_type or 'mood_very_low' in reminder_type:
            with st.container():
                st.markdown(f"<div style='padding:10px;border-radius:8px;background:#ffefef;border:1px solid #ffcccc'>🚨 <strong>{message}</strong></div>", unsafe_allow_html=True)
                if st.button("Entendi", key=f"dismiss_{reminder_id}"):
                    dismiss_reminder(reminder_id)
                    st.experimental_rerun()
        else:
            with st.container():
                st.markdown(f"<div style='padding:8px;border-radius:8px;background:#f7fbff;border:1px solid #dbeafe'>💡 {message}</div>", unsafe_allow_html=True)
                if st.button("OK", key=f"dismiss_{reminder_id}"):
                    dismiss_reminder(reminder_id)
                    st.experimental_rerun()


def export_all_data_csv(user_id):
    c = conn.cursor()
    c.execute('SELECT id, timestamp, question, emoji, text, sentiment, mood_score FROM responses WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
    responses = c.fetchall()
    df_resp = pd.DataFrame(responses, columns=['resp_id','timestamp','question','emoji','text','sentiment','mood_score']) if responses else pd.DataFrame(columns=['resp_id','timestamp','question','emoji','text','sentiment','mood_score'])

    c.execute('SELECT id, timestamp, source, heart_rate, hrv, sleep_hours, steps, stress_index FROM physio WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
    phys = c.fetchall()
    df_phys = pd.DataFrame(phys, columns=['phys_id','timestamp','source','heart_rate','hrv','sleep_hours','steps','stress_index']) if phys else pd.DataFrame(columns=['phys_id','timestamp','source','heart_rate','hrv','sleep_hours','steps','stress_index'])

    if not df_resp.empty:
        df_resp['timestamp'] = pd.to_datetime(df_resp['timestamp'])
    if not df_phys.empty:
        df_phys['timestamp'] = pd.to_datetime(df_phys['timestamp'])

    avg_mood = None
    if not df_resp.empty:
        avg_mood = df_resp['mood_score'].mean()

    output = io.StringIO()
    output.write('# Responses\n')
    df_resp.to_csv(output, index=False)
    output.write('\n# Physio samples\n')
    df_phys.to_csv(output, index=False)
    output.seek(0)

    csv_bytes = output.getvalue().encode('utf-8')
    meta = {
        'average_mood': avg_mood,
        'responses_count': len(df_resp),
        'physio_count': len(df_phys)
    }
    return csv_bytes, meta, df_resp, df_phys


def export_all_data_json(user_id):
    c = conn.cursor()
    c.execute('SELECT id, timestamp, question, emoji, text, sentiment, mood_score FROM responses WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
    responses = c.fetchall()
    resp_list = []
    for r in responses:
        resp_list.append({
            'id': r[0], 'timestamp': r[1], 'question': r[2], 'emoji': r[3], 'text': r[4], 'sentiment': r[5], 'mood_score': r[6]
        })

    c.execute('SELECT id, timestamp, source, heart_rate, hrv, sleep_hours, steps, stress_index FROM physio WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
    phys = c.fetchall()
    phys_list = []
    for p in phys:
        phys_list.append({
            'id': p[0], 'timestamp': p[1], 'source': p[2], 'heart_rate': p[3], 'hrv': p[4], 'sleep_hours': p[5], 'steps': p[6], 'stress_index': p[7]
        })

    meta = {
        'generated_at': datetime.utcnow().isoformat(),
        'responses_count': len(resp_list),
        'physio_count': len(phys_list)
    }

    payload = {
        'meta': meta,
        'responses': resp_list,
        'physio': phys_list
    }

    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
    return json_bytes, meta

# ------------- Streamlit UI (single-file) -------------

def run_streamlit():
    st.set_page_config(page_title='MindCheck Single — Improved', layout='wide', initial_sidebar_state='expanded')

    if 'user_id' not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID
    if 'last_check' not in st.session_state:
        st.session_state.last_check = None
    if 'show_celebration' not in st.session_state:
        st.session_state.show_celebration = False
    if 'show_quick_check' not in st.session_state:
        st.session_state.show_quick_check = False
    if 'scheduler_started' not in st.session_state:
        st.session_state.scheduler_started = False

    user_id = st.session_state.user_id

    if not st.session_state.scheduler_started:
        mood_scheduler.start(user_id)
        st.session_state.scheduler_started = True

    # Top-level header + compact controls
    header_col1, header_col2 = st.columns([3,1])
    with header_col1:
        st.title('🧠 MindCheck — Monitor de Bem-Estar')
        st.caption('Privado, anônimo e focado no seu bem-estar.')
    with header_col2:
        if st.button('🎯 Check-in Rápido'):
            st.session_state.show_quick_check = True
            st.experimental_rerun()

    show_reminders(user_id)
    quick_mood_check_modal_streamlit()

    # Auto-check for new reminders (every 30 minutes)
    if st.session_state.last_check is None or (datetime.now() - st.session_state.last_check).seconds > 1800:
        needed_reminders = check_reminders_needed(user_id)
        for reminder_type, message in needed_reminders:
            create_reminder(user_id, reminder_type, message)
        st.session_state.last_check = datetime.now()

    # Sidebar
    with st.sidebar:
        st.header('Conta')
        user_id = st.text_input('Seu ID anônimo', value=user_id)
        st.session_state.user_id = user_id
        st.markdown('---')
        st.header('Ações')
        if st.button('Coletar amostra sim. (Fitbit)', use_container_width=True):
            with st.spinner('Coletando dados...'):
                s = fetch_wearable(user_id, 'fitbit')
                save_physio(user_id, s)
                st.success(f"✅ Amostra salva HR={s['heart_rate']} HRV={s['hrv']:.1f}")

        if st.button('Simular dia completo', use_container_width=True):
            progress_bar = st.progress(0)
            for i in range(8):
                s = fetch_wearable(user_id, 'fitbit')
                save_physio(user_id, s)
                progress_bar.progress((i + 1) / 8)
                time.sleep(0.25)
            st.success('📈 Simulação concluída!')

        if st.button('Testar Notificação', use_container_width=True):
            show_popup_reminder(user_id)
            st.success('🔔 Notificação enviada!')

        st.markdown('---')
        st.header('Exportar')
        csv_bytes, meta_csv, df_resp, df_phys = export_all_data_csv(user_id)
        st.download_button('📥 Exportar CSV', data=csv_bytes, file_name=f'mindcheck_export_{user_id}.csv', mime='text/csv')
        json_bytes, meta_json = export_all_data_json(user_id)
        st.download_button('📥 Exportar JSON', data=json_bytes, file_name=f'mindcheck_export_{user_id}.json', mime='application/json')

    # Main tabs with friendly cards
    tabs = st.tabs(['Visão Geral','Check de Humor','Fisiologia','Histórico','Lembretes','Configurações'])

    # Visão Geral
    with tabs[0]:
        st.subheader('Sugestão do Dia')
        st.info(random.choice(WELLNESS_TIPS))

        st.subheader('Indicadores Rápidos')
        phys = recent_physio_summary(user_id) or fetch_wearable(user_id)
        cols = st.columns(4)
        cols[0].metric('❤️ Batimentos (bpm)', phys.get('heart_rate','—'))
        cols[1].metric('🧭 HRV (ms)', round(phys.get('hrv',0),1))
        cols[2].metric('😴 Sono (h)', phys.get('sleep_hours','—'))
        cols[3].metric('⚖️ Estresse', f"{phys.get('stress_index',0)*100:.0f}%")

    # Check de Humor
    with tabs[1]:
        st.header('Check de Humor')
        left, right = st.columns([1,2])
        with left:
            emoji = st.selectbox('Emoji', list(EMOJI_LABELS.keys()), key='main_emoji')
            if st.button('Enviar Check-in'):
                res = compose_feedback(user_id, emoji, st.session_state.get('detailed_text',''))
                rid = str(uuid.uuid4())
                c = conn.cursor()
                c.execute('INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)', (rid, user_id, datetime.utcnow().isoformat(), 'check', emoji, st.session_state.get('detailed_text',''), res['sentiment'], res['mood_score']))
                conn.commit()
                settings = get_user_settings(user_id)
                settings['last_mood_check'] = datetime.utcnow().isoformat()
                save_user_settings(settings)
                if res['mood_score'] >= 1:
                    st.balloons()
                    st.success('🌈 Ótimo humor registrado!')
                elif res['mood_score'] <= -1:
                    st.warning('💙 Obrigado por compartilhar.')
                else:
                    st.info('🤝 Check-in registrado.')
                st.markdown('### Feedback')
                st.write(res['feedback'])
                if res['audio']:
                    st.audio(res['audio'])
        with right:
            st.text_area('Descreva como se sente (opcional)', key='detailed_text', height=200)

    # Fisiologia
    with tabs[2]:
        st.header('Fisiologia (simulada)')
        source = st.selectbox('Fonte simulada', ['fitbit','googlefit','apple'])
        col1, col2 = st.columns(2)
        if col1.button('Atualizar dados fisiológicos'):
            with st.spinner('Coletando dados...'):
                s = fetch_wearable(user_id, source)
                save_physio(user_id, s)
                st.success('✅ Dados atualizados!')
        if col2.button('Simular 8 amostras'):
            progress = st.progress(0)
            for i in range(8):
                s = fetch_wearable(user_id, source)
                save_physio(user_id, s)
                progress.progress((i+1)/8)
                time.sleep(0.2)
            st.success('Simulação concluída!')

        data = recent_physio_summary(user_id) or fetch_wearable(user_id, source)
        st.json(data)

    # Histórico
    with tabs[3]:
        st.header('Histórico e Progresso')
        c = conn.cursor()
        c.execute('SELECT timestamp, emoji, mood_score, text FROM responses WHERE user_id = ? ORDER BY timestamp DESC LIMIT 200', (user_id,))
        rows = c.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['timestamp','emoji','mood_score','text'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.subheader('Evolução do Humor')
            st.line_chart(df.set_index('timestamp')['mood_score'])
            avg = df['mood_score'].mean()
            st.metric('Média do Humor (últimos registros)', f"{avg:.2f}")

            st.subheader('Registros Recentes')
            for _, row in df.head(12).iterrows():
                with st.expander(f"{row['timestamp'].strftime('%d/%m %H:%M')} - {row['emoji']} (Nota: {row['mood_score']})"):
                    if row['text']:
                        st.write(f"**Comentário:** {row['text']}")
                    else:
                        st.write('Sem comentário adicional')

            csv_bytes, meta, _, _ = export_all_data_csv(user_id)
            json_bytes, meta_json = export_all_data_json(user_id)
            st.download_button('📥 Exportar CSV', data=csv_bytes, file_name=f'mindcheck_export_{user_id}.csv', mime='text/csv')
            st.download_button('📥 Exportar JSON', data=json_bytes, file_name=f'mindcheck_export_{user_id}.json', mime='application/json')
        else:
            st.info('📝 Sem registros ainda — faça seu primeiro check-in!')

    # Lembretes
    with tabs[4]:
        st.header('Gestão de Lembretes')
        reminders = get_pending_reminders(user_id)
        if reminders:
            for reminder_id, reminder_type, message in reminders:
                st.write(f"🔔 {message}")
                if st.button('Dispensar', key=f'manage_{reminder_id}'):
                    dismiss_reminder(reminder_id)
                    st.experimental_rerun()
        else:
            st.success('🎉 Nenhum lembrete pendente!')

        st.subheader('Criar Lembrete')
        reminder_text = st.text_input('Mensagem do lembrete')
        if st.button('Agendar Lembrete') and reminder_text:
            create_reminder(user_id, 'personal', reminder_text)
            st.success('✅ Lembrete agendado!')

    # Configurações
    with tabs[5]:
        st.header('Configurações')
        settings = get_user_settings(user_id)
        popup_enabled = st.checkbox('Ativar verificações periódicas', value=settings['popup_enabled'])
        interval = st.selectbox('Frequência', list(MOOD_CHECK_INTERVALS.keys()), index=list(MOOD_CHECK_INTERVALS.keys()).index(settings['mood_check_interval']))
        system_notifications = st.checkbox('Notificações do sistema', value=settings['system_notifications'])

        if settings.get('last_mood_check'):
            last_check = datetime.fromisoformat(settings['last_mood_check'])
            next_check = last_check + timedelta(minutes=MOOD_CHECK_INTERVALS[interval])
            st.write(f"**Próxima verificação:** {next_check.strftime('%d/%m %H:%M')}")
        else:
            st.write('**Próxima verificação:** Agora')

        if st.button('Salvar Configurações'):
            settings['popup_enabled'] = popup_enabled
            settings['mood_check_interval'] = interval
            settings['system_notifications'] = system_notifications
            save_user_settings(settings)
            st.success('✅ Configurações salvas!')

        st.subheader('Limiar de alertas')
        alert_hr = st.slider('Limiar HR', 80, 140, ALERT_HR)
        alert_hrv = st.slider('Limiar HRV', 5, 50, ALERT_HRV)

        if st.button('Reiniciar Agendador'):
            mood_scheduler.stop()
            mood_scheduler.start(user_id)
            st.success('🔄 Agendador reiniciado!')

# Streamlit quick modal (adapted)

def quick_mood_check_modal_streamlit():
    if STREAMLIT_AVAILABLE and st.session_state.get('show_quick_check'):
        with st.container():
            st.markdown('---')
            st.subheader('🎯 Check-in Rápido')
            st.write('Como você está se sentindo agora?')
            c1, c2, c3 = st.columns(3)
            if c1.button('😊 Bem'):
                _process_quick_mood_streamlit('😊', 'Estou me sentindo bem')
                st.session_state.show_quick_check = False
                st.experimental_rerun()
            if c2.button('😐 Normal'):
                _process_quick_mood_streamlit('😐', 'Estou me sentindo normal')
                st.session_state.show_quick_check = False
                st.experimental_rerun()
            if c3.button('😔 Mal'):
                _process_quick_mood_streamlit('😔', 'Estou me sentindo mal')
                st.session_state.show_quick_check = False
                st.experimental_rerun()
            if st.button('Fechar'):
                st.session_state.show_quick_check = False
                st.experimental_rerun()


def _process_quick_mood_streamlit(emoji, default_text):
    user_id = st.session_state.user_id
    res = compose_feedback(user_id, emoji, default_text)
    rid = str(uuid.uuid4())
    c = conn.cursor()
    c.execute('INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)', (rid, user_id, datetime.utcnow().isoformat(), 'quick_modal', emoji, default_text, res['sentiment'], res['mood_score']))
    conn.commit()
    settings = get_user_settings(user_id)
    settings['last_mood_check'] = datetime.utcnow().isoformat()
    save_user_settings(settings)
    st.success('✅ Check-in registrado!')

# ------------- CLI fallback (with JSON export) -------------

def run_cli():
    print('Modo CLI: Streamlit não está disponível.')
    uid = input(f'Digite seu ID anônimo (ENTER para {DEFAULT_USER_ID}): ').strip() or DEFAULT_USER_ID
    mood_scheduler.start(uid)

    while True:
        print('\nMenu:')
        print('1) Coletar amostra simulada')
        print('2) Enviar check de humor')
        print('3) Mostrar resumo')
        print('4) Verificar lembretes')
        print('5) Forçar verificação de humor')
        print('6) Exportar todos os dados para CSV')
        print('7) Exportar todos os dados para JSON')
        print('0) Sair')
        c = input('Escolha: ').strip()

        if c == '1':
            src = input('Fonte [fitbit/googlefit/apple]: ').strip() or 'fitbit'
            s = fetch_wearable(uid, src)
            save_physio(uid, s)
            print('✅ Amostra salva:', s)
        elif c == '2':
            print('Emojis:', ' '.join(EMOJI_LABELS.keys()))
            e = input('Emoji: ').strip() or '😐'
            txt = input('Texto (opcional): ').strip()
            res = compose_feedback(uid, e, txt)
            rid = str(uuid.uuid4())
            conn.cursor().execute('INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)', (rid, uid, datetime.utcnow().isoformat(), 'cli', e, txt, res['sentiment'], res['mood_score']))
            conn.commit()
            print('💬 Feedback:', res['feedback'])
            if res['alerts']:
                print('🚨 ALERTAS:', res['alerts'])
        elif c == '3':
            phys = recent_physio_summary(uid) or {}
            print('📊 Resumo physio:', phys)
            cur = conn.cursor()
            cur.execute('SELECT timestamp, emoji, mood_score FROM responses WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (uid,))
            for r in cur.fetchall():
                print(f"📝 {r[0]}: {r[1]} (Nota: {r[2]})")
        elif c == '4':
            reminders = get_pending_reminders(uid)
            if reminders:
                print('🔔 Lembretes pendentes:')
                for rem_id, rem_type, message in reminders:
                    print(f" - {message}")
                    dismiss = input('Dispensar? (s/n): ').strip().lower()
                    if dismiss == 's':
                        dismiss_reminder(rem_id)
                        print('✅ Lembrete removido!')
            else:
                print('🎉 Nenhum lembrete pendente!')
        elif c == '5':
            show_popup_reminder(uid)
            print('🔔 Verificação de humor acionada!')
        elif c == '6':
            csv_bytes, meta, _, _ = export_all_data_csv(uid)
            out_name = f'mindcheck_export_{uid}.csv'
            with open(out_name, 'wb') as f:
                f.write(csv_bytes)
            print(f'✅ Exportado {out_name} | respostas: {meta["responses_count"]} | physio: {meta["physio_count"]}')
        elif c == '7':
            json_bytes, meta_json = export_all_data_json(uid)
            out_name = f'mindcheck_export_{uid}.json'
            with open(out_name, 'wb') as f:
                f.write(json_bytes)
            print(f'✅ Exportado {out_name} | respostas: {meta_json["responses_count"]} | physio: {meta_json["physio_count"]}')
        elif c == '0':
            mood_scheduler.stop()
            break
        else:
            print('❌ Opção inválida')

# ------------- Entrypoint -------------
if __name__ == '__main__':
    try:
        if STREAMLIT_AVAILABLE:
            try:
                run_streamlit()
            except Exception as e:
                print('Erro ao iniciar Streamlit:', e)
                print('Caindo para modo CLI...')
                mood_scheduler.stop()
                run_cli()
        else:
            run_cli()
    except KeyboardInterrupt:
        print('\n👋 Encerrando MindCheck...')
        mood_scheduler.stop()
    finally:
        mood_scheduler.stop()
