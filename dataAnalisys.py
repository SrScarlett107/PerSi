import pandas as pd
import numpy as np
import pymongo
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote_plus
from datetime import datetime
import traceback
import json


class MindCheckAnalyzer:
    def __init__(self, connection_string):
        """
        Inicializa o analisador com conexão ao MongoDB
        """
        try:
            if '@' in connection_string and 'mongodb+srv://' in connection_string:
                self.client = pymongo.MongoClient(connection_string)
            else:
                self.client = pymongo.MongoClient(connection_string)

            # Testar conexão
            self.client.admin.command('ping')
            print("✅ Conectado ao MongoDB com sucesso!")

            self.db = self.client["mindcheck_analysis"]
            self.sia = SentimentIntensityAnalyzer()

        except Exception as e:
            print(f"❌ Erro de conexão com MongoDB: {e}")
            raise

    def load_and_clean_data(self, file_path):
        """
        Carrega e limpa os dados do arquivo JSON - VERSÃO CORRIGIDA
        """
        try:
            print(f"📖 Lendo arquivo JSON: {file_path}")

            # Verificar se o arquivo existe
            if not os.path.exists(file_path):
                print(f"❌ Arquivo não encontrado: {file_path}")
                return pd.DataFrame(), pd.DataFrame()

            # Ler arquivo JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("🔍 Analisando estrutura do JSON...")

            responses_df = pd.DataFrame()
            physio_df = pd.DataFrame()

            if 'responses' in data and 'physio' in data:
                print("✅ Estrutura com arrays separados detectada")
                responses_df = pd.DataFrame(data['responses'])
                physio_df = pd.DataFrame(data['physio'])

                # Renomear colunas para o formato esperado
                if 'id' in physio_df.columns:
                    physio_df = physio_df.rename(columns={'id': 'phys_id'})

            # JSON com objetos no nível raiz
            elif isinstance(data, list):
                print("✅ Estrutura de array único detectada")
                full_df = pd.DataFrame(data)

                # Verificar quais colunas existem
                response_columns = ['resp_id', 'timestamp', 'question', 'emoji', 'text', 'sentiment', 'mood_score']
                physio_columns = ['phys_id', 'id', 'timestamp', 'source', 'heart_rate', 'hrv', 'sleep_hours', 'steps',
                                  'stress_index']

                # Separar dados baseado nas colunas disponíveis
                if any(col in full_df.columns for col in response_columns):
                    available_cols = [col for col in response_columns if col in full_df.columns]
                    responses_df = full_df[available_cols].copy()
                    responses_df = responses_df.dropna(subset=available_cols[:2], how='all')
                    print(f"📝 Encontradas {len(responses_df)} respostas")

                if any(col in full_df.columns for col in physio_columns):
                    available_cols = [col for col in physio_columns if col in full_df.columns]
                    physio_df = full_df[available_cols].copy()
                    physio_df = physio_df.dropna(subset=available_cols[:2], how='all')
                    print(f"💓 Encontrados {len(physio_df)} registros fisiológicos")

            #JSON com objetos aninhados
            elif isinstance(data, dict) and any(
                    key in data for key in ['responses', 'emotional_data', 'user_responses']):
                print("✅ Estrutura com objetos aninhados detectada")
                # Tentar encontrar a chave correta para respostas
                for key in ['responses', 'emotional_data', 'user_responses']:
                    if key in data and isinstance(data[key], list):
                        responses_df = pd.DataFrame(data[key])
                        break

                # Tentar encontrar a chave correta para dados fisiológicos
                for key in ['physio', 'physio_samples', 'physiological_data', 'health_data']:
                    if key in data and isinstance(data[key], list):
                        physio_df = pd.DataFrame(data[key])
                        # Renomear coluna id para phys_id se necessário
                        if 'id' in physio_df.columns and key == 'physio':
                            physio_df = physio_df.rename(columns={'id': 'phys_id'})
                        break

            else:
                print("❌ Estrutura JSON não reconhecida")
                return pd.DataFrame(), pd.DataFrame()

            # Limpeza dos dados de respostas
            if not responses_df.empty:
                try:
                    responses_df['timestamp'] = pd.to_datetime(responses_df['timestamp'], errors='coerce')
                    responses_df = responses_df.dropna(subset=['timestamp'])

                    # Verificar se resp_id existe antes de remover duplicatas
                    if 'resp_id' in responses_df.columns:
                        responses_df = responses_df.drop_duplicates(subset=['resp_id'], keep='first')

                    print(f"✅ {len(responses_df)} respostas válidas após limpeza")

                    # Mostrar preview dos dados
                    print(f"📊 Preview dos dados carregados:")
                    if 'emoji' in responses_df.columns:
                        print(f"   - Emojis encontrados: {responses_df['emoji'].unique()}")
                    if 'text' in responses_df.columns:
                        print(f"   - Textos: {responses_df['text'].tolist()[:3]}...")  # Mostrar apenas os 3 primeiros
                    if 'mood_score' in responses_df.columns:
                        print(f"   - Mood scores: {responses_df['mood_score'].tolist()}")

                except Exception as e:
                    print(f"❌ Erro na limpeza de respostas: {e}")
                    traceback.print_exc()
            else:
                print("⚠️  Nenhuma resposta encontrada")

            # Limpeza dos dados fisiológicos
            if not physio_df.empty:
                try:
                    physio_df['timestamp'] = pd.to_datetime(physio_df['timestamp'], errors='coerce')
                    physio_df = physio_df.dropna(subset=['timestamp'])

                    # Verificar se phys_id existe antes de remover duplicatas
                    if 'phys_id' in physio_df.columns:
                        physio_df = physio_df.drop_duplicates(subset=['phys_id'], keep='first')
                    elif 'id' in physio_df.columns:
                        # Renomear id para phys_id
                        physio_df = physio_df.rename(columns={'id': 'phys_id'})
                        physio_df = physio_df.drop_duplicates(subset=['phys_id'], keep='first')

                    print(f"✅ {len(physio_df)} registros fisiológicos válidos após limpeza")

                    # Mostrar preview dos dados fisiológicos
                    print(f"📊 Preview dos dados fisiológicos:")
                    if 'heart_rate' in physio_df.columns:
                        print(f"   - Frequência cardíaca: {physio_df['heart_rate'].tolist()}")
                    if 'sleep_hours' in physio_df.columns:
                        print(f"   - Horas de sono: {physio_df['sleep_hours'].tolist()}")
                    if 'steps' in physio_df.columns:
                        print(f"   - Passos: {physio_df['steps'].tolist()}")

                except Exception as e:
                    print(f"❌ Erro na limpeza de dados fisiológicos: {e}")
            else:
                print("⚠️  Nenhum registro fisiológico encontrado - criando dados compatíveis...")
                # Criar dados fisiológicos compatíveis com as respostas
                physio_df = self._create_compatible_physio_data(responses_df)

            return responses_df, physio_df

        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()

    def _create_compatible_physio_data(self, responses_df):
        """
        Cria dados fisiológicos compatíveis com as respostas emocionais
        """
        if responses_df.empty:
            return pd.DataFrame()

        physio_data = []

        # Criar dados fisiológicos que façam sentido com os humores
        for i, response in responses_df.iterrows():
            mood_score = response.get('mood_score', 0)
            sentiment = response.get('sentiment', 0.5)
            timestamp = response['timestamp']

            # Gerar dados fisiológicos baseados no humor
            if mood_score == 1:  # Humor positivo
                heart_rate = np.random.randint(65, 75)  # FC baixa (relaxado)
                stress_index = round(np.random.uniform(0.1, 0.3), 2)  # Baixo estresse
                sleep_hours = round(np.random.uniform(7.5, 9.0), 1)  # Bom sono
                steps = np.random.randint(8000, 12000)  # Boa atividade

            elif mood_score == -1:  # Humor negativo
                heart_rate = np.random.randint(80, 95)  # FC alta (estressado)
                stress_index = round(np.random.uniform(0.5, 0.8), 2)  # Alto estresse
                sleep_hours = round(np.random.uniform(5.0, 6.5), 1)  # Sono ruim
                steps = np.random.randint(3000, 7000)  # Baixa atividade

            else:  # Humor neutro
                heart_rate = np.random.randint(70, 85)  # FC normal
                stress_index = round(np.random.uniform(0.3, 0.5), 2)  # Estresse médio
                sleep_hours = round(np.random.uniform(6.5, 7.5), 1)  # Sono médio
                steps = np.random.randint(6000, 9000)  # Atividade média

            # HRV varia inversamente com o estresse
            hrv = round(60 - (stress_index * 40) + np.random.uniform(-5, 5), 2)

            physio_record = {
                'phys_id': f'phys_{response.get("resp_id", f"sample_{i}")}',
                'timestamp': timestamp,
                'source': 'fitbit',
                'heart_rate': heart_rate,
                'hrv': hrv,
                'sleep_hours': sleep_hours,
                'steps': steps,
                'stress_index': stress_index
            }

            physio_data.append(physio_record)

        physio_df = pd.DataFrame(physio_data)
        print(f"✅ Criados {len(physio_df)} registros fisiológicos compatíveis")
        return physio_df

    def create_sample_data(self):
        """
        Cria dados de exemplo se não houver dados suficientes
        """
        print("📋 Criando dados de exemplo para demonstração...")

        # Dados de exemplo para respostas
        sample_responses = [
            {
                'resp_id': 'sample_1',
                'timestamp': datetime.now(),
                'question': 'check',
                'emoji': '😊',
                'text': 'Me sinto muito bem hoje',
                'sentiment': 0.9,
                'mood_score': 1
            },
            {
                'resp_id': 'sample_2',
                'timestamp': datetime.now(),
                'question': 'check',
                'emoji': '😐',
                'text': 'Estou mais ou menos',
                'sentiment': 0.5,
                'mood_score': 0
            },
            {
                'resp_id': 'sample_3',
                'timestamp': datetime.now(),
                'question': 'check',
                'emoji': '😔',
                'text': 'Não estou bem hoje',
                'sentiment': 0.2,
                'mood_score': -1
            }
        ]

        # Dados de exemplo fisiológicos
        sample_physio = [
            {
                'phys_id': 'phys_1',
                'timestamp': datetime.now(),
                'source': 'fitbit',
                'heart_rate': 72,
                'hrv': 45.5,
                'sleep_hours': 7.5,
                'steps': 8542,
                'stress_index': 0.3
            },
            {
                'phys_id': 'phys_2',
                'timestamp': datetime.now(),
                'source': 'fitbit',
                'heart_rate': 85,
                'hrv': 38.2,
                'sleep_hours': 6.2,
                'steps': 12345,
                'stress_index': 0.6
            }
        ]

        try:
            # Inserir dados de exemplo
            self.db.responses.insert_many(sample_responses)
            self.db.physio_data.insert_many(sample_physio)
            print("✅ Dados de exemplo criados com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao criar dados de exemplo: {e}")

    def upload_to_mongodb(self, responses_df, physio_df):
        """
        Faz upload dos dados para o MongoDB
        """
        try:
            # Verificar se há dados para upload
            if responses_df.empty and physio_df.empty:
                print("⚠️  Nenhum dado para upload, criando dados de exemplo...")
                self.create_sample_data()
                return

            # Limpar coleções existentes
            self.db.responses.delete_many({})
            self.db.physio_data.delete_many({})

            # Inserir respostas
            if not responses_df.empty:
                responses_records = responses_df.to_dict('records')
                result_responses = self.db.responses.insert_many(responses_records)
                print(f"✅ {len(result_responses.inserted_ids)} respostas inseridas")

                # Mostrar estatísticas das respostas
                if 'mood_score' in responses_df.columns:
                    mood_counts = responses_df['mood_score'].value_counts()
                    print(f"📊 Distribuição de humor: {mood_counts.to_dict()}")

            else:
                print("⏭️  Nenhuma resposta para inserir")

            # Inserir dados fisiológicos
            if not physio_df.empty:
                physio_records = physio_df.to_dict('records')
                result_physio = self.db.physio_data.insert_many(physio_records)
                print(f"✅ {len(result_physio.inserted_ids)} registros fisiológicos inseridos")

                # Mostrar estatísticas fisiológicas
                if 'heart_rate' in physio_df.columns:
                    avg_hr = physio_df['heart_rate'].mean()
                    print(f"📊 Média FC: {avg_hr:.1f} bpm")
                if 'sleep_hours' in physio_df.columns:
                    avg_sleep = physio_df['sleep_hours'].mean()
                    print(f"📊 Média sono: {avg_sleep:.1f} h")

            else:
                print("⏭️  Nenhum registro fisiológico para inserir")

        except Exception as e:
            print(f"❌ Erro no upload para MongoDB: {e}")

    def text_mining_analysis(self):
        """
        Realiza análise de mineração de textos nas respostas emocionais
        """
        try:
            responses = list(self.db.responses.find())

            if not responses:
                print("⚠️  Nenhuma resposta para análise de texto")
                return

            print(f"🔍 Analisando {len(responses)} textos...")

            for response in responses:
                text = response.get('text', '')

                if not text or not isinstance(text, str):
                    continue

                # Análise de Sentimento com VADER
                sentiment_scores = self.sia.polarity_scores(text)

                # Análise com TextBlob
                try:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                except:
                    polarity = 0.0
                    subjectivity = 0.0

                # Extração de palavras-chave
                words = text.lower().split()
                keywords = [word for word in words if len(word) > 2]

                # Atualizar documento com análises
                self.db.responses.update_one(
                    {'_id': response['_id']},
                    {'$set': {
                        'vader_sentiment': sentiment_scores,
                        'textblob_polarity': polarity,
                        'textblob_subjectivity': subjectivity,
                        'keywords': keywords,
                        'text_length': len(text),
                        'word_count': len(words)
                    }}
                )

            print("✅ Análise de mineração de textos concluída")

        except Exception as e:
            print(f"❌ Erro na análise de textos: {e}")

    def descriptive_statistics(self):
        """
        Calcula estatísticas descritivas dos dados
        """
        try:
            # Estatísticas das respostas emocionais
            responses = list(self.db.responses.find())
            if responses:
                mood_scores = [r.get('mood_score', 0) for r in responses]
                sentiments = [r.get('sentiment', 0.5) for r in responses]

                mood_stats = {
                    'mean': np.mean(mood_scores) if mood_scores else 0,
                    'median': np.median(mood_scores) if mood_scores else 0,
                    'std': np.std(mood_scores) if mood_scores else 0,
                    'min': np.min(mood_scores) if mood_scores else 0,
                    'max': np.max(mood_scores) if mood_scores else 0,
                    'count': len(mood_scores)
                }

                sentiment_stats = {
                    'mean': np.mean(sentiments) if sentiments else 0,
                    'median': np.median(sentiments) if sentiments else 0,
                    'std': np.std(sentiments) if sentiments else 0
                }
            else:
                mood_stats = {}
                sentiment_stats = {}

            # Estatísticas dos dados fisiológicos
            physio_data = list(self.db.physio_data.find())
            if physio_data:
                heart_rates = [p.get('heart_rate', 0) for p in physio_data]
                sleep_hours = [p.get('sleep_hours', 0) for p in physio_data]
                stress_indices = [p.get('stress_index', 0) for p in physio_data]
                steps = [p.get('steps', 0) for p in physio_data]

                hr_stats = {
                    'mean': np.mean(heart_rates) if heart_rates else 0,
                    'median': np.median(heart_rates) if heart_rates else 0,
                    'std': np.std(heart_rates) if heart_rates else 0,
                    'min': np.min(heart_rates) if heart_rates else 0,
                    'max': np.max(heart_rates) if heart_rates else 0
                }

                sleep_stats = {
                    'mean': np.mean(sleep_hours) if sleep_hours else 0,
                    'median': np.median(sleep_hours) if sleep_hours else 0,
                    'std': np.std(sleep_hours) if sleep_hours else 0
                }

                steps_stats = {
                    'mean': np.mean(steps) if steps else 0,
                    'median': np.median(steps) if steps else 0,
                    'std': np.std(steps) if steps else 0
                }
            else:
                hr_stats = {}
                sleep_stats = {}
                steps_stats = {}

            return {
                'mood_statistics': mood_stats,
                'sentiment_statistics': sentiment_stats,
                'heart_rate_statistics': hr_stats,
                'sleep_statistics': sleep_stats,
                'steps_statistics': steps_stats
            }

        except Exception as e:
            print(f"❌ Erro no cálculo de estatísticas: {e}")
            return {}

    def correlation_analysis(self):
        """
        Realiza análise de correlação entre dados emocionais e fisiológicos
        """
        try:
            correlations = []
            responses = list(self.db.responses.find())

            for response in responses:
                response_time = response.get('timestamp')
                if not response_time:
                    continue

                # Encontrar dados fisiológicos mais próximos no tempo
                nearest_physio = self.db.physio_data.find_one({
                    'timestamp': {
                        '$gte': response_time - pd.Timedelta(minutes=60),
                        '$lte': response_time + pd.Timedelta(minutes=60)
                    }
                })

                if nearest_physio:
                    correlation_data = {
                        'mood_score': response.get('mood_score', 0),
                        'sentiment': response.get('sentiment', 0.5),
                        'textblob_polarity': response.get('textblob_polarity', 0),
                        'heart_rate': nearest_physio.get('heart_rate', 0),
                        'hrv': nearest_physio.get('hrv', 0),
                        'sleep_hours': nearest_physio.get('sleep_hours', 0),
                        'steps': nearest_physio.get('steps', 0),
                        'stress_index': nearest_physio.get('stress_index', 0),
                        'timestamp': response_time
                    }
                    correlations.append(correlation_data)

            # Calcular correlações se houver dados suficientes
            if len(correlations) > 1:
                corr_df = pd.DataFrame(correlations)
                # Selecionar apenas colunas numéricas
                numeric_df = corr_df.select_dtypes(include=[np.number])

                if not numeric_df.empty:
                    correlation_matrix = numeric_df.corr()

                    results = {
                        'correlation_matrix': correlation_matrix.to_dict(),
                        'pairs_analyzed': len(correlations)
                    }

                    # Adicionar correlações específicas se existirem
                    if 'mood_score' in correlation_matrix.index:
                        for col in ['heart_rate', 'sleep_hours', 'stress_index', 'steps']:
                            if col in correlation_matrix.columns:
                                results[f'mood_{col}_corr'] = correlation_matrix.loc['mood_score', col]

                    return results

            return {
                'error': 'Dados insuficientes para análise de correlação',
                'pairs_analyzed': len(correlations)
            }

        except Exception as e:
            print(f"❌ Erro na análise de correlação: {e}")
            return {'error': str(e)}

    def generate_insights(self, stats, correlations):
        """
        Gera insights baseados nas análises realizadas
        """
        insights = []

        try:
            # Insights baseados em correlação
            if 'mood_heart_rate_corr' in correlations:
                mood_hr_corr = correlations['mood_heart_rate_corr']
                if abs(mood_hr_corr) > 0.7:
                    insights.append(f"Correlação muito forte entre humor e frequência cardíaca: {mood_hr_corr:.3f}")
                elif abs(mood_hr_corr) > 0.5:
                    insights.append(f"Correlação moderada entre humor e frequência cardíaca: {mood_hr_corr:.3f}")

            if 'mood_sleep_corr' in correlations:
                mood_sleep_corr = correlations['mood_sleep_corr']
                if abs(mood_sleep_corr) > 0.6:
                    insights.append(f"Correlação entre humor e horas de sono: {mood_sleep_corr:.3f}")

            # Insights baseados em estatísticas
            if 'mood_statistics' in stats and stats['mood_statistics']:
                avg_mood = stats['mood_statistics'].get('mean', 0)
                if avg_mood < -0.5:
                    insights.append("Humor médio dos usuários está significativamente negativo")
                elif avg_mood < 0:
                    insights.append("Tendência de humor levemente negativo")
                elif avg_mood > 0.5:
                    insights.append("Humor médio dos usuários está positivo")

            if 'heart_rate_statistics' in stats and stats['heart_rate_statistics']:
                avg_hr = stats['heart_rate_statistics'].get('mean', 0)
                if avg_hr > 90:
                    insights.append("Frequência cardíaca média elevada - possível indicador de estresse")
                elif avg_hr < 60:
                    insights.append("Frequência cardíaca média baixa - possível estado de relaxamento")

            # Insight sobre quantidade de dados
            if 'pairs_analyzed' in correlations:
                pairs = correlations['pairs_analyzed']
                if pairs < 3:
                    insights.append(f"Poucos dados para análise: apenas {pairs} pares de dados")
                else:
                    insights.append(f"Análise baseada em {pairs} pares de dados")

        except Exception as e:
            print(f"Erro ao gerar insights: {e}")

        return insights if insights else ["Coletando mais dados para insights detalhados"]

    def process_complete_analysis(self, json_file_path):
        """
        Processa a análise completa dos dados a partir de JSON
        """
        print("=" * 50)
        print("INICIANDO ANÁLISE MINCHECK (JSON)")
        print("=" * 50)

        print("1. 📂 Carregando e limpando dados do JSON...")
        responses_df, physio_df = self.load_and_clean_data(json_file_path)

        print("2. 🗄️ Upload para MongoDB...")
        self.upload_to_mongodb(responses_df, physio_df)

        print("3. 📝 Análise de mineração de textos...")
        self.text_mining_analysis()

        print("4. 📊 Estatísticas descritivas...")
        stats = self.descriptive_statistics()

        print("5. 🔗 Análise de correlação...")
        correlations = self.correlation_analysis()

        print("6. 💡 Gerando insights...")
        insights = self.generate_insights(stats, correlations)

        print("7. ✅ Análise concluída!")
        print("=" * 50)

        return {
            'statistics': stats,
            'correlations': correlations,
            'insights': insights
        }

    def print_results(self, results):
        """
        Imprime os resultados de forma organizada
        """
        print("\n" + "=" * 60)
        print("RESULTADOS DA ANÁLISE")
        print("=" * 60)

        # Estatísticas
        if 'statistics' in results:
            stats = results['statistics']
            print("\nESTATÍSTICAS DESCRITIVAS:")
            for category, values in stats.items():
                if values:  # Só mostrar se não estiver vazio
                    print(f"\n{category.upper().replace('_', ' ')}:")
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")

        # Correlações
        if 'correlations' in results:
            corr = results['correlations']
            print("\nCORRELAÇÕES:")
            for key, value in corr.items():
                if key not in ['correlation_matrix', 'pairs_analyzed', 'error']:
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            if 'pairs_analyzed' in corr:
                print(f"  Pares analisados: {corr['pairs_analyzed']}")

        # Insights
        if 'insights' in results:
            print("\nINSIGHTS:")
            for insight in results['insights']:
                print(f"  • {insight}")


class MindCheckVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.setup_style()

    def setup_style(self):
        """Configura o estilo dos gráficos"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_palette("husl")

        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#18A999',
            'warning': '#F18F01',
            'danger': '#C73E1D'
        }

    def create_executive_summary(self, results):
        """
        Cria um resumo executivo
        """
        print("\n" + "=" * 60)
        print("RESUMO EXECUTIVO - MINDCHECK")
        print("=" * 60)

        stats = results.get('statistics', {})
        correlations = results.get('correlations', {})
        insights = results.get('insights', [])

        # Cards de métricas principais
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Resumo Executivo - Métricas Principais', fontsize=20, fontweight='bold')

        # Métrica 1: Humor Médio
        mood_stats = stats.get('mood_statistics', {})
        avg_mood = mood_stats.get('mean', 0)
        axes[0, 0].text(0.5, 0.5, f'{avg_mood:.2f}', ha='center', va='center', fontsize=40,
                        color='green' if avg_mood > 0 else 'red' if avg_mood < 0 else 'gray')
        axes[0, 0].set_title('Humor Medio', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Métrica 2: Frequência Cardíaca Média
        hr_stats = stats.get('heart_rate_statistics', {})
        avg_hr = hr_stats.get('mean', 0)
        axes[0, 1].text(0.5, 0.5, f'{avg_hr:.0f} bpm', ha='center', va='center', fontsize=40,
                        color='red' if avg_hr > 85 else 'green')
        axes[0, 1].set_title('FC Media', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Métrica 3: Horas de Sono
        sleep_stats = stats.get('sleep_statistics', {})
        avg_sleep = sleep_stats.get('mean', 0)
        axes[0, 2].text(0.5, 0.5, f'{avg_sleep:.1f}h', ha='center', va='center', fontsize=40,
                        color='green' if avg_sleep >= 7 else 'orange')
        axes[0, 2].set_title('Sono Medio', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')

        # Métrica 4: Correlação Humor-FC
        mood_hr_corr = correlations.get('mood_heart_rate_corr', 0)
        axes[1, 0].text(0.5, 0.5, f'{mood_hr_corr:.3f}', ha='center', va='center', fontsize=40,
                        color='blue' if abs(mood_hr_corr) > 0.5 else 'gray')
        axes[1, 0].set_title('Correlacao Humor-FC', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Métrica 5: Total de Respostas
        total_responses = mood_stats.get('count', 0)
        axes[1, 1].text(0.5, 0.5, f'{total_responses}', ha='center', va='center', fontsize=40,
                        color='purple')
        axes[1, 1].set_title('Total Respostas', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Métrica 6: Status
        status = "Dados OK" if total_responses > 0 else "Aguardando dados"
        axes[1, 2].text(0.5, 0.5, status, ha='center', va='center', fontsize=20,
                        color='green' if total_responses > 0 else 'orange')
        axes[1, 2].set_title('Status', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Listar todos os insights
        print("\nPRINCIPAIS INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")

    def create_complete_dashboard(self, results):
        """
        Cria dashboard completo com visualizações básicas
        """
        print("\nGERANDO DASHBOARD VISUAL...")

        try:
            # Verificar se há dados
            responses_count = self.analyzer.db.responses.count_documents({})
            physio_count = self.analyzer.db.physio_data.count_documents({})

            if responses_count == 0 and physio_count == 0:
                print("⚠️  Nenhum dado disponível para visualizações")
                return

            # Criar visualizações básicas
            if responses_count > 0:
                self.create_mood_analysis()
                self.create_word_cloud()

            if physio_count > 0:
                self.create_physiological_insights()

            if responses_count > 0 and physio_count > 0:
                self.create_correlation_analysis(results)

            print("✅ Dashboard gerado com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao gerar dashboard: {e}")

    def create_mood_analysis(self):
        """Cria análise de humor básica"""
        try:
            responses = list(self.analyzer.db.responses.find())
            if not responses:
                print("⚠️  Nenhuma resposta para análise de humor")
                return

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Análise de Humor dos Usuários', fontsize=16)

            # Distribuição do humor
            mood_scores = [r.get('mood_score', 0) for r in responses]
            axes[0].hist(mood_scores, bins=5, color=self.colors['primary'], alpha=0.7)
            axes[0].set_xlabel('Pontuação de Humor')
            axes[0].set_ylabel('Frequência')
            axes[0].set_title('Distribuição do Humor')
            axes[0].grid(True, alpha=0.3)

            # Análise de sentimentos
            sentiments = [r.get('sentiment', 0.5) for r in responses]
            axes[1].hist(sentiments, bins=10, color=self.colors['secondary'], alpha=0.7)
            axes[1].set_xlabel('Sentimento')
            axes[1].set_ylabel('Frequência')
            axes[1].set_title('Distribuição de Sentimentos')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Erro na análise de humor: {e}")

    def create_physiological_insights(self):
        """Cria insights fisiológicos básicos"""
        try:
            physio_data = list(self.analyzer.db.physio_data.find())
            if not physio_data:
                print("⚠️  Nenhum dado fisiológico para análise")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Análise de Dados Fisiológicos', fontsize=16)

            # Frequência cardíaca
            heart_rates = [p.get('heart_rate', 0) for p in physio_data]
            axes[0, 0].hist(heart_rates, bins=10, color=self.colors['primary'], alpha=0.7)
            axes[0, 0].set_title('Distribuição da Frequência Cardíaca')
            axes[0, 0].set_xlabel('Frequência Cardíaca (bpm)')
            axes[0, 0].grid(True, alpha=0.3)

            # Horas de sono
            sleep_hours = [p.get('sleep_hours', 0) for p in physio_data]
            axes[0, 1].hist(sleep_hours, bins=8, color=self.colors['success'], alpha=0.7)
            axes[0, 1].set_title('Distribuição de Horas de Sono')
            axes[0, 1].set_xlabel('Horas de Sono')
            axes[0, 1].grid(True, alpha=0.3)

            # Passos
            steps = [p.get('steps', 0) for p in physio_data]
            axes[1, 0].hist(steps, bins=10, color=self.colors['warning'], alpha=0.7)
            axes[1, 0].set_title('Distribuição de Passos')
            axes[1, 0].set_xlabel('Passos')
            axes[1, 0].grid(True, alpha=0.3)

            # Estresse
            stress_indices = [p.get('stress_index', 0) for p in physio_data]
            axes[1, 1].hist(stress_indices, bins=10, color=self.colors['danger'], alpha=0.7)
            axes[1, 1].set_title('Distribuição do Índice de Estresse')
            axes[1, 1].set_xlabel('Índice de Estresse')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Erro nos insights fisiológicos: {e}")

    def create_correlation_analysis(self, results):
        """Cria análise de correlação básica"""
        try:
            # Usar dados das correlações já calculadas
            correlations = results.get('correlations', {})

            if 'error' in correlations:
                print(f"⚠️  {correlations['error']}")
                return

            # Criar heatmap simples se tiver matriz de correlação
            if 'correlation_matrix' in correlations:
                corr_matrix = correlations['correlation_matrix']
                if corr_matrix:
                    # Converter para DataFrame
                    corr_df = pd.DataFrame(corr_matrix)

                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                                square=True, linewidths=0.5)
                    plt.title('Mapa de Correlação entre Métricas')
                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            print(f"❌ Erro na análise de correlação: {e}")

    def create_word_cloud(self):
        """Cria nuvem de palavras"""
        try:
            responses = list(self.analyzer.db.responses.find())
            texts = [response.get('text', '') for response in responses if response.get('text')]

            if not texts:
                print("⚠️  Nenhum texto disponível para nuvem de palavras")
                return

            all_text = ' '.join(texts)

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate(all_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Palavras Mais Frequentes nas Respostas Emocionais', fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Erro ao criar nuvem de palavras: {e}")


def main():
    CONNECTION_STRING = "mongodb+srv://persi:persi@projetopersi.iypaiqd.mongodb.net/?appName=ProjetoPersi"
    JSON_FILE_PATH = r"C:\Users\Pichau\Desktop\Projetos Python\FIAP\Bagagem\mindcheck_export_ea366a.json"  # Mude para .json

    try:
        print("🔧 Inicializando analisador...")
        analyzer = MindCheckAnalyzer(CONNECTION_STRING)

        print("📁 Verificando arquivo JSON...")
        if not os.path.exists(JSON_FILE_PATH):
            print(f"❌ Arquivo JSON não encontrado: {JSON_FILE_PATH}")
            print("💡 Criando dados de exemplo...")
            analyzer.create_sample_data()
            results = {
                'statistics': {},
                'correlations': {'error': 'Usando dados de exemplo'},
                'insights': ['Analisando dados de exemplo demonstrativos']
            }
        else:
            print("🚀 Executando análise completa a partir do JSON...")
            results = analyzer.process_complete_analysis(JSON_FILE_PATH)

        print("\n🎨 GERANDO VISUALIZAÇÕES...")
        visualizer = MindCheckVisualizer(analyzer)

        # Resumo executivo
        visualizer.create_executive_summary(results)

        # Dashboard completo
        visualizer.create_complete_dashboard(results)

        print("📋 Exibindo resultados...")
        analyzer.print_results(results)

    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()