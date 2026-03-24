"""
ORACLE TTU-MC³ - Version Supabase avec Attracteur Circulaire
Théorie Triadique Unifiée - Modèle de Cohérence Cubique
"""

import streamlit as st
import os
import json
import uuid
import datetime
import hashlib
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

# Supabase
from supabase import create_client, Client

# Désactiver Plotly si nécessaire (optionnel)
PLOTLY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# Extraction de fichiers
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    CONVERGENCE_THRESHOLD = 1e-6
    ALPHA_M = 0.618
    ALPHA_C = 0.382
    ALPHA_D = 0.1
    ATTRACTOR_RADIUS = 1.0
    MAX_ITERATIONS = 50
    EMBEDDING_DIM = 128
    CHUNK_SIZE = 1000

# ==========================================
# ANALYSEUR MULTI-NIVEAUX
# ==========================================
class MultiLevelAnalyzer:
    def __init__(self):
        self.vowels = set('aeiouyàâäéèêëïîôöùûüÿAEIOUYÀÂÄÉÈÊËÏÎÔÖÙÛÜŸ')
        
    def analyze_letters(self, text: str) -> Dict:
        letters = [c for c in text if c.isalpha()]
        letter_freq = Counter(letters)
        vowels_count = sum(1 for c in letters if c in self.vowels)
        consonants_count = len(letters) - vowels_count
        return {
            "total_letters": len(letters),
            "unique_letters": len(letter_freq),
            "letter_frequency": dict(letter_freq.most_common(10)),
            "vowels": vowels_count,
            "consonants": consonants_count,
            "vowel_consonant_ratio": vowels_count / max(1, consonants_count)
        }
    
    def analyze_syllables(self, text: str) -> Dict:
        words = re.findall(r'\b\w+\b', text.lower())
        syllables = []
        for word in words:
            syl = re.findall(r'[aeiouyàâäéèêëïîôöùûüÿ]+[^aeiouyàâäéèêëïîôöùûüÿ]*', word)
            syllables.extend(syl)
        return {
            "total_syllables": len(syllables),
            "unique_syllables": len(set(syllables)),
            "avg_syllables_per_word": len(syllables) / max(1, len(words)),
            "most_common_syllables": dict(Counter(syllables).most_common(10))
        }
    
    def analyze_words(self, text: str) -> Dict:
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        return {
            "total_words": len(words),
            "unique_words": len(word_freq),
            "lexical_diversity": len(word_freq) / max(1, len(words)),
            "most_common_words": dict(word_freq.most_common(20)),
            "avg_word_length": sum(len(w) for w in words) / max(1, len(words))
        }
    
    def analyze_sentences(self, text: str) -> Dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]
        return {
            "total_sentences": len(sentences),
            "avg_sentence_length": sum(sentence_lengths) / max(1, len(sentences)),
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "sentences": sentences[:10]
        }
    
    def analyze_text_structure(self, text: str) -> Dict:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return {
            "total_paragraphs": len(paragraphs),
            "avg_paragraph_length": sum(len(p) for p in paragraphs) / max(1, len(paragraphs)),
            "total_characters": len(text),
            "total_lines": text.count('\n') + 1
        }
    
    def analyze_context(self, text: str) -> Dict:
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        themes = word_freq.most_common(15)
        return {
            "main_themes": themes,
            "lexical_richness": len(word_freq) / max(1, len(words))
        }
    
    def full_analysis(self, text: str) -> Dict:
        return {
            "letters": self.analyze_letters(text),
            "syllables": self.analyze_syllables(text),
            "words": self.analyze_words(text),
            "sentences": self.analyze_sentences(text),
            "structure": self.analyze_text_structure(text),
            "context": self.analyze_context(text)
        }

# ==========================================
# EXTRACTEUR DE FICHIERS
# ==========================================
class FileExtractor:
    @staticmethod
    def extract_text(file) -> Tuple[str, Dict]:
        filename = file.name.lower()
        file_type = "unknown"
        text = ""
        metadata = {"filename": filename, "size": 0}
        
        try:
            if filename.endswith('.txt') or hasattr(file, 'type') and file.type == 'text/plain':
                text = file.read().decode('utf-8')
                file_type = "text"
                metadata["size"] = len(text)
            
            elif filename.endswith('.pdf') and PDF_AVAILABLE:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
                file_type = "pdf"
                metadata["size"] = len(text)
            
            elif filename.endswith('.docx') and DOCX_AVAILABLE:
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                file_type = "word"
                metadata["size"] = len(text)
            
            elif filename.endswith(('.xlsx', '.xls', '.csv')):
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                text = df.to_string()
                file_type = "spreadsheet"
                metadata["rows"] = len(df)
                metadata["columns"] = len(df.columns)
                metadata["size"] = len(text)
            
            else:
                try:
                    text = file.read().decode('utf-8')
                    file_type = "text"
                except:
                    text = str(file.read())
                    file_type = "binary"
        
        except Exception as e:
            text = f"Erreur d'extraction: {str(e)}"
        
        return text, {"type": file_type, **metadata}
    
    @staticmethod
    def extract_by_chunks(file, chunk_size: int = Config.CHUNK_SIZE) -> List[str]:
        text, _ = FileExtractor.extract_text(file)
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# ==========================================
# ÉTAT TRIADIQUE
# ==========================================
class TriadicState:
    def __init__(self, phi_M=None, phi_C=None, phi_D=0.0, analysis=None):
        self.phi_M = list(phi_M) if phi_M else [0.0] * Config.EMBEDDING_DIM
        self.phi_C = list(phi_C) if phi_C else [0.0] * Config.EMBEDDING_DIM
        self.phi_D = float(phi_D)
        self.timestamp = datetime.datetime.now().timestamp()
        self.stability = 0.0
        self.convergence_history = []
        self.analysis = analysis or {}

# ==========================================
# FLOT TRIADIQUE AVEC ATTRACTEUR CIRCULAIRE
# ==========================================
class TriadicFlow:
    def __init__(self):
        self.alpha_M = Config.ALPHA_M
        self.alpha_C = Config.ALPHA_C
        self.alpha_D = Config.ALPHA_D
        self.radius = Config.ATTRACTOR_RADIUS
    
    def _norm(self, vec):
        try:
            return np.sqrt(sum(v * v for v in vec))
        except:
            return 0.0
    
    def _normalize(self, vec):
        norm = self._norm(vec)
        if norm > 1e-6:
            factor = self.radius / norm
            return [v * factor for v in vec]
        return vec
    
    def flow_equations(self, state: TriadicState, dt: float = 0.01) -> TriadicState:
        """
        Équations de flot corrigées avec attracteur de cercle.
        """
        try:
            # Équations différentielles originales
            dM = [-self.alpha_M * m + self.alpha_C * c + self.alpha_D * state.phi_D 
                  for m, c in zip(state.phi_M, state.phi_C)]
            dC = [-self.alpha_C * c + self.alpha_D * state.phi_D + self.alpha_M * m 
                  for m, c in zip(state.phi_M, state.phi_C)]
            dD = (-self.alpha_D * state.phi_D + 
                  self.alpha_M * sum(state.phi_M)/len(state.phi_M) + 
                  self.alpha_C * sum(state.phi_C)/len(state.phi_C))

            # Mise à jour par Euler
            new_M = [m + dt * dm for m, dm in zip(state.phi_M, dM)]
            new_C = [c + dt * dc for c, dc in zip(state.phi_C, dC)]
            new_D = state.phi_D + dt * dD

            # Force d'attracteur vers le cercle unité
            r = self._norm(new_M) + self._norm(new_C)
            if r > 1e-6:
                # Correction proportionnelle à l'écart par rapport à 1
                correction_factor = 1.0 + self.alpha_D * (1.0 - r) * dt * 10
                new_M = [m * correction_factor for m in new_M]
                new_C = [c * correction_factor for c in new_C]

            # Renormalisation finale pour garantir la stabilité
            new_M = self._normalize(new_M)
            new_C = self._normalize(new_C)

            # Ajustement supplémentaire pour éviter la dérive
            current_r = self._norm(new_M) + self._norm(new_C)
            if current_r < 0.95 or current_r > 1.05:
                final_factor = 1.0 / current_r
                new_M = [m * final_factor for m in new_M]
                new_C = [c * final_factor for c in new_C]

            return TriadicState(phi_M=new_M, phi_C=new_C, phi_D=new_D, analysis=state.analysis)
        except Exception as e:
            print(f"Erreur dans flow_equations: {e}")
            return state
    
    def converge(self, initial_state: TriadicState, max_iter: int = None) -> TriadicState:
        if max_iter is None:
            max_iter = Config.MAX_ITERATIONS
        
        state = initial_state
        history = []
        
        for _ in range(max_iter):
            try:
                prev_stability = state.stability
                state = self.flow_equations(state)
                norm_M = self._norm(state.phi_M)
                norm_C = self._norm(state.phi_C)
                state.stability = norm_M + norm_C + abs(state.phi_D)
                history.append(state.stability)
                if abs(state.stability - prev_stability) < Config.CONVERGENCE_THRESHOLD:
                    break
            except:
                break
        
        state.convergence_history = history
        return state
    
    def attractor_projection(self, state: TriadicState) -> Tuple[float, float]:
        try:
            if state.phi_M and state.phi_C:
                # Projection sur le cercle unité via les premières composantes
                cos_theta = state.phi_M[0] / self.radius if self.radius > 0 else 0.0
                sin_theta = state.phi_C[0] / self.radius if self.radius > 0 else 0.0
                # Normalisation pour rester sur le cercle
                norm = np.sqrt(cos_theta**2 + sin_theta**2)
                if norm > 1e-6:
                    cos_theta /= norm
                    sin_theta /= norm
            else:
                cos_theta, sin_theta = 0.0, 0.0
        except:
            cos_theta, sin_theta = 0.0, 0.0
        return (cos_theta, sin_theta)

# ==========================================
# ORACLE TTU-MC³ AVEC SUPABASE
# ==========================================
class TTUOracle:
    def __init__(self):
        # Initialisation du modèle d'embedding
        self.model = None
        if TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except:
                pass
        
        self.flow = TriadicFlow()
        self.analyzer = MultiLevelAnalyzer()
        self.extractor = FileExtractor()
        
        # Connexion Supabase
        self.supabase: Client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
        
        self.states: List[TriadicState] = []
        self.cycles: Dict[str, dict] = {}
        self.global_coherence = 0.0
        self._load_memory()
    
    def _load_memory(self):
        """Charge tous les cycles depuis Supabase."""
        try:
            response = self.supabase.table("knowledge").select("*").execute()
            for row in response.data:
                try:
                    # Conversion des vecteurs stockés au format texte
                    phi_m = self._parse_vector(row.get('phi_m'))
                    phi_c = self._parse_vector(row.get('phi_c'))
                    analysis = row.get('metadata', {})
                    if isinstance(analysis, str):
                        analysis = json.loads(analysis)
                    state = TriadicState(phi_M=phi_m, phi_C=phi_c, phi_D=row.get('phi_d', 0.0), analysis=analysis)
                    
                    self.cycles[row['id']] = {
                        "id": row['id'],
                        "state": state,
                        "attractor": (row.get('attractor_cos', 0.0), row.get('attractor_sin', 0.0))
                    }
                    self.states.append(state)
                except Exception as e:
                    print(f"Erreur chargement cycle {row.get('id')}: {e}")
                    continue
            self._update_coherence()
        except Exception as e:
            st.error(f"Erreur chargement depuis Supabase: {e}")
    
    def _parse_vector(self, vec_data):
        """Convertit une chaîne de type '[0.1,0.2,...]' en liste de floats."""
        if vec_data is None:
            return [0.0] * (Config.EMBEDDING_DIM // 2)
        if isinstance(vec_data, list):
            return vec_data
        if isinstance(vec_data, str):
            try:
                # Supprimer les crochets et splitter
                vec_str = vec_data.strip('[]')
                return [float(x) for x in vec_str.split(',') if x.strip()]
            except:
                return [0.0] * (Config.EMBEDDING_DIM // 2)
        return [0.0] * (Config.EMBEDDING_DIM // 2)
    
    def _encode_text_with_analysis(self, text: str, analysis: Dict = None) -> Tuple[List[float], List[float], float]:
        try:
            if self.model:
                embedding = self.model.encode(text[:500])
                if len(embedding) > Config.EMBEDDING_DIM:
                    embedding = embedding[:Config.EMBEDDING_DIM]
                elif len(embedding) < Config.EMBEDDING_DIM:
                    embedding = np.pad(embedding, (0, Config.EMBEDDING_DIM - len(embedding)))
            else:
                hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
                np.random.seed(hash_val % 2**32)
                embedding = np.random.randn(Config.EMBEDDING_DIM)
            
            if analysis:
                features = np.array([
                    analysis.get("words", {}).get("lexical_diversity", 0),
                    analysis.get("sentences", {}).get("avg_sentence_length", 0) / 50,
                    analysis.get("letters", {}).get("vowel_consonant_ratio", 0),
                    min(1.0, analysis.get("structure", {}).get("total_paragraphs", 0) / 100)
                ])
                if len(features) < len(embedding):
                    embedding[:len(features)] = embedding[:len(features)] * (1 + features)
            
            split = Config.EMBEDDING_DIM // 2
            phi_M = embedding[:split].tolist()
            phi_C = embedding[split:].tolist()
            phi_D = float(np.mean(embedding))
            
            # Normalisation pour qu'ils soient sur le cercle unité
            norm_M = np.sqrt(sum(v*v for v in phi_M))
            norm_C = np.sqrt(sum(v*v for v in phi_C))
            if norm_M > 0:
                phi_M = [v / norm_M for v in phi_M]
            if norm_C > 0:
                phi_C = [v / norm_C for v in phi_C]
            
            return phi_M, phi_C, phi_D
        except:
            return [0.0] * (Config.EMBEDDING_DIM // 2), [0.0] * (Config.EMBEDDING_DIM // 2), 0.5
    
    def learn(self, text: str, source: str = "text", analysis: Dict = None) -> Optional[str]:
        try:
            if analysis is None:
                analysis = self.analyzer.full_analysis(text[:5000])
            
            phi_M, phi_C, phi_D = self._encode_text_with_analysis(text, analysis)
            state = TriadicState(phi_M=phi_M, phi_C=phi_C, phi_D=phi_D, analysis=analysis)
            converged = self.flow.converge(state)
            attractor = self.flow.attractor_projection(converged)
            
            coherence = 1.0 - abs(attractor[0]) - abs(attractor[1])
            coherence = max(0.0, min(1.0, coherence))
            
            # Préparer les données pour Supabase
            data = {
                "content": text[:5000],
                "source": source,
                "phi_m": converged.phi_M,
                "phi_c": converged.phi_C,
                "phi_d": converged.phi_D,
                "attractor_cos": attractor[0],
                "attractor_sin": attractor[1],
                "coherence": coherence,
                "metadata": analysis
            }
            
            # Insertion
            response = self.supabase.table("knowledge").insert(data).execute()
            if response.data:
                cycle_id = response.data[0]['id']
                self.cycles[cycle_id] = {"id": cycle_id, "state": converged, "attractor": attractor}
                self.states.append(converged)
                self._update_coherence()
                return cycle_id
            return None
        except Exception as e:
            st.error(f"Erreur d'apprentissage: {e}")
            return None
    
    def learn_document(self, uploaded_file) -> Dict[str, Any]:
        try:
            text, metadata = self.extractor.extract_text(uploaded_file)
            
            if not text.strip():
                return {"success": False, "error": "Texte vide", "cycles": []}
            
            full_analysis = self.analyzer.full_analysis(text[:5000])
            cycle_id = self.learn(text[:5000], source=uploaded_file.name, analysis=full_analysis)
            
            chunks = self.extractor.extract_by_chunks(uploaded_file)
            chunk_cycles = []
            
            for i, chunk in enumerate(chunks[:5]):
                if chunk.strip():
                    chunk_analysis = self.analyzer.full_analysis(chunk[:5000])
                    chunk_id = self.learn(chunk[:5000], source=f"{uploaded_file.name} (chunk {i+1})", analysis=chunk_analysis)
                    if chunk_id:
                        chunk_cycles = []
            
            for i, chunk in enumerate(chunks[:5]):
                if chunk.strip():
                    chunk_analysis = self.analyzer.full_analysis(chunk[:5000])
                    chunk_id = self.learn(chunk[:5000], source=f"{uploaded_file.name} (chunk {i+1})", analysis=chunk_analysis)
                    if chunk_id:
                        chunk_cycles.append(chunk_id)
            
            return {
                "success": True,
                "cycles": [cycle_id] if cycle_id else [],
                "chunk_cycles": chunk_cycles,
                "metadata": metadata,
                "analysis": full_analysis,
                "total_chunks": len(chunks)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "cycles": []}
    
    def _update_coherence(self):
        if not self.cycles:
            self.global_coherence = 0.0
            return
        coherence_sum = 0.0
        for cycle in self.cycles.values():
            coherence = 1.0 - abs(cycle["attractor"][0]) - abs(cycle["attractor"][1])
            coherence_sum += max(0.0, min(1.0, coherence))
        self.global_coherence = coherence_sum / len(self.cycles)
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Recherche vectorielle utilisant la fonction match_knowledge_triadic."""
        try:
            phi_M_q, phi_C_q, _ = self._encode_text_with_analysis(query)
            # Appel de la fonction RPC
            response = self.supabase.rpc(
                'match_knowledge_triadic',
                {
                    'query_embedding_m': phi_M_q,
                    'query_embedding_c': phi_C_q,
                    'match_threshold': 0.5,
                    'match_count': top_k
                }
            ).execute()
            
            results = []
            for row in response.data:
                results.append({
                    "id": row['id'],
                    "similarity": row['combined_similarity'],
                    "attractor": None  # on peut récupérer plus tard si besoin
                })
            return results
        except Exception as e:
            print(f"Erreur recherche: {e}")
            return []
    
    def reason(self, question: str) -> Dict[str, Any]:
        results = self.search(question)
        if not results:
            return {"response": "Aucune connaissance trouvée.", "coherence": 0.0, "sources": [], "analysis": {}}
        
        # Récupérer les contenus depuis Supabase
        knowledge = []
        for r in results[:2]:
            try:
                row = self.supabase.table("knowledge").select("content, coherence, metadata, source").eq("id", r["id"]).execute()
                if row.data:
                    item = row.data[0]
                    knowledge.append({
                        "text": item["content"][:500],
                        "coherence": item["coherence"],
                        "similarity": r["similarity"],
                        "analysis": item.get("metadata", {}),
                        "source": item.get("source", "inconnu")
                    })
            except Exception as e:
                print(f"Erreur récupération source: {e}")
        
        coherence = knowledge[0]["coherence"] if knowledge else 0.0
        return {
            "response": self._generate_response(question, knowledge, coherence),
            "coherence": coherence,
            "sources": [{"text": k["text"][:200] + "...", "source": k.get("source", "inconnu")} for k in knowledge],
            "analysis": knowledge[0]["analysis"] if knowledge else {}
        }
    
    def _generate_response(self, question: str, knowledge: List[dict], coherence: float) -> str:
        if not knowledge:
            return "Aucune connaissance pertinente."
        
        parts = []
        for k in knowledge[:2]:
            parts.append(f"**Source:** {k.get('source', 'inconnu')}\n{k['text']}")
        
        coherence_ind = "✓" if coherence > 0.6 else "⚠" if coherence > 0.3 else "✗"
        return "\n\n---\n\n".join(parts) + f"\n\n---\n🌀 **Cohérence:** {coherence_ind} {coherence:.2f}"
    
    def get_stats(self) -> Dict:
        try:
            response = self.supabase.table("knowledge").select("word_count, sentence_count, letter_count", count="exact").execute()
            total_cycles = len(response.data)
            total_words = sum(row.get('word_count', 0) for row in response.data)
            total_sentences = sum(row.get('sentence_count', 0) for row in response.data)
            total_letters = sum(row.get('letter_count', 0) for row in response.data)
        except Exception as e:
            total_cycles = len(self.cycles)
            total_words = total_sentences = total_letters = 0
        
        stable = sum(1 for s in self.states if s.stability < 0.1)
        return {
            "cycles": total_cycles,
            "coherence": self.global_coherence,
            "stable": stable,
            "total_words": total_words,
            "total_sentences": total_sentences,
            "total_letters": total_letters
        }
    
    def get_attractors(self) -> pd.DataFrame:
        data = []
        for cycle_id, cycle in list(self.cycles.items())[:50]:
            data.append({
                "id": cycle_id[:6],
                "cos": cycle["attractor"][0],
                "sin": cycle["attractor"][1],
                "coherence": 1.0 - abs(cycle["attractor"][0]) - abs(cycle["attractor"][1])
            })
        return pd.DataFrame(data)
    
    def get_triad_state(self) -> Dict:
        if not self.states:
            return {"M": 0.0, "C": 0.0, "D": 0.0}
        return {
            "M": float(np.mean([s.phi_M[0] if s.phi_M else 0 for s in self.states])),
            "C": float(np.mean([s.phi_C[0] if s.phi_C else 0 for s in self.states])),
            "D": float(np.mean([s.phi_D for s in self.states]))
        }
    
    def analyze_text(self, text: str) -> Dict:
        return self.analyzer.full_analysis(text[:5000])

# ==========================================
# APPLICATION STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Oracle TTU-MC³", page_icon="🌀", layout="wide")
    
    @st.cache_resource
    def get_oracle():
        return TTUOracle()
    
    oracle = get_oracle()
    
    # CSS
    st.markdown("""
    <style>
    .title { text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px; }
    .high { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .medium { background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .low { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    .stat-card { background: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px; }
    .metric { font-size: 24px; font-weight: bold; }
    </style>
    <div class="title">
        <h1>🌀 Oracle TTU-MC³</h1>
        <p>Φ = (Φ_M, Φ_C, Φ_D) → Cercle Unité | Analyse multi-niveaux: lettre → syllabe → mot → phrase → texte → contexte</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("État Triadique")
        stats = oracle.get_stats()
        triad = oracle.get_triad_state()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cycles", stats["cycles"])
            st.metric("Cohérence", f"{stats['coherence']:.2f}")
        with col2:
            st.metric("Stables", stats["stable"])
            st.metric("Φ_D", f"{triad['D']:.2f}")
        
        st.divider()
        st.subheader("📊 Statistiques textuelles")
        
        st.markdown(f"""
        <div class="stat-card">
        📝 Mots: {stats['total_words']:,}<br>
        📖 Phrases: {stats['total_sentences']:,}<br>
        🔤 Lettres: {stats['total_letters']:,}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**État triadique:**")
        st.write(f"Φ_M: {triad['M']:.2f}")
        st.progress(min(1.0, max(0.0, triad['M'])))
        st.write(f"Φ_C: {triad['C']:.2f}")
        st.progress(min(1.0, max(0.0, triad['C'])))
        st.write(f"Φ_D: {triad['D']:.2f}")
        st.progress(min(1.0, max(0.0, triad['D'])))
        
        st.divider()
        st.json(stats)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌀 Apprentissage", "🔍 Interrogation", "🎯 Carte", "📊 Analyse"])
    
    with tab1:
        st.header("Apprentissage Triadique Multi-niveaux")
        st.markdown("""
        L'apprentissage analyse chaque texte à tous les niveaux:
        - **Lettres**: fréquences, voyelles/consonnes
        - **Syllabes**: structure phonétique
        - **Mots**: vocabulaire, diversité lexicale
        - **Phrases**: longueur, structure
        - **Texte**: paragraphes, structure globale
        - **Contexte**: thèmes, co-occurrences
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Texte")
            texte = st.text_area("Texte à apprendre", height=300, key="learn_text")
            
            if st.button("🌀 Apprendre le texte", type="primary", key="learn_btn"):
                if texte.strip():
                    with st.spinner("Analyse multi-niveaux et convergence..."):
                        cycle_id = oracle.learn(texte)
                        if cycle_id:
                            st.success(f"✅ Appris: {cycle_id[:8]}")
                            st.info(f"📊 Cohérence globale: {oracle.global_coherence:.3f}")
                            
                            analysis = oracle.analyze_text(texte)
                            with st.expander("📊 Analyse multi-niveaux"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.write("**Lettres**")
                                    st.write(f"Total: {analysis['letters']['total_letters']}")
                                    st.write(f"Voyelles: {analysis['letters']['vowels']}")
                                    st.write(f"Consonnes: {analysis['letters']['consonants']}")
                                with col_b:
                                    st.write("**Mots**")
                                    st.write(f"Total: {analysis['words']['total_words']}")
                                    st.write(f"Uniques: {analysis['words']['unique_words']}")
                                    st.write(f"Diversité: {analysis['words']['lexical_diversity']:.2f}")
                                with col_c:
                                    st.write("**Phrases**")
                                    st.write(f"Total: {analysis['sentences']['total_sentences']}")
                                    st.write(f"Moyenne: {analysis['sentences']['avg_sentence_length']:.1f} mots")
                            st.rerun()
                        else:
                            st.error("Erreur d'apprentissage")
                else:
                    st.warning("Entrez un texte")
        
        with col2:
            st.subheader("📁 Document")
            uploaded_file = st.file_uploader(
                "Choisissez un fichier",
                type=['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls'],
                help="Formats supportés: TXT, PDF, Word, Excel, CSV"
            )
            
            if uploaded_file:
                st.info(f"Fichier: {uploaded_file.name}")
                
                if st.button("📚 Apprendre le document", type="primary", key="learn_doc_btn"):
                    with st.spinner("Extraction, analyse multi-niveaux et convergence..."):
                        result = oracle.learn_document(uploaded_file)
                        if result["success"]:
                            st.success(f"✅ Document appris: {len(result['cycles'])} cycles principaux + {len(result['chunk_cycles'])} chunks")
                            st.info(f"📊 Métadonnées: {result['metadata']}")
                            
                            analysis = result.get("analysis", {})
                            with st.expander("📊 Analyse multi-niveaux du document"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.write("**Lettres**")
                                    st.write(f"Total: {analysis.get('letters', {}).get('total_letters', 0)}")
                                    st.write(f"Voyelles: {analysis.get('letters', {}).get('vowels', 0)}")
                                with col_b:
                                    st.write("**Mots**")
                                    st.write(f"Total: {analysis.get('words', {}).get('total_words', 0)}")
                                    themes = analysis.get('context', {}).get('main_themes', [])
                                    if themes:
                                        st.write("**Thèmes:**")
                                        for t in themes[:3]:
                                            st.write(f"- {t[0]}")
                                with col_c:
                                    st.write("**Structure**")
                                    st.write(f"Paragraphes: {analysis.get('structure', {}).get('total_paragraphs', 0)}")
                                    st.write(f"Phrases: {analysis.get('sentences', {}).get('total_sentences', 0)}")
                        else:
                            st.error(f"Erreur: {result.get('error', 'Inconnue')}")
    
    with tab2:
        st.header("Interrogation Triadique")
        
        question = st.text_input("💭 Votre question", key="question_input")
        
        if st.button("🌀 Interroger", type="primary", key="reason_btn"):
            if question.strip():
                with st.spinner("Convergence vers l'attracteur..."):
                    result = oracle.reason(question)
                    
                    coh = result["coherence"]
                    if coh > 0.6:
                        st.markdown(f'<div class="high">📊 Cohérence: {coh:.2f} (Élevée)</div>', unsafe_allow_html=True)
                    elif coh > 0.3:
                        st.markdown(f'<div class="medium">📊 Cohérence: {coh:.2f} (Moyenne)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="low">📊 Cohérence: {coh:.2f} (Faible)</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🌀 Réponse")
                    st.write(result["response"])
                    
                    if result["sources"]:
                        with st.expander("📚 Sources"):
                            for src in result["sources"]:
                                st.write(f"**{src['source']}**")
                                st.write(src["text"])
            else:
                st.warning("Entrez une question")
    
    with tab3:
        st.header("Carte des Attracteurs")
        st.markdown("Visualisation tabulaire des connaissances stabilisées.")
        
        df = oracle.get_attractors()
        
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            
            st.subheader("Statistiques des attracteurs")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Moyenne cos", f"{df['cos'].mean():.3f}")
                st.metric("Min cos", f"{df['cos'].min():.3f}")
                st.metric("Max cos", f"{df['cos'].max():.3f}")
            with col2:
                st.metric("Moyenne sin", f"{df['sin'].mean():.3f}")
                st.metric("Min sin", f"{df['sin'].min():.3f}")
                st.metric("Max sin", f"{df['sin'].max():.3f}")
            with col3:
                st.metric("Moyenne cohérence", f"{df['coherence'].mean():.3f}")
                st.metric("Min cohérence", f"{df['coherence'].min():.3f}")
                st.metric("Max cohérence", f"{df['coherence'].max():.3f}")
        else:
            st.info("Aucune connaissance apprise. Commencez par apprendre des textes ou documents.")
    
    with tab4:
        st.header("Analyse Multi-niveaux")
        st.markdown("Analysez n'importe quel texte à tous les niveaux")
        
        text_to_analyze = st.text_area("Texte à analyser", height=200, key="analyze_text")
        
        if st.button("🔍 Analyser", key="analyze_btn"):
            if text_to_analyze.strip():
                with st.spinner("Analyse multi-niveaux en cours..."):
                    analysis = oracle.analyze_text(text_to_analyze)
                    
                    tabs = st.tabs(["📝 Mots", "📖 Phrases", "🔤 Lettres", "🎵 Syllabes", "📚 Structure", "🌐 Contexte"])
                    
                    with tabs[0]:
                        st.subheader("Analyse lexicale")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mots totaux", analysis['words']['total_words'])
                        with col2:
                            st.metric("Mots uniques", analysis['words']['unique_words'])
                        with col3:
                            st.metric("Diversité lexicale", f"{analysis['words']['lexical_diversity']:.3f}")
                        st.write("**Mots les plus fréquents:**")
                        for word, count in list(analysis['words']['most_common_words'].items())[:10]:
                            st.write(f"- {word}: {count}")
                    
                    with tabs[1]:
                        st.subheader("Analyse des phrases")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Phrases totales", analysis['sentences']['total_sentences'])
                        with col2:
                            st.metric("Longueur moyenne", f"{analysis['sentences']['avg_sentence_length']:.1f} mots")
                        with col3:
                            st.metric("Longueur max", f"{analysis['sentences']['max_sentence_length']} mots")
                        st.write("**Échantillon de phrases:**")
                        for s in analysis['sentences']['sentences'][:5]:
                            st.write(f"- {s[:100]}...")
                    
                    with tabs[2]:
                        st.subheader("Analyse des lettres")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Lettres totales", analysis['letters']['total_letters'])
                        with col2:
                            st.metric("Lettres uniques", analysis['letters']['unique_letters'])
                        with col3:
                            st.metric("Ratio V/C", f"{analysis['letters']['vowel_consonant_ratio']:.2f}")
                        st.write("**Fréquence des lettres:**")
                        for letter, count in analysis['letters']['letter_frequency'].items():
                            st.write(f"- {letter}: {count}")
                    
                    with tabs[3]:
                        st.subheader("Analyse des syllabes")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Syllabes totales", analysis['syllables']['total_syllables'])
                        with col2:
                            st.metric("Syllabes uniques", analysis['syllables']['unique_syllables'])
                        st.write(f"**Moyenne syllabes/mot:** {analysis['syllables']['avg_syllables_per_word']:.2f}")
                    
                    with tabs[4]:
                        st.subheader("Structure du texte")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Paragraphes", analysis['structure']['total_paragraphs'])
                        with col2:
                            st.metric("Lignes", analysis['structure']['total_lines'])
                        st.write(f"**Caractères totaux:** {analysis['structure']['total_characters']:,}")
                    
                    with tabs[5]:
                        st.subheader("Contexte et thèmes")
                        st.write("**Thèmes principaux:**")
                        for theme, count in analysis['context']['main_themes'][:10]:
                            st.write(f"- {theme}: {count}")
            else:
                st.warning("Entrez un texte à analyser")
    
    st.divider()
    st.caption("🌀 Oracle TTU-MC³ - Théorie Triadique Unifiée | Analyse multi-niveaux: lettre → syllabe → mot → phrase → texte → contexte")

if __name__ == "__main__":
    main()
