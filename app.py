# app.py - Application Streamlit avec intégration de corpus (PDF, Word, TXT)

import streamlit as st
import numpy as np
import json
import sqlite3
import os
import uuid
import re
import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
import io
import tempfile

# =====================================================================
# TRAITEMENT DES FICHIERS (extraction de texte)
# =====================================================================

def extraire_texte_pdf(file_bytes: bytes) -> str:
    """Extrait le texte d'un fichier PDF."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except ImportError:
        st.error("PyPDF2 n'est pas installé. Installez-le avec: pip install PyPDF2")
        return ""

def extraire_texte_docx(file_bytes: bytes) -> str:
    """Extrait le texte d'un fichier DOCX."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        st.error("python-docx n'est pas installé. Installez-le avec: pip install python-docx")
        return ""

def extraire_texte_txt(file_bytes: bytes) -> str:
    """Extrait le texte d'un fichier TXT."""
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except:
        return file_bytes.decode("latin-1", errors="ignore")

def traiter_fichier(uploaded_file) -> str:
    """Traite un fichier uploadé et retourne son contenu texte."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == ".pdf":
        return extraire_texte_pdf(file_bytes)
    elif ext == ".docx":
        return extraire_texte_docx(file_bytes)
    elif ext == ".txt":
        return extraire_texte_txt(file_bytes)
    else:
        st.warning(f"Format non supporté : {ext}")
        return ""

# =====================================================================
# MODÈLES (reprenant models_phase3.py)
# =====================================================================

from dataclasses import dataclass, field

@dataclass
class CycleSpiral:
    accumulation: List[str] = field(default_factory=list)
    desordre: List[str] = field(default_factory=list)
    complexite: List[str] = field(default_factory=list)
    retour: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    resonance: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def est_complet(self) -> bool:
        return bool(self.accumulation) and bool(self.desordre) and bool(self.complexite) and bool(self.retour)

@dataclass
class EntreeDictionnaire:
    mot_francais: str
    mot_nzebi: str
    classe: int = 0
    definition: str = ""
    synonymes: List[str] = field(default_factory=list)
    antonymes: List[str] = field(default_factory=list)
    analogies_kuma: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class RelationSynergie:
    terme_a: str
    terme_b: str
    classe_a: int
    classe_b: int
    principe: str
    intensite: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class CandidateReponse:
    texte: str
    source_id: str
    score: float
    clan_support: Optional[str] = None
    proverbe_associe: Optional[str] = None

@dataclass
class JouteMbomo:
    question: str
    candidat_a: CandidateReponse
    candidat_b: CandidateReponse
    vainqueur: Optional[str] = None
    arbitre: str = "muyambili"
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class MemoireLongTerme:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    contenu: str = ""
    embedding: Optional[List[float]] = None
    force: float = 1.0
    dernier_acces: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    clan_associe: Optional[str] = None
    classes_associees: List[int] = field(default_factory=list)

# =====================================================================
# MOTEUR ORACLE PHASE 3 (adapté pour Streamlit)
# =====================================================================

# Configuration
MEMORY_FOLDER = "oracle_nzebi_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "oracle_phase3.db")
SATURATION_THRESHOLD = 0.42
TEMPERATURE_INITIALE = 0.75
SEUIL_SILENCE = 0.15
SEUIL_PROVERBE = 0.30
SEUIL_PARTIEL = 0.50

os.makedirs(MEMORY_FOLDER, exist_ok=True)

# Dictionnaire Muroni (extrait)
DICT_MURONI_EXTRACTION = [
    {"fr": "homme", "nz": "moutou", "classe": 1},
    {"fr": "femme", "nz": "moukassa", "classe": 1},
    {"fr": "cœur", "nz": "mutema", "classe": 2},
    {"fr": "arbre", "nz": "muti", "classe": 2},
    {"fr": "eau", "nz": "mamba", "classe": 3},
    {"fr": "chemin", "nz": "ndzela", "classe": 5},
    {"fr": "rivière", "nz": "ndzeli", "classe": 5},
    {"fr": "chaleur", "nz": "ndzoungouli", "classe": 7},
    {"fr": "transpiration", "nz": "ndzounguili", "classe": 7},
    {"fr": "perroquet", "nz": "koussou", "classe": 9},
    {"fr": "rougeur", "nz": "kusu", "classe": 9},
    {"fr": "constipation", "nz": "niodi", "classe": 9},
    {"fr": "eczéma", "nz": "ngomba", "classe": 9},
    {"fr": "abcès", "nz": "ivanga", "classe": 6},
    {"fr": "maladie", "nz": "mabeda", "classe": 3},
    {"fr": "sorcier", "nz": "mouloghi", "classe": 1},
    {"fr": "divinité", "nz": "boundzambi", "classe": 7},
    {"fr": "temps", "nz": "bouchi", "classe": 7},
    {"fr": "grandeur", "nz": "bunène", "classe": 7},
    {"fr": "intelligence", "nz": "bouyedi", "classe": 7},
    {"fr": "cécité", "nz": "boupipidi", "classe": 7},
    {"fr": "savoir", "nz": "yaba", "classe": 8},
]

# Synergies
SYNERGIES_EXAMPLES = [
    RelationSynergie("mutema", "muti", 2, 2, "Le cœur est l'arbre du corps ; l'arbre est le cœur de la forêt."),
    RelationSynergie("mamba", "ndzeli", 3, 5, "L'eau et la rivière sont une même essence."),
    RelationSynergie("moutema", "nzina", 2, 10, "Le sang et le cœur sont liés par la vie."),
]

# Clans fondateurs (simplifiés pour l'exemple)
CLANS_FONDATEURS = {
    "Buku": {"clan": "Mwanda", "devise": "Buku mwana Nzèbi, mwana wa Kana", "ikoko": "panthère", "lebutu": "palmier à huile"},
    "Mwélé": {"clan": "Makhamba", "devise": "Mwélé inéni, songa Malemba", "ikoko": "varan", "lebutu": "safoutier"},
    "Mombo": {"clan": "Seyi", "devise": "Mombo Mabakha, aba bakuna", "ikoko": "chat sauvage", "lebutu": "courge"},
    "Kombolo": {"clan": "Bakhuli", "devise": "Kombolo mu mutanya", "ikoko": "perroquet", "lebutu": "arachide"},
    "Bunzanga": {"clan": "Basanga", "devise": "Bunzanga mu mikélé", "ikoko": "chat domestique", "lebutu": "maïs"},
    "Ndombi": {"clan": "Mitsimba", "devise": "Ndombi abèta buye", "ikoko": "crocodile", "lebutu": "miel"},
    "Nyimbi": {"clan": "Mbundu", "devise": "Nyimbi a Mirumbi", "ikoko": "éléphant", "lebutu": "igname"},
}

# Proverbes
PROVERBES = [
    {"nzebi": "Na koungi tata ndzoungou itèghèvè", "francais": "Avec trois bûches, une marmite ne se renverse pas", "implicature": "Solidarité clanique"},
    {"nzebi": "Koutou a tsinga mou mandongui", "francais": "L'homme devient habile grâce aux conseils", "implicature": "Sagesse relationnelle"},
    {"nzebi": "Noulemba moumochi sa tsoka lêboundzou vè", "francais": "Un seul doigt ne peut pas laver la figure", "implicature": "Entraide nécessaire"},
    {"nzebi": "Bandrèkè batswala bola kèlè", "francais": "Les tisserins ont amené le bruit dans le village", "implicature": "Parole excessive"},
    {"nzebi": "Tchindi a vida mèdi mou sèbè bangwabata", "francais": "L'écureuil manque de graisse parce qu'il se rit des grands", "implicature": "Respect des anciens"},
    {"nzebi": "Nga moukéla chi takagha mbwagha vè", "francais": "Celui qui possède une queue ne doit pas traverser le feu", "implicature": "Prudence et conséquences cycliques"},
    {"nzebi": "Houghi ghou youlou, batchwi ghou ndzeli", "francais": "L'abeille est au ciel, les poissons dans la rivière", "implicature": "Limites de la connaissance"},
]

# =====================================================================
# CLASSE OraclePhase3 (adaptée)
# =====================================================================

class OraclePhase3:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except:
            st.warning("SentenceTransformers non installé. Utilisation d'un modèle simulé (fonctionnalités limitées).")
            self.model = None
        
        self.phi_m = 0.15
        self.phi_c = 0.0
        self.phi_d = 0.0
        self.temperature = TEMPERATURE_INITIALE
        self.phase = 0.0
        self.tour = 0
        self.cycles = []
        self.cycle_actuel = {"accumulation": [], "desordre": [], "complexite": [], "retour": []}
        self.memory = []
        self.embeddings_matrix = None
        self.memoire_lt = []
        self.synergies = SYNERGIES_EXAMPLES
        self._proverbes = PROVERBES
        self._proverbe_embeddings = None
        
        self._init_db()
        self._load_memory()
        self._build_embedding_matrix()
    
    def _init_db(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, name TEXT, timestamp TEXT, source TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS paragraphs (id TEXT PRIMARY KEY, doc_id TEXT, text TEXT, embedding TEXT, timestamp TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS sentences (id TEXT PRIMARY KEY, para_id TEXT, text TEXT, embedding TEXT, timestamp TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS words (id TEXT PRIMARY KEY, word TEXT, nominal_class INTEGER, frequency INTEGER, doc_id TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS concepts (concept TEXT, ref_id TEXT, ref_type TEXT, weight REAL, PRIMARY KEY (concept, ref_id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS cycles (id TEXT PRIMARY KEY, accumulation TEXT, desordre TEXT, complexite TEXT, retour TEXT, timestamp TEXT, resonance REAL)")
        cursor.execute("CREATE TABLE IF NOT EXISTS memoire_lt (id TEXT PRIMARY KEY, contenu TEXT, embedding TEXT, force REAL, dernier_acces REAL, clan_associe TEXT, classes_associees TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS proverbes_db (id TEXT PRIMARY KEY, nzebi TEXT, francais TEXT, implicature TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS dictionnaire (id TEXT PRIMARY KEY, mot_francais TEXT, mot_nzebi TEXT, classe INTEGER, definition TEXT, synonymes TEXT, antonymes TEXT, analogies TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS synergies (id TEXT PRIMARY KEY, terme_a TEXT, terme_b TEXT, classe_a INTEGER, classe_b INTEGER, principe TEXT, intensite REAL)")
        cursor.execute("CREATE TABLE IF NOT EXISTS joutes_mbomo (id TEXT PRIMARY KEY, question TEXT, candidat_a TEXT, candidat_b TEXT, vainqueur TEXT, timestamp REAL)")
        # Insérer les proverbes
        for p in PROVERBES:
            cursor.execute("INSERT OR IGNORE INTO proverbes_db (id, nzebi, francais, implicature) VALUES (?,?,?,?)",
                          (str(uuid.uuid4()), p["nzebi"], p["francais"], p["implicature"]))
        for entree in DICT_MURONI_EXTRACTION:
            cursor.execute("INSERT OR IGNORE INTO dictionnaire (id, mot_francais, mot_nzebi, classe) VALUES (?,?,?,?)",
                          (str(uuid.uuid4()), entree["fr"], entree["nz"], entree["classe"]))
        for syn in SYNERGIES_EXAMPLES:
            cursor.execute("INSERT OR IGNORE INTO synergies (id, terme_a, terme_b, classe_a, classe_b, principe, intensite) VALUES (?,?,?,?,?,?,?)",
                          (syn.id, syn.terme_a, syn.terme_b, syn.classe_a, syn.classe_b, syn.principe, syn.intensite))
        conn.commit()
        conn.close()
    
    def _load_memory(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, embedding, timestamp FROM paragraphs")
        for row in cursor.fetchall():
            emb = np.array(json.loads(row[2])) if row[2] else None
            self.memory.append({"id": row[0], "text": row[1], "embedding": emb, "timestamp": row[3]})
        # Charger les cycles
        cursor.execute("SELECT id, accumulation, desordre, complexite, retour, timestamp, resonance FROM cycles")
        for row in cursor.fetchall():
            cycle = CycleSpiral(
                accumulation=json.loads(row[1]) if row[1] else [],
                desordre=json.loads(row[2]) if row[2] else [],
                complexite=json.loads(row[3]) if row[3] else [],
                retour=json.loads(row[4]) if row[4] else [],
                timestamp=float(row[5]) if row[5] else datetime.datetime.now().timestamp(),
                resonance=float(row[6]) if row[6] else 0.0,
                id=row[0]
            )
            self.cycles.append(cycle)
            if cycle.est_complet():
                self.tour += 1
        # Charger mémoire LT
        cursor.execute("SELECT id, contenu, embedding, force, dernier_acces, clan_associe, classes_associees FROM memoire_lt")
        for row in cursor.fetchall():
            emb = np.array(json.loads(row[2])) if row[2] else None
            self.memoire_lt.append({
                "id": row[0],
                "contenu": row[1],
                "embedding": emb,
                "force": row[3],
                "dernier_acces": row[4],
                "clan": row[5],
                "classes": json.loads(row[6]) if row[6] else []
            })
        conn.close()
    
    def _build_embedding_matrix(self):
        if self.model is None:
            self.embeddings_matrix = None
            return
        vectors = [m["embedding"] for m in self.memory if m["embedding"] is not None]
        if not vectors:
            self.embeddings_matrix = None
            return
        mat = np.vstack(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.embeddings_matrix = mat / np.where(norms == 0, 1.0, norms)
    
    def detecter_classe(self, mot: str) -> int:
        mot = mot.lower()
        if mot.startswith("mu") or mot.startswith("mou"):
            if any(mot.startswith(p) for p in ['moutou', 'mouloghi', 'moukassa']):
                return 1
            return 2
        if mot.startswith("ba"):
            return 1
        if mot.startswith("mi"):
            return 2
        if mot.startswith("bi"):
            return 6
        if mot.startswith("le"):
            return 4
        if mot.startswith("ma"):
            if mot in ['mamba', 'massoba']:
                return 3
            return 5
        if mot.startswith("i"):
            return 6
        if mot.startswith("bu") or mot.startswith("bou"):
            return 7
        return 9
    
    def trouver_synergies(self, mot: str) -> List[Dict]:
        result = []
        for syn in self.synergies:
            if syn.terme_a == mot or syn.terme_b == mot:
                result.append({"a": syn.terme_a, "b": syn.terme_b, "principe": syn.principe, "intensite": syn.intensite})
        return result
    
    def charger_proverbes(self) -> List[Dict]:
        return self._proverbes
    
    def apprendre(self, text: str, source: str = "direct") -> int:
        if not text.strip() or self.model is None:
            return 0
        # Découpage en blocs sémantiques (paragraphes)
        blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 30]
        if not blocks:
            blocks = [text.strip()] if len(text.strip()) > 30 else []
        if not blocks:
            return 0
        doc_id = str(uuid.uuid4())
        ts = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents VALUES (?,?,?,?)", (doc_id, source, ts, "text_stream"))
        count = 0
        for block in blocks:
            emb = self.model.encode(block)
            para_id = str(uuid.uuid4())
            self.memory.append({"id": para_id, "text": block, "embedding": emb, "timestamp": ts})
            cursor.execute("INSERT INTO paragraphs VALUES (?,?,?,?,?)", 
                          (para_id, doc_id, block, json.dumps(emb.tolist()), ts))
            clean_words = re.findall(r"\w+", block.lower())
            stop_words = {"le", "la", "les", "de", "des", "en", "dans", "est", "pour", "par", "qui", "que", "et", "à", "au", "aux"}
            for word in clean_words:
                if word not in stop_words and len(word) > 3:
                    cls = self.detecter_classe(word)
                    cursor.execute("INSERT OR IGNORE INTO words (id, word, nominal_class, frequency, doc_id) VALUES (?,?,?,?,?)",
                                  (str(uuid.uuid4()), word, cls, 1, doc_id))
                    cursor.execute("INSERT OR IGNORE INTO concepts VALUES (?,?,?,?)", 
                                  (word, para_id, "paragraph", 1.0))
            count += 1
        conn.commit()
        conn.close()
        self._build_embedding_matrix()
        return count
    
    def calculer_resonance_spiralee(self, query_embed: np.ndarray) -> Tuple[List[Tuple[Dict, float]], float]:
        if self.embeddings_matrix is None or len(self.memory) == 0 or self.model is None:
            return [], 0.0
        scores = np.dot(self.embeddings_matrix, query_embed)
        self.phi_c = float(np.max(scores)) if len(scores) > 0 else 0.0
        self.phi_d = float(np.var(scores) * 4.5) if len(scores) > 1 else 0.35
        phase_factor = 1.0 + 0.4 * np.sin(self.phase * 2 * np.pi)
        resonance = self.phi_c / (self.phi_d + 0.01)
        resonance_ajustee = resonance * phase_factor
        seuil = SATURATION_THRESHOLD * self.temperature * (0.7 + 0.3 * (1 - self.phase))
        condensated = []
        for idx, score in enumerate(scores):
            if score > seuil:
                condensated.append((self.memory[idx], float(score)))
        condensated.sort(key=lambda x: x[1], reverse=True)
        return condensated[:5], resonance_ajustee
    
    def evoluer_spirale(self, resonance: float, question: str = ""):
        pas = 0.015 + 0.035 * min(1.0, resonance / 1.5)
        ancienne_phase = self.phase
        self.phase = (self.phase + pas) % 1.0
        if self.phase < 0.1 and ancienne_phase >= 0.1:
            self.cycle_actuel["accumulation"].append(f"Résonance: {resonance:.3f} | {question[:50]}")
        elif 0.25 <= self.phase < 0.35 and ancienne_phase < 0.25:
            self.cycle_actuel["desordre"].append(f"Purification: {self.phi_d:.3f}")
        elif 0.50 <= self.phase < 0.60 and ancienne_phase < 0.50:
            self.cycle_actuel["complexite"].append(f"Synthèse: {self.phi_c:.3f}")
        elif 0.75 <= self.phase < 0.85 and ancienne_phase < 0.75:
            self.cycle_actuel["retour"].append(f"Retour: {self.phi_m:.3f}")
            if all(len(self.cycle_actuel[k]) > 0 for k in ["accumulation", "desordre", "complexite", "retour"]):
                cycle = CycleSpiral(
                    accumulation=self.cycle_actuel["accumulation"],
                    desordre=self.cycle_actuel["desordre"],
                    complexite=self.cycle_actuel["complexite"],
                    retour=self.cycle_actuel["retour"],
                    resonance=resonance
                )
                self.cycles.append(cycle)
                self.tour += 1
                self.cycle_actuel = {"accumulation": [], "desordre": [], "complexite": [], "retour": []}
                # Sauvegarde en base
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO cycles (id, accumulation, desordre, complexite, retour, timestamp, resonance) VALUES (?,?,?,?,?,?,?)",
                              (cycle.id, json.dumps(cycle.accumulation), json.dumps(cycle.desordre), 
                               json.dumps(cycle.complexite), json.dumps(cycle.retour), str(cycle.timestamp), resonance))
                conn.commit()
                conn.close()
    
    def ajouter_memoire_lt(self, contenu: str, clan: Optional[str] = None, classes: Optional[List[int]] = None):
        if self.model is None:
            return
        emb = self.model.encode(contenu).tolist()
        mem = MemoireLongTerme(contenu=contenu, embedding=emb, force=1.0, clan_associe=clan, classes_associees=classes or [])
        self.memoire_lt.append({
            "id": mem.id,
            "contenu": mem.contenu,
            "embedding": np.array(mem.embedding),
            "force": mem.force,
            "dernier_acces": mem.dernier_acces,
            "clan": mem.clan_associe,
            "classes": mem.classes_associees
        })
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memoire_lt (id, contenu, embedding, force, dernier_acces, clan_associe, classes_associees) VALUES (?,?,?,?,?,?,?)",
                      (mem.id, mem.contenu, json.dumps(mem.embedding), mem.force, mem.dernier_acces, mem.clan_associe, json.dumps(mem.classes_associees)))
        conn.commit()
        conn.close()
    
    def rafraichir_memoire_lt(self):
        a_supprimer = []
        for i, mem in enumerate(self.memoire_lt):
            mem["force"] *= 0.99
            if mem["force"] < 0.01:
                a_supprimer.append(i)
        for idx in reversed(a_supprimer):
            del self.memoire_lt[idx]
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memoire_lt WHERE id = ?", (mem["id"],))
            conn.commit()
            conn.close()
    
    def joute_mbomo(self, question: str, candidat_a: str, candidat_b: str) -> JouteMbomo:
        if self.model is None:
            # Fallback
            return JouteMbomo(question=question, 
                            candidat_a=CandidateReponse(texte=candidat_a, source_id="", score=0.5),
                            candidat_b=CandidateReponse(texte=candidat_b, source_id="", score=0.5),
                            vainqueur="a")
        q_emb = self.model.encode(question)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        emb_a = self.model.encode(candidat_a)
        emb_a_norm = emb_a / (np.linalg.norm(emb_a) + 1e-8)
        emb_b = self.model.encode(candidat_b)
        emb_b_norm = emb_b / (np.linalg.norm(emb_b) + 1e-8)
        score_a = float(np.dot(emb_a_norm, q_norm))
        score_b = float(np.dot(emb_b_norm, q_norm))
        vainqueur = "a" if score_a > score_b else "b"
        ca = CandidateReponse(texte=candidat_a, source_id="", score=score_a)
        cb = CandidateReponse(texte=candidat_b, source_id="", score=score_b)
        joute = JouteMbomo(question=question, candidat_a=ca, candidat_b=cb, vainqueur=vainqueur)
        # Sauvegarde
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO joutes_mbomo (id, question, candidat_a, candidat_b, vainqueur, timestamp) VALUES (?,?,?,?,?,?)",
                      (joute.id, joute.question, candidat_a, candidat_b, vainqueur, joute.timestamp))
        conn.commit()
        conn.close()
        return joute
    
    def raisonner(self, question: str) -> str:
        if not question.strip():
            return "🌀 La question est vide."
        if self.model is None:
            return "⚠️ Modèle non disponible. Veuillez installer sentence-transformers."
        q_emb = self.model.encode(question)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        results, resonance = self.calculer_resonance_spiralee(q_norm)
        for doc, _ in results[:1]:
            self.ajouter_memoire_lt(doc["text"][:200], clan=None)
        self.rafraichir_memoire_lt()
        self.evoluer_spirale(resonance, question)
        if resonance < SEUIL_SILENCE:
            return self._construire_reponse_silence(question, resonance)
        elif resonance < SEUIL_PROVERBE:
            return self._construire_reponse_proverbe(question, resonance)
        elif resonance < SEUIL_PARTIEL:
            return self._construire_reponse_partielle(question, results, resonance)
        else:
            return self._construire_reponse_complete(question, results, resonance)
    
    def _construire_reponse_silence(self, question: str, resonance: float) -> str:
        output = f"⚠️ SILENCE SIGNIFIANT (Mbomo)\n{'━'*50}\n"
        output += "La question retourne à son origine comme un miroir.\n"
        output += "Rien ne s'est cristallisé. Le savoir s'est purifié dans le silence.\n"
        output += f"\nRésonance: {resonance:.3f} | Phase: {self.phase:.2f} | Tour: {self.tour}"
        return output
    
    def _construire_reponse_proverbe(self, question: str, resonance: float) -> str:
        proverbes = self.charger_proverbes()
        if proverbes and self.model is not None:
            q_emb = self.model.encode(question)
            best_prov = None
            best_score = -1
            for p in proverbes:
                emb = self.model.encode(p["francais"])
                score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
                if score > best_score:
                    best_score = score
                    best_prov = p
            if best_prov and best_score > 0.25:
                output = f"🌀 PAROLE PROVERBIALE (Bisega)\n{'━'*50}\n"
                output += f"« {best_prov['nzebi']} »\n→ {best_prov['francais']}\nImplicature : {best_prov['implicature']}\n"
                output += f"\nRésonance: {resonance:.3f} | Phase: {self.phase:.2f} | Tour: {self.tour}"
                return output
        return self._construire_reponse_silence(question, resonance * 0.8)
    
    def _construire_reponse_partielle(self, question: str, results: List[Tuple[Dict, float]], resonance: float) -> str:
        output = f"🌀 LIQUÉFACTION (Phase de désordre/complexité)\n{'━'*50}\n"
        output += f"| Φ_M: {self.phi_m:.4f} | Φ_C: {self.phi_c:.4f} | Φ_D: {self.phi_d:.4f}\n"
        output += f"| Résonance: {resonance:.3f} | Phase: {self.phase:.2f} | Tour: {self.tour}\n{'━'*50}\n\n"
        if results:
            best = results[0]
            output += f"« {best[0]['text'][:400]} »\n[Confiance: {best[1]:.3f}]\n"
        else:
            output += "Aucune substance condensée. Le savoir reste en suspension.\n"
        return output
    
    def _construire_reponse_complete(self, question: str, results: List[Tuple[Dict, float]], resonance: float) -> str:
        output = f"🌀 CRISTALLISATION (Attracteur Stable)\n{'━'*50}\n"
        output += f"| Φ_M: {self.phi_m:.4f} | Φ_C: {self.phi_c:.4f} | Φ_D: {self.phi_d:.4f}\n"
        output += f"| Résonance: {resonance:.3f} | Phase: {self.phase:.2f} | Tour: {self.tour}\n{'━'*50}\n\n"
        # Synergies
        mots = re.findall(r'\w+', question.lower())
        synergies_trouvees = []
        for mot in mots:
            syns = self.trouver_synergies(mot)
            if syns:
                synergies_trouvees.extend(syns)
        if synergies_trouvees:
            output += "⚡ SYNERGIES DÉTECTÉES :\n"
            for syn in synergies_trouvees[:3]:
                output += f"  • {syn['a']} ↔ {syn['b']} : {syn['principe']}\n"
            output += "\n"
        if results:
            output += "=== SUBSTANCE CONDENSÉE ===\n"
            for i, (doc, score) in enumerate(results[:3]):
                output += f"\n[{i+1}] (Résonance {score:.3f})\n{doc['text'][:500]}\n"
        else:
            output += "Aucune substance condensée.\n"
        return output
    
    def statut_spiral(self) -> Dict:
        return {
            "phi_m": self.phi_m,
            "phi_c": self.phi_c,
            "phi_d": self.phi_d,
            "temperature": self.temperature,
            "phase": self.phase,
            "tour": self.tour,
            "cycles_complets": len([c for c in self.cycles if c.est_complet()]),
            "nb_memoires": len(self.memory),
            "nb_memoire_lt": len(self.memoire_lt),
            "nb_synergies": len(self.synergies)
        }
    
    def get_donnees_spirale(self) -> Dict:
        return {
            "phi_m": self.phi_m,
            "phi_c": self.phi_c,
            "phi_d": self.phi_d,
            "phase": self.phase,
            "tour": self.tour,
            "cycles": [c.to_dict() if hasattr(c, 'to_dict') else {
                "accumulation": c.accumulation,
                "desordre": c.desordre,
                "complexite": c.complexite,
                "retour": c.retour,
                "resonance": c.resonance
            } for c in self.cycles],
            "synergies": [{"a": s.terme_a, "b": s.terme_b, "principe": s.principe} for s in self.synergies],
            "memoire_lt": [{"contenu": m["contenu"][:100], "force": m["force"]} for m in self.memoire_lt[-20:]]
        }

# =====================================================================
# APPLICATION STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="🌀 TTU-MC³ – Spirale Vivante",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #080b0a; }
    .stMetric { background-color: rgba(0,255,136,0.04); border: 1px solid rgba(0,255,136,0.12); border-radius: 8px; padding: 10px; }
    .stMarkdown { font-family: 'Courier New', monospace; }
    .css-1d391kg { background-color: #080b0a; }
    </style>
""", unsafe_allow_html=True)

# Initialisation du moteur dans session_state
if 'engine' not in st.session_state:
    st.session_state.engine = OraclePhase3()
engine = st.session_state.engine

# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.title("🌀 TTU-MC³")
st.sidebar.markdown("**Phase 4 · Spirale Vivante**")
st.sidebar.markdown("---")

st.sidebar.metric("Φ_M · Inertie", f"{engine.phi_m:.4f}")
st.sidebar.metric("Φ_C · Flux", f"{engine.phi_c:.4f}")
st.sidebar.metric("Φ_D · Dissipation", f"{engine.phi_d:.4f}")
st.sidebar.metric("🌡️ Température", f"{engine.temperature:.2f}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Phase :** {engine.phase:.2f}")
st.sidebar.markdown(f"**Tour :** {engine.tour}")
st.sidebar.markdown(f"**Mémoire :** {len(engine.memory)}")
st.sidebar.markdown(f"**Cycles :** {len(engine.cycles)}")

# =====================================================================
# MAIN
# =====================================================================

st.title("🌀 TTU-MC³ – Temps Spiralé")
st.caption("Cosmogonie Nzèbi · Algorithme Cognitif Bantu B52")

# Métriques principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("Φ_M · Inertie (Koto)", f"{engine.phi_m:.4f}")
col2.metric("Φ_C · Flux (Tchengl)", f"{engine.phi_c:.4f}")
col3.metric("Φ_D · Dissipation", f"{engine.phi_d:.4f}")
col4.metric("🌡️ Température", f"{engine.temperature:.2f}")

# Phase Bar
st.markdown("---")
cols = st.columns(4)
phases = ["Accumulation", "Désordre", "Complexité", "Retour"]
for i, (col, label) in enumerate(zip(cols, phases)):
    seuil_bas = i * 0.25
    seuil_haut = (i + 1) * 0.25
    actif = engine.phase >= seuil_bas and engine.phase < seuil_haut
    passe = engine.phase >= seuil_haut
    color = "#00ff88" if actif else ("rgba(0,255,136,0.2)" if passe else "rgba(0,255,136,0.05)")
    col.markdown(f"<div style='background:{color}; padding:8px; border-radius:6px; text-align:center; font-size:0.8rem;'>{label}</div>", unsafe_allow_html=True)
st.markdown("---")

# Layout 2 colonnes
col_left, col_right = st.columns([3, 2])

# Colonne gauche : Visualisation
with col_left:
    st.subheader("🌀 Spirale des cycles")
    # Visualisation avec matplotlib
    try:
        import matplotlib.pyplot as plt
        data = engine.get_donnees_spirale()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor('#080b0a')
        fig.patch.set_facecolor('#080b0a')
        cx, cy = 0.5, 0.5
        radius = 0.4
        # Cercles concentriques
        for r in np.linspace(0.1, radius, 5):
            circle = plt.Circle((cx, cy), r, fill=False, color='rgba(0,255,136,0.05)', linewidth=0.5)
            ax.add_patch(circle)
        # Lignes radiales
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            ax.plot([cx, cx + radius*np.cos(angle)], [cy, cy + radius*np.sin(angle)], color='rgba(0,255,136,0.03)', linewidth=0.5)
        # Point central
        ax.plot(cx, cy, 'o', color='#00ff88', markersize=4, alpha=0.6)
        # Cycles passés (spirale)
        cycles = data.get('cycles', [])
        if cycles:
            points = []
            for idx, cycle in enumerate(cycles):
                t = (idx + 1) / (len(cycles) + 1)
                r = radius * (0.2 + 0.6 * t)
                angle = (idx / len(cycles)) * 2 * np.pi - np.pi/2
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                points.append((x, y))
            # Ajouter point actuel
            phase_angle = engine.phase * 2 * np.pi - np.pi/2
            current_r = radius * (0.3 + 0.6 * (engine.phi_m + engine.phi_c) / 2)
            x_cur = cx + current_r * np.cos(phase_angle)
            y_cur = cy + current_r * np.sin(phase_angle)
            points.append((x_cur, y_cur))
            if len(points) > 1:
                xs, ys = zip(*points)
                ax.plot(xs, ys, color='rgba(0,255,136,0.4)', linewidth=2, linestyle='dashed')
        # Point actuel
        ax.plot(x_cur, y_cur, 'o', color='#00ff88', markersize=8, alpha=0.9)
        ax.text(x_cur + 0.02, y_cur - 0.02, 'actuel', color='#00ff88', fontsize=8, alpha=0.6)
        # Étiquettes des phases
        phase_labels = ['Accum.', 'Désordre', 'Complex.', 'Retour']
        for i, label in enumerate(phase_labels):
            angle = (i / 4) * 2 * np.pi - np.pi/2
            r = radius + 0.08
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            is_active = (engine.phase >= i * 0.25 and engine.phase < (i + 1) * 0.25)
            ax.text(x, y, label, color='#00ff88' if is_active else 'rgba(0,255,136,0.2)', fontsize=9, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig)
    except:
        st.info("Visualisation nécessite matplotlib. Installez matplotlib pour afficher la spirale.")

# Colonne droite : Formulaires
with col_right:
    # Query
    with st.expander("🔍 Interroger la spirale", expanded=True):
        with st.form("query_form"):
            question = st.text_input("Question", placeholder="Soumettre une trajectoire conceptuelle...")
            submit_query = st.form_submit_button("Mesurer la Résonance")
            if submit_query and question:
                with st.spinner("Résonance en cours..."):
                    answer = engine.raisonner(question)
                    st.session_state.last_answer = answer
    # Affichage réponse
    if 'last_answer' in st.session_state:
        st.markdown("---")
        st.markdown("**Réponse spiralée**")
        st.code(st.session_state.last_answer, language="text")

# =====================================================================
# INTÉGRATION DE CORPUS (Upload de fichiers)
# =====================================================================

st.markdown("---")
st.subheader("📚 Intégrer un corpus complet (PDF, DOCX, TXT)")

with st.container():
    uploaded_files = st.file_uploader(
        "Choisir des fichiers (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(uploaded_files)
        total_blocks = 0
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Traitement de : {file.name} ({i+1}/{total})")
            with st.spinner(f"Extraction du texte de {file.name}..."):
                text = traiter_fichier(file)
            if text.strip():
                with st.spinner(f"Indexation de {file.name}..."):
                    blocks = engine.apprendre(text, source=file.name)
                    total_blocks += blocks
                    st.success(f"✅ {file.name} : {blocks} blocs indexés")
            else:
                st.warning(f"⚠️ Aucun texte extrait de {file.name}")
            progress_bar.progress((i + 1) / total)
        
        status_text.text(f"Intégration terminée. Total : {total_blocks} blocs indexés.")
        st.balloons()

# Apprentissage direct (texte)
with st.expander("✏️ Saisie directe (texte)"):
    with st.form("learn_form"):
        text = st.text_area("Texte", placeholder="Saisir la substance textuelle brute à indexer...", height=150)
        submit_learn = st.form_submit_button("Intégrer")
        if submit_learn and text:
            blocks = engine.apprendre(text)
            st.success(f"🌀 Intégration effectuée : {blocks} cellules informationnelles stabilisées.")

# Joute Mbomo
with st.expander("⚔️ Joute oratoire (Mbomo)"):
    with st.form("mbomo_form"):
        q_mb = st.text_input("Question")
        a_mb = st.text_input("Candidat A")
        b_mb = st.text_input("Candidat B")
        submit_mb = st.form_submit_button("Lancer la joute")
        if submit_mb and q_mb and a_mb and b_mb:
            joute = engine.joute_mbomo(q_mb, a_mb, b_mb)
            output = f"🌀 JOUTE MBOMO\n{'━'*50}\nQuestion: {q_mb}\n\n"
            output += f"Candidat A: {a_mb[:100]}... (score: {joute.candidat_a.score:.3f})\n"
            output += f"Candidat B: {b_mb[:100]}... (score: {joute.candidat_b.score:.3f})\n"
            output += f"\n🏆 Vainqueur: {'A' if joute.vainqueur == 'a' else 'B'}\n"
            output += f"Arbitre: {joute.arbitre}"
            st.code(output, language="text")

# Status
if st.button("📊 Statut complet"):
    stat = engine.statut_spiral()
    output = "📊 STATUT COMPLET\n" + "━"*50 + "\n"
    for k, v in stat.items():
        output += f"{k}: {v}\n"
    output += "\n🌀 Bulong tchengl bwa tchengl"
    st.code(output, language="text")

st.markdown("---")
st.caption("🌀 TTU-MC³ · Algorithme Cognitif Bantu (Nzèbi B52) · Temps Spiralé")
st.caption("« Bulong tchengl bwa tchengl » — La Terre est un tournoiement perpétuel.")