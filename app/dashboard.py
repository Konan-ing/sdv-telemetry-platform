import os
import requests
import streamlit as st
from langchain.prompts import PromptTemplate

# --- Configuration ---
st.set_page_config(page_title="SDV Fleet Command", page_icon="🚗", layout="wide")
st.title("🚗 Supervision de Flotte SDV & Maintenance Prédictive")
st.markdown("Interface connectée à l'API XGBoost et au module d'assistance LLM (LangChain + Hugging Face).")

API_TIMEOUT = 30
FASTAPI_URL = "http://127.0.0.1:8000/predict"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "openai/gpt-oss-120b:fastest"

try:
    HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.environ.get("HF_API_TOKEN", ""))
except Exception:
    HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

if not HF_API_TOKEN:
    st.sidebar.warning("⚠️ Aucune clé Hugging Face configurée.")

st.session_state.setdefault("api_response", None)
st.session_state.setdefault("messages", [])

# --- Sidebar : télémétrie ---
st.sidebar.header("📡 Télémétrie du véhicule")
veh_id = st.sidebar.text_input("ID Véhicule", "RENAULT-SDV-001")
temp = st.sidebar.slider("Température Moteur (°C)", 70.0, 130.0, 95.0, 0.5)
vib = st.sidebar.slider("Vibration (G)", 0.5, 5.0, 1.2, 0.1)

# --- Analyse FastAPI / XGBoost ---
if st.sidebar.button("🔍 Analyser l'état du véhicule", use_container_width=True):
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Analyse terminée. Avez-vous besoin de détails sur la procédure de remplacement ou l'outillage ?"
    }]
    payload = {
        "vehicle_id": veh_id,
        "engine_temp": temp,
        "battery_voltage": 12.4,
        "vibration": vib,
        "vibration_delta": vib - 1.0,
        "temp_rolling_avg": 90.0,
    }
    resp = None
    try:
        with st.spinner("Analyse XGBoost..."):
            resp = requests.post(FASTAPI_URL, json=payload, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "failure_probability" in data and "alert_status" in data:
            st.session_state.api_response = data
            st.sidebar.success("✅ Analyse terminée")
        else:
            st.error(f"Réponse API incomplète : {data}")
            st.session_state.api_response = None
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de contacter FastAPI.\n\nLance : `uvicorn main:app --reload --port 8000`")
        st.session_state.api_response = None
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout de l'API FastAPI.")
        st.session_state.api_response = None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Erreur HTTP FastAPI : {e}\n\n{resp.text if resp is not None else ''}")
        st.session_state.api_response = None
    except ValueError:
        st.error("❌ La réponse FastAPI n'est pas un JSON valide.")
        st.session_state.api_response = None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur réseau : {e}")
        st.session_state.api_response = None

# --- Affichage ---
if st.session_state.api_response:
    response = st.session_state.api_response
    failure_probability = response.get("failure_probability", 0)
    alert_status = response.get("alert_status", "UNKNOWN")

    st.subheader("🧠 Diagnostic IA — XGBoost")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilité de défaillance (48h)", f"{failure_probability}%")
    col2.metric("Température moteur", f"{temp:.1f} °C")
    col3.metric("Vibration", f"{vib:.2f} G")

    if alert_status == "CRITICAL":
        st.error("🚨 ALERTE CRITIQUE : Maintenance requise")
    elif alert_status in ("WARNING", "WARN"):
        st.warning("⚠️ ALERTE : Surveillance recommandée")
    else:
        st.success("✅ Véhicule sain")

    # --- Copilote technique ---
    if alert_status == "CRITICAL":
        st.markdown("---")
        st.subheader("🤖 Diagnostic IA & Copilote Technique")
        st.warning("🔍 Recherche dans la documentation constructeur...")
        st.info(f"""
**Diagnostic automatique — Manuel SDV-2026, Section 4.2**

- Température moteur : **{temp:.1f} °C**
- Vibration : **{vib:.2f} G**
- État : **CRITICAL**

Hypothèse principale : défaillance de la pompe à eau.
Référence pièce : `#RT-890`
Action recommandée : inspection du circuit de refroidissement avant remplacement.
""")

        st.markdown("### 💬 Questions de suivi")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ex : Quels outils utiliser pour la pièce #RT-890 ?")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            fallback_response = (
                "⚠️ Mode hors-ligne : Hugging Face est indisponible.\n\n"
                "Pour la pompe à eau (#RT-890), prévoyez un jeu de clés Torx "
                "et l'outillage adapté à la procédure constructeur."
            )
            bot_response = fallback_response

            with st.chat_message("assistant"):
                with st.spinner("🤖 L'IA consulte la documentation..."):
                    if not HF_API_TOKEN:
                        st.warning("Token Hugging Face absent.")
                    else:
                        template = """
Tu es un expert mécanicien automobile spécialisé dans les véhicules Software Defined Vehicle (SDV).
Tu assistes un technicien de maintenance Renault.

Contexte :
- Température moteur : {temp} °C
- Vibration : {vib} G
- État : CRITICAL
- Composant suspect : pompe à eau
- Référence : #RT-890

Question du technicien :
{question}

Consignes : réponds uniquement en français, sois technique et très concis,
donne des étapes pratiques si nécessaire, n'invente pas d'informations constructeur.
"""
                        final_prompt = PromptTemplate(
                            template=template, input_variables=["temp", "vib", "question"]
                        ).format(temp=temp, vib=vib, question=question)

                        headers = {
                            "Authorization": f"Bearer {HF_API_TOKEN}",
                            "Content-Type": "application/json",
                        }
                        api_payload = {
                            "model": HF_MODEL,
                            "messages": [
                                {"role": "system", "content": "Tu es un expert mécanicien automobile SDV. Réponds uniquement en français."},
                                {"role": "user", "content": final_prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 250,
                            "stream": False,
                        }

                        try:
                            api_response = requests.post(
                                HF_API_URL, headers=headers, json=api_payload, timeout=API_TIMEOUT
                            )
                            if api_response.status_code != 200:
                                st.error(f"❌ Erreur Hugging Face\n\nHTTP {api_response.status_code}\n\n{api_response.text}")
                            else:
                                result = api_response.json()
                                bot_response = result["choices"][0]["message"]["content"].strip() or fallback_response
                        except requests.exceptions.Timeout:
                            st.error("⏱️ Timeout Hugging Face.")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Impossible de contacter Hugging Face.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Erreur Hugging Face : {e}")
                        except (KeyError, IndexError, TypeError, ValueError) as e:
                            st.error(f"❌ Réponse Hugging Face invalide : {e}")

                st.markdown(bot_response)

            st.session_state.messages.append({"role": "assistant", "content": bot_response})